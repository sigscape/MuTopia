import sys
from math import prod
from .mode_config import ModeConfig
from itertools import product, chain, starmap
from ..model import *
from ..tuning import sample_params
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from functools import reduce, partial
from itertools import chain
from collections import Counter
from tqdm import tqdm
from pyfaidx import Fasta
import xarray as xr
import tempfile
import subprocess
from ..utils import stream_subprocess_output, logger
from ..genome_utils.fancy_iterators import streaming_local_sort
from ..genome_utils.bed12_utils import stream_bed12
logger = logging.getLogger(' Mutopia-MotifModel ')

CONTEXTS = sorted(
    map(lambda x : ''.join(x), product('ATCG','ATCG','ATCG', 'ATCG')), 
    key = lambda x : (x[0], x[1], x[2], x[3])
)

cmap = plt.colormaps['tab10']

_context_palette = {
    'A': cmap(0.3),
    'C': cmap(0.2),
    'G': cmap(0.9),
    'T': cmap(0.4)
}

ALPHA_LIST = [0.5 if fourmer[1] in ['A','G'] else 1 for fourmer in CONTEXTS]
COLOR_LIST = [_context_palette[fourmer[0]] for fourmer in CONTEXTS]
for i in range(len(COLOR_LIST)):
    color_tuple = list(COLOR_LIST[i])
    color_tuple[-1] = ALPHA_LIST[i]
    COLOR_LIST[i] = tuple(color_tuple)


def parse_bamfile(
    bam_file,
    *weight_tags,
    output=sys.stdout,
):

    def _get_fragment_end(record):
        if record.is_forward:
            pos = (
                record.reference_start,
                record.reference_start+1,
                '+'
            )
        else:
            pos = (
                record.reference_end-1,
                record.reference_end,
                '-',
            )
        
        return (
            record.reference_name,
            *pos,
            *[record.get_tag(tag) for tag in weight_tags],
        )

    try:
        import pysam
    except ImportError:
        raise ImportError('"pysam" is required for ingesting observations from BAM files')
    
    with pysam.AlignmentFile(bam_file, 'rb') as bam:
        
        data = filter(
            lambda r : not r.is_unmapped and not r.is_secondary and not r.is_supplementary \
                and not r.is_duplicate and r.is_paired and r.mapping_quality > 0,
            bam
        )

        data = map(_get_fragment_end, data)

        data = streaming_local_sort(
            data,
            key = lambda x : x[1],
            has_lapsed = lambda curr, buffval : \
                curr[0] != buffval[0] or curr[1] - buffval[1] > 10000,
        )

        for record in data:
            print(*record, sep='\t', file=output)
        

class FragmentMotif(ModeConfig):

    PALETTE = COLOR_LIST

    @property
    def coords(self):
        return {'context' : CONTEXTS}
    
    @property
    def make_model(self):
        return MotifModel
    
    @property
    def sample_params(self):
        return sample_params
    
    @property
    def palette(self):
        return COLOR_LIST
    
    @property
    def available_components(self):
        filepath = os.path.join(os.path.dirname(__file__), 'fragment_motifs.json')
        with open(filepath, 'r') as f:
            database = json.load(f)
        return list(database.keys())

    @classmethod
    def load_components(cls, *init_components):
        
        filepath = os.path.join(os.path.dirname(__file__), 'fragment_motifs.json')
        with open(filepath, 'r') as f:
            database = json.load(f)

        comps = []
        for component in init_components:
            if not component in database:
                raise ValueError(f"Component {component} not found in database")
            comps.append(
                [database[component][context_mut] for context_mut in cls().coords['context']]
            )
        
        return xr.DataArray(
            np.array(comps, dtype=float),
            dims = ('component', 'context'),
        )

    
    def _ingest_observations(
        self,
        bam_file,
        weight_tags,
        in_motifs = True,
        *,
        dim_sizes,
        regions_file,
        fasta_file,
    ):
        
        try:
            import pysam
        except ImportError:
            raise ImportError('"pysam" is required for ingesting observations from BAM files')
        
        with tempfile.NamedTemporaryFile('w') as bam_genome, \
            tempfile.NamedTemporaryFile('w') as sorted_regions_file, \
            Fasta(fasta_file) as fa, \
            pysam.AlignmentFile(bam_file, 'rb') as bam:

            # 1. print out the chromosomes in the order defined by the BAM file
            for contig in bam.header.references:
                print(
                    contig, 
                    bam.header.get_reference_length(contig), 
                    sep='\t', 
                    file=bam_genome
                )
            bam_genome.flush()

            # 2. re-sort the regions file to match the order of the BAM file
            subprocess.check_call(
                [
                    'bedtools', 
                    'sort', 
                    '-i', regions_file,
                    '-g', bam_genome.name,
                ],
                stdout=sorted_regions_file,
                universal_newlines=True,
                bufsize=10000,
            )
            sorted_regions_file.flush()

            # 3. parse the BAM file - get the fragment ends,
            #    filter and sort them, and get the final weight
            parse_process = subprocess.Popen(
                [
                    'gtensor-fragments',
                    'parse-bam',
                    bam_file,
                    *weight_tags,
                ],
                stderr=sys.stderr,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
            )

            # 4. intersect the sorted regions with the parsed fragments
            intersect_process = subprocess.Popen(
                [
                'bedtools',
                'intersect',
                '-a', sorted_regions_file.name,
                '-b', 'stdin', 
                '-sorted',
                '-wa',
                '-wb',
                '-split',
                '-g', bam_genome.name,
                ],
                stderr=sys.stderr,
                stdin=parse_process.stdout,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
            )
            
            n_tags = len(weight_tags)
            n_success = 0; n_failure = 0
            
            context_idx_map = {c : j for j,c in enumerate(self.coords['context'])}

            obs_matrix = np.zeros(
                (self.sizes['context'], dim_sizes['locus']),
                dtype = np.float16,
            )

            for line in tqdm(
                stream_subprocess_output(intersect_process),
                desc='Parsing bam records',
                ncols=100,
            ):
                
                fields = line.strip().split('\t')
                
                if len(fields) < 4 + n_tags:
                    raise ValueError(f'Error while parsing record: {line}\nThe input regions file may be malformed')
                
                try:
                    locus_idx = int(fields[3])
                    contig = fields[0]

                    read_direction, *tags = fields[-(n_tags + 1):]

                    pos = int(fields[-(n_tags + 3)])
                    is_rev = read_direction == '-'
                    weight = prod(map(float, tags))

                except ValueError as err:
                    n_failure+=1
                    logger.warning(f'Error while parsing record: {line}:\n\t' + str(err).strip())
                    continue
                
                shift = int(is_rev)
                rslop = 4 * (in_motifs ^ is_rev) + shift
                lslop = -4 * (in_motifs ^ (not is_rev)) + shift

                #              OUT MOTIFS
                #     1:5
                #   * * * *
                #           5____________12
                #                           * * * *
                #                            13:17
                # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6


                #              IN MOTIFS            
                #       2:6
                #     * * * * 
                #     2_____________9
                #             * * * *
                #               6:10
                # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
                
                #               LSLOP,RSLOP
                #       rev  | False | True
                # -----------+-------+-------
                # in_motifs  |       |
                #    True    |  0,4  | -3,1
                #   False    | -4,0  |  1,5

                kmer = fa[contig][pos+lslop : pos+rslop]

                if is_rev:
                    kmer = kmer.reverse.complement.seq.upper()
                else:
                    kmer = kmer.seq.upper()

                if kmer in context_idx_map:
                    context_idx = context_idx_map[kmer]
                    obs_matrix[context_idx, locus_idx] += weight
                    n_success+=1
                else:
                    n_failure+=1
        
        logger.info(f'Processed {n_success} records successfully, {n_failure} failed')

        if n_failure/(n_success + n_failure) > 0.05:
            logger.warning('Quite a lot of fragments failed to be parsed ...')
        
        return xr.DataArray(
            obs_matrix,
            dims = ('context', 'locus'),
        )


    def _get_context_frequencies( 
        self,
        fwd_slop,
        rev_slop,
        kmer_width : int =4,
        *,
        regions_file,
        fasta_file,
    ):
        def _get_window_seq(chrom, start, end):
            return (
                fasta_object[chrom][max(start+fwd_slop[0],0) : end+fwd_slop[1]].seq.upper(),
                fasta_object[chrom][max(start+rev_slop[0],0) : end+rev_slop[1]].reverse.complement.seq.upper(),
            )

        def _rolling(seq, w = 4):
                for i in range(len(seq) - w + 1):
                    yield seq[ i : i+w ]

        def _reduce_count(counts, seq):
            if not 'N' in counts:
                counts[seq]+=0.5
            else:
                counts['N']+=0.5
            return counts

        def _count_trinucs(bed12_region):
            segments = chain.from_iterable(starmap(_get_window_seq, bed12_region.segments()))
            trinucs = chain.from_iterable(map(partial(_rolling, w=kmer_width), segments))
            counts = reduce(_reduce_count, trinucs, Counter())

            N_counts = counts.pop('N', 0)
            pseudocount = N_counts/len(self.coords['context'])

            context_counts = [
                counts[context]+pseudocount
                for context in self.coords['context']
            ]

            return context_counts
        
        
        n_loci = sum(1 for _ in stream_bed12(regions_file))
        
        trinuc_matrix = np.zeros(
            (self.sizes['context'], n_loci),
            dtype = np.float32,
        )

        with Fasta(fasta_file) as fasta_object:
            for i, region in tqdm(
                enumerate(stream_bed12(regions_file)), 
                ncols=100, 
                desc = 'Aggregating k-mer content'
            ):
                trinuc_matrix[...,i] = np.array(_count_trinucs(region), dtype=np.float32)

        return xr.DataArray(
            trinuc_matrix,
            dims = ('context', 'locus'),
        )
    

class InFragmentMotif(FragmentMotif):

    MODE_ID='FRAGMENT_MOTIF_IN5P'

    def get_context_frequencies(self,*, regions_file, fasta_file):
        return self._get_context_frequencies(
            (0,3),
            (-3,0),
            regions_file=regions_file, 
            fasta_file=fasta_file
        )
    
    def ingest_observations(
        self,
        bam_file,
        weight_tags,
        *,
        dim_sizes,
        regions_file,
        fasta_file,
        **kw,
    ):
        return self._ingest_observations(
            bam_file,
            weight_tags,
            in_motifs = True,
            dim_sizes = dim_sizes,
            regions_file = regions_file,
            fasta_file = fasta_file,
        )
    

class OutFragmentMotif(FragmentMotif):

    MODE_ID='FRAGMENT_MOTIF_OUT5P'

    def get_context_frequencies(self,*, regions_file, fasta_file):
        return self._get_context_frequencies(
            (-4,-1),
            (1,4),
            regions_file=regions_file, 
            fasta_file=fasta_file
        )
    
    def ingest_observations(
        self,
        bam_file,
        weight_tags,
        *,
        dim_sizes,
        regions_file,
        fasta_file,
        **kw,
    ):
        return self._ingest_observations(
            bam_file,
            weight_tags,
            in_motifs = False,
            dim_sizes = dim_sizes,
            regions_file = regions_file,
            fasta_file = fasta_file,
        )



def _make_feature_name(subsequence_code):
    default = {0 : 'N', 1 : 'N', 2 : 'N', 3 : 'N'}
    default.update(subsequence_code)
    return ''.join([default[i] for i in range(4)])


def MotifModel(
    train_corpuses,
    test_corpuses,
    num_components=15,
    init_components=None,
    seed=0,
    # context model
    context_reg=0.0001,
    kmer_reg=0.005,
    conditioning_alpha=1e-9,
    context_encoder='diagonal',
    # mutation model
    # locals model
    pi_prior=1.,
    # locus model
    locus_model_type='gbt',
    tree_learning_rate=0.1, 
    max_depth = 5,
    max_trees_per_iter = 25,
    max_leaf_nodes = 31,
    min_samples_leaf = 30,
    max_features = 0.5,
    n_iter_no_change=1,
    use_groups=True,
    add_corpus_intercepts=False,
    convolution_width=1,
    l2_regularization=1,
    # optimization settings
    empirical_bayes = True,
    begin_prior_updates = 20,
    stop_condition=50,
    num_epochs = 2000,
    locus_subsample=None,
    batch_subsample=None,
    threads = 1,
    kappa = 0.5,
    tau = 0,
    callback=None,
    eval_every=1,
    verbose=0,
    max_iter=25,
    init_variance=(0.1, 0.05),
):
    random_state = np.random.RandomState(seed)

    logger.info('Initializing model parameters and transformations...')
    theta_model = \
        (GBTThetaModel if locus_model_type == 'gbt' \
        else LinearThetaModel)\
        (
            train_corpuses,
            init_variance=init_variance[1],
            n_components=num_components,
            tree_learning_rate=tree_learning_rate,
            max_depth=max_depth,
            max_trees_per_iter=max_trees_per_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_iter_no_change=n_iter_no_change,
            use_groups=use_groups,
            random_state=random_state,
            add_corpus_intercepts=add_corpus_intercepts,
            convolution_width=convolution_width,
            l2_regularization=l2_regularization,
        )
    
    if context_encoder == 'diagonal':
        kmer_encoder = DiagonalEncoder()
    elif context_encoder == 'kmer':
        kmer_encoder = KmerEncoder(
            ['ATCG','ATCG','ATCG', 'ATCG'],
            kmer_extractor=tuple,
            feature_name_fn=_make_feature_name,
        )
    else:
        raise ValueError(f'Unknown context encoder: {context_encoder}')

    context_model = UnstrandedContextModel(
        train_corpuses,
        kmer_encoder,
        n_components=num_components,
        random_state=random_state,
        init_variance=init_variance[0],
        tol=5e-4,
        reg=context_reg,
        kmer_reg=kmer_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
        max_iter=max_iter,
    )

    locals_model = LDAUpdateDense(
        train_corpuses,
        n_components=num_components,
        random_state=random_state,
        prior_alpha=pi_prior,
    )

    model_state = ModelState(
        train_corpuses,
        context_model=context_model,
        theta_model=theta_model,
        locals_model=locals_model,
    )

    (model_state, train_scores, test_scores) = \
        fit_model(
            train_corpuses,
            test_corpuses,
            model_state,
            random_state,
            empirical_bayes=empirical_bayes,
            begin_prior_updates=begin_prior_updates,
            stop_condition=stop_condition,
            num_epochs=num_epochs,
            locus_subsample=locus_subsample,
            batch_subsample=batch_subsample,
            threads=threads,
            kappa=kappa,
            tau=tau,
            callback=callback,
            eval_every=eval_every,
            verbose=verbose,
        )

    return (
        Model(
            model_state, 
            train_corpuses[0].modality()
        ),
        train_scores,
        test_scores,
    )
