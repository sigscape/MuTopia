
from math import prod
from .mode_config import ModeConfig
from itertools import product
from ..model import *
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
from pyfaidx import Fasta
import subprocess
from ..utils import stream_subprocess_output
from ..genome_utils.bed12_utils import stream_bed12
from ..plot.signature_plot import _plot_linear_signature
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


class FragmentMotif(ModeConfig):

    MODE_ID='fragment-motif'

    @property
    def coords(self):
        return {'context' : CONTEXTS}
    
    @property
    def make_model(self):
        return MotifModel
    
    @property
    def sample_params(self):
        return _sample_params
    
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
        return np.expand_dims(np.array(comps), axis=-1)

    @classmethod
    def plot(
        cls,
        signature,
        palette = COLOR_LIST,
        select = ['Baseline'],
        **kwargs,
    ):
        cls.validate_signatures(
            signature,
            required_dims=('context',),
        )
        signature = signature.transpose(...,'context',)
        lead_dim = signature.dims[0]
        
        _plot_linear_signature(
            CONTEXTS,
            palette,
            *list(map(
                lambda s : s.ravel(),
                signature.loc[{lead_dim : list(select)}].data
            )),
            **kwargs
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
        
        
        with tempfile.NamedTemporaryFile('w') as query_file, \
            pysam.AlignmentFile(bam_file, 'rb') as bam:

            write_record = partial(print, sep = '\t', file = query_file)

            for record in filter(
                lambda r : not r.is_unmapped, 
                bam
            ):
                
                if record.is_forward:
                    write_record(
                        record.reference_name,
                        record.reference_start,
                        record.reference_start+1,
                        '+',
                        *[record.get_tag(tag) for tag in weight_tags],
                    )
                else:
                    write_record(
                        record.reference_name,
                        record.reference_end-1,
                        record.reference_end,
                        '-',
                        *[record.get_tag(tag) for tag in weight_tags],
                    )

        intersect_process = subprocess.Popen(
            ['bedtools',
            'intersect',
            '-a', regions_file, 
            '-b', query_file.name, 
            '-sorted',
            '-wa',
            '-wb',
            '-split'],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )
        
        n_tags = len(weight_tags)
        n_success = 0; n_failure = 0

        obs_matrix = np.zeros(
            (self.dims['context'], dim_sizes['locus']),
            dtype = np.float32,
        )

        with stream_subprocess_output(intersect_process) as lines, \
            Fasta(fasta_file) as fa:
            
            for line in lines:
                
                fields = line.strip().split('\t')

                locus_idx = int(fields[3])
                contig = fields[0]

                try:
                    read_direction, *tags = fields[-(n_tags + 1):]

                    pos = int(fields[-(n_tags - 3)])
                    is_rev = read_direction == '-'
                    weight = prod(map(float, tags))

                except ValueError as err:
                    n_failure+=1
                    logger.warning(f'Error while parsing record: {line}:\n\t' + str(err).strip())
                    continue
                else:
                    n_success+=1
                
                rslop = 4 * (in_motifs ^ is_rev)
                lslop = -4 * (in_motifs ^ (not is_rev))

                kmer = fa[contig][pos+lslop : pos+rslop].seq.upper()

                if not 'N' in kmer:
                    context_idx = self.coords['context'].index(kmer)
                    obs_matrix[context_idx, locus_idx] += weight
                    n_success+=1
                else:
                    n_failure+=1
        
        logger.info(f'Processed {n_success} records successfully, {n_failure} failed')

        return xr.DataArray(
            obs_matrix,
            dims = ('context', 'locus'),
        )


    def _get_context_frequencies( 
        self,
        lslop : int, 
        rslop : int,
        kmer_width : int =4,
        *,
        regions_file,
        fasta_file
    ):
        def _get_window_seq(fasta_object, chrom, start, end):
            return fasta_object[chrom][max(start-lslop,0) : end+rslop].seq.upper()

        def _rolling(seq, w = 4):
                for i in range(len(seq) - w + 1):
                    yield seq[ i : i+w ]

        def _reduce_count(counts, seq):
            if not 'N' in counts:
                counts[seq]+=1
            else:
                counts['N']+=1
            return counts

        def _count_trinucs(bed12_region, fasta_object):
            
            segments = map(lambda x : _get_window_seq(fasta_object, *x), bed12_region.segments())
            trinucs = chain.from_iterable(map(partial(_rolling, w=kmer_width), segments))
            counts = reduce(_reduce_count, trinucs, Counter())

            N_counts = counts.pop('N', 0)
            pseudocount = N_counts/len(self.coords['context'])

            return [counts[context]+pseudocount for context in self.coords['context']]
        
        with Fasta(fasta_file) as fasta_object:
            trinuc_matrix = [
                _count_trinucs(w, fasta_object)/(len(w) + rslop + lslop - kmer_width + 1)*len(w) 
                for w in tqdm(
                    stream_bed12(regions_file), 
                    nrows=100, 
                    desc = 'Aggregating k-mer content'
                )
            ]

        # LxDxC => DxCxL
        trinuc_matrix = np.array(trinuc_matrix)\
            .astype(np.float32).T
        # DON'T (!) add a pseudocount

        return xr.DataArray(
            trinuc_matrix,
            dims = ('context', 'locus'),
        )
    

class InFragmentMotif(FragmentMotif):

    def get_context_frequencies(self,*, regions_file, fasta_file):
        return self._get_context_frequencies(
            -3, 3, 
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
    ):
        return self._ingest_observations(
            bam_file,
            weight_tags,
            in_motifs = True,
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
    n_components=15,
    init_components=None,
    seed=0,
    # context model
    context_reg=0.0001,
    conditioning_alpha=5e-5,
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
    smoothing_size=1000,
    add_corpus_intercepts=True,
    context_encoder='diagonal',
    kmer_reg=0.001,
    l2_regularization=1,
    # optimization settings
    empirical_bayes = True,
    begin_prior_updates = 10,
    stop_condition=50,
    num_epochs = 2000,
    locus_subsample = None,
    threads = 1,
    kappa = 0.5,
    tau = 1.,
    callback=None,
    eval_every=10,
    verbose=0,
):
    random_state = np.random.RandomState(seed)

    logger.info('Initializing model parameters and transformations...')
    theta_model = \
        (GBTThetaModel if locus_model_type == 'gbt' \
        else LinearThetaModel)\
        (
            train_corpuses,
            n_components=n_components,
            tree_learning_rate=tree_learning_rate,
            max_depth=max_depth,
            max_trees_per_iter=max_trees_per_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_iter_no_change=n_iter_no_change,
            use_groups=use_groups,
            random_state=random_state,
            smoothing_size=smoothing_size,
            #add_corpus_intercepts=add_corpus_intercepts,
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
        n_components=n_components,
        random_state=random_state,
        kmer_encoder=kmer_encoder,
        kmer_reg=kmer_reg,
        tol=5e-4,
        reg=context_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
    )

    locals_model = LDAUpdateSparse(
        train_corpuses,
        n_components=n_components,
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


def _sample_params(study, trial, extensive=False):

    params = {
        'l2_regularization' : trial.suggest_float('l2_regularization', 1e-5, 100., log=True),
        'max_features' : trial.suggest_categorical('max_features', [0.25, 0.33, 0.5]),
    }

    if params:
        params.update({
            'init_variance' : (
                trial.suggest_float('init_variance_mutation', 0.01, 0.1),
                trial.suggest_float('init_variance_context', 0.01, 0.1),
                trial.suggest_float('init_variance_theta', 0.01, 0.1),
            ),
            'context_reg' : trial.suggest_float('context_reg', 1e-5, 5e-3, log=True),
            'mutation_reg' : trial.suggest_float('mutation_reg', 1e-5, 5e-2, log=True),
            'kmer_reg' : trial.suggest_float('kmer_reg', 1e-4, 5e-2, log=True),
            'conditioning_alpha' : trial.suggest_float('conditioning_alpha', 1e-10, 1e-7, log=True),
            'begin_prior_updates' : trial.suggest_int('begin_prior_updates', 5, 50),
            'tree_learning_rate' : trial.suggest_float('tree_learning_rate', 0.05, 0.2),
            'convolution_width' : trial.suggest_int('convolution_width', 0, 3),
        })
    
    return params