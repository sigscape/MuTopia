import numpy as np
import xarray as xr
import tempfile
import subprocess
from math import prod
import os
import json
import logging
from itertools import starmap
import sys
from .mode_config import ModeConfig
from ..model import *
from ..genome_utils.bed12_utils import stream_bed12
from ..utils import stream_subprocess_output, logger
from ..genome_utils.fancy_iterators import streaming_local_sort, streaming_groupby

_bin_edges = [
    (70, 80),  (80, 90),  (90, 100), (100, 105), (105, 110),
    (110, 115), (115, 120), (120, 125), (125, 130), (130, 135), (135, 140), (140, 145),
    (145, 150), (150, 155), (155, 160), (160, 165), (165, 170), (170, 175), (175, 180),
    (180, 185), (185, 190), (190, 195), (195, 200), (200, 205), (205, 210), (210, 220),
    (220, 230), (230, 240), (240, 250), (250, 260), (260, 270), (270, 280), (280, 290),
    (290, 300), (300, 310), (310, 320), (320, 330), (330, 340), (340, 350), (350, 700)
]
LENGTH_BINS = [f'{l}-{r}' for l, r in _bin_edges]



def bam_to_fragments(
    bam_file,
    *weight_tags,
    output=sys.stdout,
):
    
    def to_fragments(read1, read2): 

        fwd, rev = (read1, read2) if read1.is_forward else (read2, read1)
        start, end = fwd.reference_start, rev.reference_end

        return (
            read1.reference_name,
            fwd.reference_start,
            fwd.reference_start+1,
            start,
            end,
            read1.query_name,
            end-start,
            *[read1.get_tag(tag) for tag in weight_tags],
        )

    try:
        import pysam
    except ImportError:
        raise ImportError('"pysam" is required for ingesting observations from BAM files')
    
    with pysam.AlignmentFile(bam_file, 'rb') as bam:

        logger.info('Parsing BAM file ...')
        
        data = filter(
            lambda r : not r.is_unmapped and not r.is_secondary and not r.is_supplementary \
                and not r.is_duplicate \
                and r.is_paired and r.mapping_quality > 0,
            bam
        )

        data = streaming_groupby(
            data,
            groupby_key=lambda r : r.query_name,
            has_lapsed = lambda curr, group: \
                len(group)==2 or \
                curr.reference_start > (group[-1].reference_start + 10000) or \
                curr.reference_name != group[-1].reference_name,
        )

        # get rid of the key
        data = map(lambda x : x[1], data)

        # filter out unpaired reads and discontinuous fragments
        data = filter(lambda g : len(g) == 2, data)

        # make into fragments
        data = starmap(to_fragments, data)

        # reorder the fragments
        data = streaming_local_sort(
            data,
            key = lambda x : x[1],
            has_lapsed = lambda curr, buffval : \
                curr[0] != buffval[0] or curr[1] - buffval[1] > 10000,
        )

        for record in data:
            print(*record, sep='\t', file=output)



class FragmentLength(ModeConfig):

    MODE_ID='FRAGMENT_LENGTH'
    PALETTE='lightgrey'

    @property
    def coords(self):
        return {'context' : LENGTH_BINS}
    
    @property
    def make_model(self):
        return FragmentLengthModel
    
    @property
    def sample_params(self):
        return _sample_params
    
    @property
    def available_components(self):
        filepath = os.path.join(os.path.dirname(__file__), 'fragment_lengths.json')
        with open(filepath, 'r') as f:
            database = json.load(f)
            
        return list(database.keys())


    @classmethod
    def load_components(cls, *init_components):
        
        filepath = os.path.join(os.path.dirname(__file__), 'fragment_lengths.json')
        with open(filepath, 'r') as f:
            database = json.load(f)

        comps = []
        for component in init_components:
            if not component in database:
                raise ValueError(f"Component {component} not found in database")
            
            comps.append([database[component][l] for l in cls().coords['context']])
        
        return xr.DataArray(
            np.array(comps, dtype=float),
            dims = ('component', 'context'),
        )
    

    def ingest_observations(
        self,
        bam_file : str,
        weight_tags : list,
        *,
        dim_sizes : dict,
        regions_file : str,
        **kw,
    ):
        try:
            import pysam
        except ImportError:
            raise ImportError('"pysam" is required for ingesting observations from BAM files')
        
        with tempfile.NamedTemporaryFile('w') as bam_genome, \
            tempfile.NamedTemporaryFile('w') as sorted_regions_file, \
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
                    'bam-to-fragments',
                    bam_file,
                    *weight_tags,
                ],
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
                stdin=parse_process.stdout,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
            )
            
            n_tags = len(weight_tags)
            n_success = 0; n_failure = 0

            obs_matrix = np.zeros(
                (self.sizes['context'], dim_sizes['locus']),
                dtype = np.float16,
            )

            bins_flat = np.array([l for l, r in _bin_edges] + [_bin_edges[-1][-1]])

            for line in stream_subprocess_output(intersect_process):
                
                fields = line.strip().split('\t')
                
                if len(fields) < 4 + n_tags:
                    raise ValueError(f'Error while parsing record: {line}\nThe input regions file may be malformed')
                
                try:
                    locus_idx = int(fields[3])
                    contig = fields[0]
                    tags = fields[-n_tags:]
                    length = int(fields[-(n_tags + 1)])
                    weight = prod(map(float, tags))

                except ValueError as err:
                    n_failure+=1
                    logger.warning(f'Error while parsing record: {line}:\n\t' + str(err).strip())
                    continue
                else:
                    n_success+=1
                
                try:
                    bin_idx = np.digitize(length, bins_flat) - 1
                    obs_matrix[bin_idx, locus_idx] += weight
                    n_success+=1
                except IndexError:
                    n_failure+=1
        
        logger.info(f'Processed {n_success} records successfully, {n_failure} failed')

        if n_success==0:
            raise ValueError(
                'No records were successfully processed.\n'
                'Please check that:\n'
                '* 1. The BAM file is valid, contains reads, and position-sorted\n'
                '* 2. The BAM file contains paired-end reads, since mate information is needed to get the fragment length.\n'
                '* 3. The fragment size is in the expected range (70-500bp).\n'
            )
        
        return xr.DataArray(
            obs_matrix,
            dims = ('context', 'locus'),
        )


    def get_context_frequencies(
        self,
        n_jobs = 1,
        *,
        regions_file,
        fasta_file,
    ):
        region_lengths = list(len(r) for r in stream_bed12(regions_file))
        region_lengths = np.array(region_lengths).astype(np.float32) 
        
        return xr.DataArray(
            region_lengths,
            dims = ('locus',),
        )



def FragmentLengthModel(
    train_corpuses,
    test_corpuses,
    n_components=15,
    init_components=None,
    seed=0,
    # context model
    reg=0.0001,
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
    sparse=True
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

    fraglength_model = UnstrandedContextModel(
        train_corpuses,
        n_components=n_components,
        random_state=random_state,
        tol=5e-4,
        reg=reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
    )


    locals_model = \
        (LDAUpdateSparse if sparse else LDAUpdateDense)(
            train_corpuses,
            n_components=n_components,
            random_state=random_state,
            prior_alpha=pi_prior,
        )

    model_state = ModelState(
        train_corpuses,
        fraglength_model=fraglength_model,
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


def _sample_params(study, trial):
    return {
        'reg' : trial.suggest_float('context_reg', 1e-5, 5e-3, log=True),
        'conditioning_alpha' : trial.suggest_float('conditioning_alpha', 1e-6, 1e-3, log=True),
        'empirical_bayes' : trial.suggest_categorical('empirical_bayes', [True, False]),
        'max_features' : trial.suggest_float('max_features', 0.1, 1.),
        'locus_subsample' : trial.suggest_categorical('locus_subsample', [None, 0.125, 0.25, 0.5]),
        'kappa' : trial.suggest_float('kappa', 0.5, 0.9),
    }