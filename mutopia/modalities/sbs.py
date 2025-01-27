
import json
import numpy as np
import os
from functools import reduce
from itertools import chain
from collections import Counter
from tqdm import tqdm
from pyfaidx import Fasta
import xarray as xr
from ..model import *
from ..tuning import sample_params
from ..utils import logger
from ..genome_utils.bed12_utils import stream_bed12
from .mode_config import ModeConfig
from ._sbs_nucdata import *
from ._sbs_ingestion import featurize_mutations, _revcomp


_transition_palette = {
    ('C','A') : (0.33, 0.75, 0.98),
    ('C','G') : (0.0, 0.0, 0.0),
    ('C','T') : (0.85, 0.25, 0.22),
    ('T','A') : (0.78, 0.78, 0.78),
    ('T','C') : (0.51, 0.79, 0.24),
    ('T','G') : (0.89, 0.67, 0.72)
}

MUTATION_PALETTE = [color for color in _transition_palette.values() for i in range(16)]

MUTOPIA_TO_COSMIC_IDX = np.array([
    MUTOPIA_ORDER.index(cosmic)
    for cosmic in COSMIC_SORT_ORDER
])


class SBSMode(ModeConfig):

    MODE_ID = 'sbs'
    MUTOPIA_TO_COSMIC_IDX = MUTOPIA_TO_COSMIC_IDX
    PALETTE = MUTATION_PALETTE

    @property
    def coords(self):
        return {
            'clustered' : ['no','yes'],
            'configuration' : CONFIGURATIONS,
            'context' : MUTOPIA_ORDER,
        }
    
    @property
    def make_model(self):
        return SBSModel
    
    @property
    def sample_params(self):
        return sample_params
    
    @property
    def available_components(self):
        filepath = os.path.join(os.path.dirname(__file__), 'musical_sbs.json')
        with open(filepath, 'r') as f:
            database = json.load(f)
            
        return list(database.keys())


    @classmethod
    def load_components(cls, *init_components):
        
        filepath = os.path.join(os.path.dirname(__file__), 'musical_sbs.json')
        with open(filepath, 'r') as f:
            database = json.load(f)

        comps = []
        for component in init_components:
            if not component in database:
                raise ValueError(f"Component {component} not found in database")
            comps.append(
                np.array([database[component][context_mut] for context_mut in MUTOPIA_ORDER])\
                    .reshape( (cls().sizes['context'], cls().sizes['mutation']) )
            )

        return xr.DataArray(
            np.array(comps),
            dims=('component', 'context', 'mutation'),
        )
    
    @classmethod
    def unstack(cls, corpus):

        from ..corpus import update_view

        corpus = update_view(
            corpus,
            regions=corpus.regions.assign_coords({'mutation' : corpus.coords['mutation']}).to_dataset(),
        )
        corpus.regions.context_frequencies = corpus.regions.context_frequencies.expand_dims({'mutation' : 3})

        stacked = corpus.drop_nodes(('features',))\
                .stack(observation=('context','mutation'))
        
        stacked = stacked.assign_coords(
            observation=\
                stacked\
                    .indexes['observation']\
                    .map(lambda x : format_as_cosmic(*x))
        )

        stacked = stacked.rename({'observation' : 'context'})
        corpus = update_view(
            stacked,
            features = corpus.features.to_dataset(),
        )

        return corpus
    

    @classmethod
    def _flatten_observations(cls, signature):
        signature = super()\
            ._flatten_observations(signature)\
            .isel(context=MUTOPIA_TO_COSMIC_IDX)
        
        return signature
    
    
    def get_context_frequencies(
        self,
        n_jobs = 1,
        *,
        regions_file,
        fasta_file,
    ):
        
        def _get_window_seq(fasta_object, chrom, start, end):
            return fasta_object[chrom][max(start-1,0) : end+1].seq.upper()


        def _rolling(seq, w = 3):
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
            trinucs = chain.from_iterable(map(_rolling, segments))
            counts = reduce(_reduce_count, trinucs, Counter())

            N_counts = counts.pop('N', 0)
            pseudocount = N_counts/(2*len(CONTEXTS))
            contexts = CONTEXTS

            return [
                [counts[context]+pseudocount for context in contexts],
                [counts[_revcomp(context)]+pseudocount for context in contexts]
            ]
        
        with Fasta(fasta_file) as fasta_object:
            trinuc_matrix = [
                _count_trinucs(w, fasta_object) 
                for w in tqdm(
                    stream_bed12(regions_file), 
                    nrows=100, 
                    desc = 'Aggregating trinucleotide content'
                )
            ]

        # LxDxC => DxCxL
        trinuc_matrix = np.array(trinuc_matrix)\
            .transpose(((1,2,0)))\
            .astype(np.float32) 
        # DON'T (!) add a pseudocount

        # 2xCxL => 2xCx3xL => 2x(C*3)xL
        trinuc_matrix = np.expand_dims(trinuc_matrix, axis=2)
        trinuc_matrix = np.repeat(trinuc_matrix, 3, axis=2).reshape((2, -1, trinuc_matrix.shape[-1]))

        return xr.DataArray(
            trinuc_matrix,
            dims = ('configuration', 'context', 'locus'),
        )


    def ingest_observations(
        self,
        vcf_file,
        chr_prefix='',
        pass_only=True,
        weight_col = None, 
        mutation_rate_file=None,
        sample_weight=None,
        sample_name=None,
        skip_sort=False,
        cluster=True,
        *,
        locus_dim,
        regions_file,
        fasta_file,
        **kw,
    ):
        return featurize_mutations(
            vcf_file,
            regions_file,
            fasta_file,
            chr_prefix=chr_prefix,
            weight_col=weight_col,
            mutation_rate_file=mutation_rate_file,
            sample_weight=sample_weight,
            sample_name=sample_name,
            pass_only=pass_only,
            skip_sort=skip_sort,
            cluster=cluster,
            locus_dim=locus_dim,
        )


def _make_feature_name(subsequence_code):
    default = {0 : 'N', 1 : 'N', 2 : 'N'}
    default.update(subsequence_code)
    return ''.join([default[i] for i in range(3)])


def SBSModel(
    train_corpuses,
    test_corpuses,
    num_components=15,
    init_components=None,
    seed=0,
    # context model
    context_reg=0.0001,
    context_conditioning=1e-5,
    kmer_reg=0.005,
    conditioning_alpha=1e-9,
    context_encoder='diagonal',
    # locals model
    pi_prior=1.,
    # locus model
    locus_model_type='gbt',
    tree_learning_rate=0.15, 
    max_depth=5,
    max_trees_per_iter=25,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    max_features=0.5,
    n_iter_no_change=1,
    use_groups=True,
    add_corpus_intercepts=False,
    convolution_width=2,
    l2_regularization=1,
    # optimization settings
    empirical_bayes=True,
    begin_prior_updates=15,
    stop_condition=50,
    num_epochs = 2000,
    locus_subsample=None,
    batch_subsample=None,
    threads=1,
    kappa=0.5,
    tau=1,
    callback=None,
    eval_every=1,
    verbose=0,
    max_iter=25,
    init_variance_theta=0.05,
    init_variance_context=0.1,
):
    
    random_state = np.random.RandomState(seed)
    
    logger.info('Initializing model parameters and transformations...')
    
    if context_encoder == 'diagonal':
        kmer_encoder = DiagonalEncoder()
    elif context_encoder == 'kmer':
        kmer_encoder = KmerEncoder(
            ['ACTG','CT','ACTG'],
            kmer_extractor=tuple,
            feature_name_fn=_make_feature_name,
        )
    else:
        raise ValueError(f'Unknown context encoder: {context_encoder}')

    context_model = StrandedContextModel(
        train_corpuses,
        kmer_encoder,
        n_components=num_components,
        random_state=random_state,
        init_variance=init_variance_context,
        tol=5e-4,
        reg=context_reg,
        context_conditioning=context_conditioning,
        kmer_reg=kmer_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
        max_iter=max_iter,
    )

    theta_model = \
        (GBTThetaModel if locus_model_type == 'gbt' \
        else LinearThetaModel)\
        (
            train_corpuses,
            init_variance=init_variance_theta,
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

    sparse = train_corpuses[0].X.is_sparse()
    if not all(corpus.X.is_sparse() == sparse for corpus in train_corpuses + test_corpuses):
        raise ValueError('All corpuses must be either sparse or dense - mixtures are not allowed!')
    
    locals_model = \
        (LDAUpdateSparse if sparse else LDAUpdateDense)(
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
            np.random.RandomState(seed),
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
