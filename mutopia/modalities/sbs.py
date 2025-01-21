
import json
import numpy as np
import os
from functools import reduce
from itertools import chain
from collections import Counter
from tqdm import tqdm
from pyfaidx import Fasta
import xarray as xr
from ..plot.signature_plot import _plot_linear_signature
from ..model import *
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
        return _sample_params
    
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
                    .reshape( (cls.dim_context(), cls.dim_mutation()) )
            )

        return xr.DataArray(
            np.array(comps),
            dims=('component', 'context', 'mutation'),
        )
    

    @classmethod
    def _flatten_observations(cls, signature):
        
        cls.validate_signatures(
            signature,
            required_dims=('context', 'mutation'),
        )
        signature = signature.transpose(...,'context','mutation')

        signature = signature.stack(observation=('context','mutation'))\
            .isel(observation=MUTOPIA_TO_COSMIC_IDX)
        
        signature = signature.reset_index('observation')\
            .assign_coords(observation=\
                signature\
                    .indexes['observation']\
                    .map(lambda x : format_as_cosmic(*x))
            )
        
        return signature


    @classmethod
    def plot(cls,
        signature,
        *select,
        palette = 'tab10',
        sig_names = None,
        normalize = False,
        title=None,
        **kwargs,
    ):
        
        signature = cls._flatten_observations(signature)
        pl_signatures, sig_names = cls._parse_signatures(
            signature,
            select=select,
            sig_names=sig_names,
            normalize=normalize,
        )
            
        _plot_linear_signature(
            COSMIC_SORT_ORDER,
            palette if len(pl_signatures) > 1 else MUTATION_PALETTE,
            *pl_signatures,
            sig_names=sig_names,
            **kwargs
        )
    
    
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
    kmer_reg=0.005,
    conditioning_alpha=1e-9,
    context_encoder='diagonal',
    # mutation model
    mutation_reg=0.0005,
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
    convolution_width=2,
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
    sparse=True,
    verbose=0,
    max_iter=25,
    init_variance=(0.1, 0.1, 0.05)
):
    
    random_state = np.random.RandomState(seed)
    
    logger.info('Initializing model parameters and transformations...')
    
    mutation_model = StrandedConditionalConsequenceModel(
        'mutation', # require the mutation dimension - this is the stranded conditional consequence
        train_corpuses,
        n_components=num_components,
        init_variance=init_variance[0],
        random_state=random_state,
        tol=5e-4,
        reg=mutation_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
        max_iter=max_iter,
    )

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
        init_variance=init_variance[1],
        tol=5e-4,
        reg=context_reg,
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
            init_variance=init_variance[2],
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
        mutation_model=mutation_model,
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


def _sample_params(study, trial, extensive=0):

    params = {
        'l2_regularization' : trial.suggest_float('l2_regularization', 1e-5, 1000., log=True),
        'max_features' : trial.suggest_categorical('max_features', [0.25, 0.33, 0.5, 0.75, 1.]),
        'tree_learning_rate' : trial.suggest_float('tree_learning_rate', 0.025, 0.2),
        'init_variance' : (0.1, 0.1, trial.suggest_float('init_variance_theta', 0.01, 0.1)),
    }

    if extensive>0:
        params.update({
            'context_reg' : trial.suggest_float('context_reg', 1e-5, 5e-3, log=True),
            'mutation_reg' : trial.suggest_float('mutation_reg', 1e-5, 5e-2, log=True),
        })

    if extensive>1:
        params.update({
            'init_variance' : (
                trial.suggest_float('init_variance_mutation', 0.01, 0.1),
                trial.suggest_float('init_variance_context', 0.01, 0.1),
                trial.suggest_float('init_variance_theta', 0.01, 0.1),
            ),
            'context_encoder' : trial.suggest_categorical('context_encoder', ['diagonal', 'kmer']),
            'kmer_reg' : trial.suggest_float('kmer_reg', 1e-4, 5e-2, log=True),
            'conditioning_alpha' : trial.suggest_float('conditioning_alpha', 1e-10, 1e-7, log=True),
            'convolution_width' : trial.suggest_int('convolution_width', 0, 3),
        })

    if extensive>2:
        params.update({
            'batch_subsample' : trial.suggest_categorical('batch_subsample', [None, 0.0625, 0.125, 0.25, 0.5]),
            'locus_subsample' : trial.suggest_categorical('locus_subsample', [None, 0.0625, 0.125, 0.25, 0.5]),
        })
    
    return params
