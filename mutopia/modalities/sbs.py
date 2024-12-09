
import json
import numpy as np
import os
from sparse import COO
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
            'context' : CONTEXTS,
            'mutation' : MUTATIONS,
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

        return np.array(comps)

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
        cls.validate_signatures(
            signature,
            required_dims=('context', 'mutation'),
        )
        signature = signature.transpose(...,'context','mutation')
        has_extra_dims = len(signature.dims) > 2

        if len(select) > 0 and not has_extra_dims:
            raise ValueError('`select` is only valid if the signature has one extra dimension to choose from.')

        if len(select) > 0:            
            lead_dim = signature.dims[0]

            if sig_names and not len(select) == len(sig_names):
                raise ValueError('If both `sig_names` and `select` are specified, they must have the same length.')
            
            pl_signatures = list(map(
                lambda s : s.ravel()[MUTOPIA_TO_COSMIC_IDX],
                signature.loc[{lead_dim : list(select)}].data
            ))

            sig_names = sig_names or select

        elif not has_extra_dims:
            pl_signatures = [signature.data.ravel()[MUTOPIA_TO_COSMIC_IDX]]
            select = ['']
        else:
            raise ValueError('If the signature has extra dimensions, `select` must be specified.')
        
        if normalize:
            pl_signatures = [s/s.sum() for s in pl_signatures]
            
        _plot_linear_signature(
            COSMIC_SORT_ORDER,
            'tab10' if len(pl_signatures) > 1 else MUTATION_PALETTE,
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
            pseudocount = N_counts/(2*len(self.coords['context']))
            contexts = self.coords['context']

            return [
                [counts[context]+pseudocount for context in contexts],
                [counts[_revcomp(context)]+pseudocount for context in CONTEXTS]
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
        *,
        dim_sizes,
        regions_file,
        fasta_file,
    ):
        
        coords, data = featurize_mutations(
            vcf_file,
            regions_file,
            fasta_file,
            chr_prefix=chr_prefix,
            weight_col=weight_col,
            mutation_rate_file=mutation_rate_file,
            sample_weight=sample_weight,
            sample_name=sample_name,
            pass_only=pass_only,
        )
        
        return self._arr_to_xr(dim_sizes, coords, data)


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
    conditioning_alpha=1e-9,
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
    smoothing_size=1000,
    add_corpus_intercepts=False,
    convolution_width=1,
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
    sparse=True,
    verbose=0,
    max_iter=25,
    init_variance=(0.1, 0.1, 0.1)
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

    '''kmer_encoder = KmerEncoder(
        ['ACTG','CT','ACTG'],
        kmer_extractor=tuple,
        feature_name_fn=_make_feature_name,
    )'''

    context_model = StrandedContextModel(
        train_corpuses,
        DiagonalEncoder(),
        n_components=num_components,
        random_state=random_state,
        init_variance=init_variance[1],
        tol=5e-4,
        reg=context_reg,
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
            smoothing_size=smoothing_size,
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
        'context_reg' : trial.suggest_float('context_reg', 1e-5, 5e-3, log=True),
        'mutation_reg' : trial.suggest_float('mutation_reg', 1e-5, 5e-3, log=True),
        'max_features' : trial.suggest_float('max_features', 0.1, 1.),
        'locus_subsample' : trial.suggest_categorical('locus_subsample', [None, 0.125, 0.25, 0.5]),
        'l2_regularization' : trial.suggest_float('l2_regularization', 1e-5, 10, log=True),
        #'max_iter' : trial.suggest_categorical('max_iter', [25, 50, 100, 300]),
        'init_variance' : (
            trial.suggest_float('init_variance_mutation', 1e-3, 5e-1, log=True),
            trial.suggest_float('init_variance_context', 1e-3, 5e-1, log=True),
            trial.suggest_float('init_variance_theta', 1e-3, 5e-1, log=True),
        ),
        'convolution_width' : trial.suggest_int('convolution_width', 0, 3),
    }