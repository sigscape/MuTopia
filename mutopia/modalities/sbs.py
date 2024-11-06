
import json
from itertools import product
import numpy as np
import os
import logging
from ..plot.signature_plot import _plot_linear_signature
from ..model import *
from .base import ModeConfig
logger = logging.getLogger(' Mutopia-SBSModel ')


COSMIC_SORT_ORDER = [
 'A[C>A]A',
 'A[C>A]C',
 'A[C>A]G',
 'A[C>A]T',
 'C[C>A]A',
 'C[C>A]C',
 'C[C>A]G',
 'C[C>A]T',
 'G[C>A]A',
 'G[C>A]C',
 'G[C>A]G',
 'G[C>A]T',
 'T[C>A]A',
 'T[C>A]C',
 'T[C>A]G',
 'T[C>A]T',
 'A[C>G]A',
 'A[C>G]C',
 'A[C>G]G',
 'A[C>G]T',
 'C[C>G]A',
 'C[C>G]C',
 'C[C>G]G',
 'C[C>G]T',
 'G[C>G]A',
 'G[C>G]C',
 'G[C>G]G',
 'G[C>G]T',
 'T[C>G]A',
 'T[C>G]C',
 'T[C>G]G',
 'T[C>G]T',
 'A[C>T]A',
 'A[C>T]C',
 'A[C>T]G',
 'A[C>T]T',
 'C[C>T]A',
 'C[C>T]C',
 'C[C>T]G',
 'C[C>T]T',
 'G[C>T]A',
 'G[C>T]C',
 'G[C>T]G',
 'G[C>T]T',
 'T[C>T]A',
 'T[C>T]C',
 'T[C>T]G',
 'T[C>T]T',
 'A[T>A]A',
 'A[T>A]C',
 'A[T>A]G',
 'A[T>A]T',
 'C[T>A]A',
 'C[T>A]C',
 'C[T>A]G',
 'C[T>A]T',
 'G[T>A]A',
 'G[T>A]C',
 'G[T>A]G',
 'G[T>A]T',
 'T[T>A]A',
 'T[T>A]C',
 'T[T>A]G',
 'T[T>A]T',
 'A[T>C]A',
 'A[T>C]C',
 'A[T>C]G',
 'A[T>C]T',
 'C[T>C]A',
 'C[T>C]C',
 'C[T>C]G',
 'C[T>C]T',
 'G[T>C]A',
 'G[T>C]C',
 'G[T>C]G',
 'G[T>C]T',
 'T[T>C]A',
 'T[T>C]C',
 'T[T>C]G',
 'T[T>C]T',
 'A[T>G]A',
 'A[T>G]C',
 'A[T>G]G',
 'A[T>G]T',
 'C[T>G]A',
 'C[T>G]C',
 'C[T>G]G',
 'C[T>G]T',
 'G[T>G]A',
 'G[T>G]C',
 'G[T>G]G',
 'G[T>G]T',
 'T[T>G]A',
 'T[T>G]C',
 'T[T>G]G',
 'T[T>G]T']

_transition_palette = {
    ('C','A') : (0.33, 0.75, 0.98),
    ('C','G') : (0.0, 0.0, 0.0),
    ('C','T') : (0.85, 0.25, 0.22),
    ('T','A') : (0.78, 0.78, 0.78),
    ('T','C') : (0.51, 0.79, 0.24),
    ('T','G') : (0.89, 0.67, 0.72)
}

MUTATION_PALETTE = [color for color in _transition_palette.values() for i in range(16)]

CONTEXTS = sorted(
                map(lambda x : ''.join(x), product('ATCG','TC','ATCG')), 
                key = lambda x : (x[1], x[0], x[2])
                )
MUTATIONS = ['T->A/C->A','T->C/C->G','T->G/C->T']
CONFIGURATIONS = ['C/T-centered','A/G-centered']

def _reformat_mut(context, mut):
    (t_centered, c_centered) = list(map(lambda s : s[3], mut.split('/')))
    mut = t_centered if context[1] == 'T' else c_centered
    cosmic = "{}[{}>{}]{}".format(context[0], context[1], mut, context[2])
    return cosmic

MUTOPIA_ORDER = [
    _reformat_mut(context, mut)
    for context in CONTEXTS
    for mut in MUTATIONS
]

MUTOPIA_TO_COSMIC_IDX = np.array([
    MUTOPIA_ORDER.index(cosmic)
    for cosmic in COSMIC_SORT_ORDER
])


class SBSMode(ModeConfig):

    MODE_ID = 'sbs'
    CONTEXTS = CONTEXTS
    MUTATIONS = MUTATIONS
    CONFIGURATIONS = CONFIGURATIONS

    @property
    def coords(self):
        return {
            'configuration' : self.CONFIGURATIONS,
            'context' : self.CONTEXTS,
            'mutation' : self.MUTATIONS,
        }
    
    @property
    def make_model(self):
        return SBSModel
    
    @property
    def sample_params(self):
        return _sample_params

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
                np.array(
                    [database[component][context_mut] for context_mut in MUTOPIA_ORDER]
                ).reshape(
                    (cls.dim_context(), cls.dim_mutation())
                )
            )

        return np.array(comps)


    @classmethod
    def plot(cls,
        signature,
        palette = MUTATION_PALETTE,
        select = ['Baseline'],
        **kwargs,
    ):
        cls.validate_signatures(
            signature,
            required_dims=('context', 'mutation'),
        )
        signature = signature.transpose(...,'context','mutation')
        lead_dim = signature.dims[0]
        
        _plot_linear_signature(
            COSMIC_SORT_ORDER,
            palette,
            *list(map(
                lambda s : s.ravel()[MUTOPIA_TO_COSMIC_IDX],
                signature.loc[{lead_dim : list(select)}].data
            )),
            **kwargs
        )
    

    @classmethod
    def get_context_frequencies(
        cls,
        regions_file,
        fasta_file,
        n_jobs = 1,
    ):
        pass


    @classmethod
    def ingest_observations(
        cls,
        *,
        input_file,
        regions_file,
        fasta_file,
        **kwargs,
    ):
        pass



def SBSModel(
    train_corpuses,
    test_corpuses,
    n_components=15,
    init_components=None,
    seed=0,
    # context model
    context_reg=0.0001,
    conditioning_alpha=5e-5,
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
):
    
    random_state = np.random.RandomState(seed)
    
    mutation_model = MutationModel(
        train_corpuses,
        n_components=n_components,
        random_state=random_state,
        tol=5e-4,
        reg=mutation_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
    )

    context_model = StrandedContextModel(
        train_corpuses,
        n_components=n_components,
        random_state=random_state,
        tol=5e-4,
        reg=context_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
    )

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

    locals_model = LocalUpdateSparse(
        train_corpuses,
        n_components=n_components,
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
        'conditioning_alpha' : trial.suggest_float('conditioning_alpha', 1e-6, 1e-3, log=True),
        'empirical_bayes' : trial.suggest_categorical('empirical_bayes', [True, False]),
        'max_features' : trial.suggest_float('max_features', 0.1, 1.),
        'locus_subsample' : trial.suggest_categorical('locus_subsample', [None, 0.125, 0.25, 0.5]),
        'kappa' : trial.suggest_float('kappa', 0.5, 0.9),
    }