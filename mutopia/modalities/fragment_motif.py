

from .base import ModeConfig
from itertools import product
from ..model import *
import numpy as np
import logging
logger = logging.getLogger(' Mutopia-MotifModel ')

CONTEXTS = sorted(
    map(lambda x : ''.join(x), product('ATCG','ATCG','ATCG', 'ATCG')), 
    key = lambda x : (x[0], x[1], x[2], x[3])
    )

class FragmentMotif(ModeConfig):

    CONTEXTS=CONTEXTS
    MODE_ID='fragment-motif'

    @property
    def coords(self):
        return {'context' : self.CONTEXTS}
    
    @property
    def make_model(self):
        return MotifModel
    
    @property
    def sample_params(self):
        return _sample_params
    
    def get_context_frequencies(self, *args,**kw):
        pass

    def load_components(self, *args, **kw):
        pass

    def plot(self, *args, **kw):
        pass

    def ingest_observations(self, *args, **kw):
        pass


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

    context_model = UnstrandedContextModel(
        train_corpuses,
        n_components=n_components,
        random_state=random_state,
        tol=5e-4,
        reg=context_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
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
        'conditioning_alpha' : trial.suggest_float('conditioning_alpha', 1e-6, 1e-3, log=True),
        'empirical_bayes' : trial.suggest_categorical('empirical_bayes', [True, False]),
        'max_features' : trial.suggest_float('max_features', 0.1, 1.),
        'locus_subsample' : trial.suggest_categorical('locus_subsample', [None, 0.125, 0.25, 0.5]),
        'kappa' : trial.suggest_float('kappa', 0.5, 0.9),
    }