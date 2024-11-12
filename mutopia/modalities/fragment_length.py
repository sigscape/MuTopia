from .base import ModeConfig
from itertools import product
from ..model import *
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from ..plot.signature_plot import _plot_linear_signature
logger = logging.getLogger(' Mutopia-LengthModel ')

_bin_edges = [
    (70, 80),  (80, 90),  (90, 100), (100, 105), (105, 110),
    (110, 115), (115, 120), (120, 125), (125, 130), (130, 135), (135, 140), (140, 145),
    (145, 150), (150, 155), (155, 160), (160, 165), (165, 170), (170, 175), (175, 180),
    (180, 185), (185, 190), (190, 195), (195, 200), (200, 205), (205, 210), (210, 220),
    (220, 230), (230, 240), (240, 250), (250, 260), (260, 270), (270, 280), (280, 290),
    (290, 300), (300, 310), (310, 320), (320, 330), (330, 340), (340, 350), (350, 700)
]
LENGTH_BINS = [f'{l}-{r}' for l, r in _bin_edges]

# Create mutation indices

class FragmentLength(ModeConfig):

    MODE_ID='fragment-length'

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
        
        return np.expand_dims(np.array(comps), axis=-1)

    @classmethod
    def plot(
        cls,
        signature,
        palette = 'lightgrey',
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
            LENGTH_BINS,
            palette,
            *list(map(
                lambda s : s.ravel(),
                signature.loc[{lead_dim : list(select)}].data
            )),
            **kwargs
        )

    def ingest_observations(self, *args, **kw):
        pass

    def get_context_frequencies(self, *args,**kw):
        pass



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

    '''fraglength_model = UnconditionalConsequenceModel(
        'fragment-length',
        train_corpuses,
        n_components=n_components,
        random_state=random_state,
        tol=5e-4,
        reg=reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
    )'''

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