
from .model_state import ModelState
from sklearn.base import BaseEstimator
from optim import VI_step, SVI_step,\
     locus_slice_generator, learning_rate_schedule
from .model_components import *
from .latent_var_models import *

class Model(BaseEstimator):

    def __init__(self,
        *,
        model_type='gbt',
        n_components = 10,
        random_state = 0,
        context_reg=0.0001,
        mutation_reg=0.0005,
        l2_regularization = 1.,
        dtype = float,
        pi_prior = 1.,
        num_epochs = 300, 
        difference_tol = 1e-4,
        estep_iterations = 1000,
        threads = 1,
        locus_subsample = 1,
        empirical_bayes = True,
        kappa = 0.5,
        tau = 1.,
        begin_prior_updates = 10,
        init_components = None,
        add_corpus_intercepts = False,
        conditioning_alpha = 1e-5,
        tree_learning_rate=0.1, 
        max_depth = 5,
        max_trees_per_iter = 25,
        max_leaf_nodes = 31,
        min_samples_leaf = 30,
        max_features = 0.75,
        n_iter_no_change=2,
        use_groups=True,
        ):

        self.n_components = n_components
        self.random_state = random_state
        self.context_reg = context_reg
        self.mutation_reg = mutation_reg
        self.l2_regularization = l2_regularization
        self.dtype = dtype
        self.pi_prior = pi_prior
        self.num_epochs = num_epochs
        self.difference_tol = difference_tol
        self.estep_iterations = estep_iterations
        self.threads = threads
        self.locus_subsample = locus_subsample
        self.empirical_bayes = empirical_bayes
        self.kappa = kappa
        self.tau = tau
        self.begin_prior_updates = begin_prior_updates
        self.init_components = init_components
        self.add_corpus_intercepts = add_corpus_intercepts
        self.conditioning_alpha = conditioning_alpha
        self.tree_learning_rate = tree_learning_rate
        self.max_depth = max_depth
        self.max_trees_per_iter = max_trees_per_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_iter_no_change = n_iter_no_change
        self.use_groups = use_groups
        self.model_type = model_type

    
    def init_model(self, *corpuses):

        theta_model = (GBTThetaModel if self.model_type == 'gbt' else LinearThetaModel)(
            corpuses,
            tree_learning_rate=self.tree_learning_rate,
            max_depth=self.max_depth,
        )


    def fit(self, *corpuses):
        pass
