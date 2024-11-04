
from .model_state import ModelState
from .corpus_state import CorpusState as CS
from sklearn.base import BaseEstimator
from optim import *
from .model_components import *
from .latent_var_models import *
from .eval import *

class Model(BaseEstimator):

    def __init__(self,
        *,
        # model structure
        locus_model_type='gbt',
        n_components = 10,
        init_components = None,
        # regularization
        context_reg=0.0001,
        mutation_reg=0.0005,
        l2_regularization = 1.,
        smoothing_size = 1000,
        conditioning_alpha = 1e-5,
        # prior settings
        pi_prior = 1.,
        empirical_bayes = True,
        begin_prior_updates = 10,
        # optimization settings
        num_epochs = 5000,
        locus_subsample = None,
        threads = 1,
        kappa = 0.5,
        tau = 1.,
        difference_tol = 1e-4,
        estep_iterations = 300,
        stop_condition = 50,
        # tree settings
        tree_learning_rate=0.1, 
        max_depth = 5,
        max_trees_per_iter = 25,
        max_leaf_nodes = 31,
        min_samples_leaf = 30,
        max_features = 0.5,
        n_iter_no_change= 2,
        use_groups=True,
        dtype = float,
        add_corpus_intercepts = False,
        random_state = 0,
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
        self.locus_model_type = locus_model_type
        self.smoothing_size = smoothing_size
        self.stop_condition = stop_condition

    
    def _init_model(self, *corpuses):

        theta_model = \
            (GBTThetaModel if self.locus_model_type == 'gbt' \
            else LinearThetaModel)\
            (
                corpuses,
                tree_learning_rate=self.tree_learning_rate,
                max_depth=self.max_depth,
                max_trees_per_iter=self.max_trees_per_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_iter_no_change=self.n_iter_no_change,
                use_groups=self.use_groups,
                random_state=self.random_state,
                smoothing_size=self.smoothing_size,
                add_intercepts=self.add_corpus_intercepts,
                l2_regularization=self.l2_regularization,
            )

        context_model = StrandedContextModel(
            corpuses,
            n_components=self.n_components,
            random_state=self.random_state,
            tol=5e-4,
            reg=self.context_reg,
            conditioning_alpha=self.conditioning_alpha,
            init_components=self.init_components,
        )

        mutation_model = MutationModel(
            corpuses,
            n_components=self.n_components,
            random_state=self.random_state,
            tol=5e-4,
            reg=self.mutation_reg,
            init_components=self.init_components,
        )

        locals_model = LocalUpdateSparse(
            corpuses,
            random_state=self.random_state,
            prior_alpha=self.pi_prior,
            estep_iterations=self.estep_iterations,
        )

        model_state = ModelState(
            corpuses,
            context_model=context_model,
            mutation_model=mutation_model,
            theta_model=theta_model,
            locals_model=locals_model,
        )

        return model_state


    def _setup_corpus(self, corpus):
        corpus = CS.init_corpusstate(
            corpus,
            self.model_state_,
        )

        corpus = CS.update_corpusstate(
            corpus,
            self.model_state_,
            from_scratch=True,
        )

        return corpus
    

    def _should_stop(self, test_scores):
        return len(test_scores) > self.stop_condition \
            and test_scores.index(max(test_scores)) < (len(test_scores) - self.stop_condition)
            

    def _fit(self, 
        *corpuses, 
        train_test_splitter=lambda *x : (x, []),
    ):
        
        train_corpuses, test_corpuses = train_test_splitter(*corpuses)
        self.model_state_ = self._init_model(*train_corpuses)

        for corpus in train_corpuses:   
            CS.init_corpusstate(
                corpus,
                self.model_state_,
            )

        for test_corpus in test_corpuses:
            CS.init_corpusstate(
                test_corpus,
                self.model_state_,
            )

        train_corpus_dict = {c.attrs['name'] : c for c in train_corpuses}
        test_score_fn = partial(
            deviance,
            corpuses=test_corpuses,
            exposures_fn = lambda test_corpus, sample_name : \
                CS.fetch_topic_compositions(
                    train_corpus_dict[test_corpus.attrs['name']], 
                    sample_name
                ),
        )

        show_score = partial(
            perplexity,
            get_n_mutations(train_corpuses)
        )

        step_kw = dict(
            model_state=self.model_state_,
            corpuses=train_corpuses,
            test_score_fn=test_score_fn,
        )

        if self.locus_subsample is None:
            step_fn = partial(
                VI_step, 
                **step_kw
            )
        else:
            step_fn = partial(
                SVI_step, 
                **step_kw,
                batch_generator=partial(
                    locus_slice_generator, 
                    self.random_state, 
                    subsample_rate=self.locus_subsample
                ),
                subsample_rate=self.locus_subsample,
            )

        self.train_scores_ = []
        self.test_scores_ = []

        with ParContext(self.threads) as par:
            
            for epoch in range(self.num_epochs):
                
                train_score, test_score = step_fn(
                    parallel_context=par,
                    update_prior = epoch >= self.begin_prior_updates \
                        and self.empirical_bayes,
                    learning_rate = learning_rate_schedule(epoch, self.tau, self.kappa),
                )

                for test_corpus in test_corpuses:
                    CS.update_corpusstate(
                        test_corpus,
                        self.model_state_,
                        from_scratch=False,
                    )

                self.train_scores_.append(train_score)
                self.test_scores_.append(test_score)

                yield (
                    epoch, 
                    show_score(train_score), 
                    show_score(test_score),
                )

                if self._should_stop(self.test_scores_):
                    break


    def fit(self, *corpuses, train_test_splitter=lambda x : (x, [])):
        
        for epoch, train_score, test_score in self._fit(
            *corpuses, 
            train_test_splitter=train_test_splitter
        ):
            pass

        return self 
    

    def score(self, *corpuses):
        
        ##
        # Need to make sure the corpus has been initialized
        # before scoring.
        ##

        return perplexity(
            get_n_mutations(corpuses),
            score(
                corpuses=corpuses,
                model_state=self.model_state_,
                locals_weight=0.0,
            )
        )
    

    def predict_exposures(self, corpus):
        pass