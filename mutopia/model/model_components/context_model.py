
from functools import partial, reduce
import numpy as np
from xarray import DataArray
from pandas import DataFrame
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.base import clone

#from ._poisson_elastic_net import make_optimizer, SklearnWrapper, simple_ridge
from ._glm_compiled import make_optimizer, setup_mixed_solver, get_lsqr_solver
from ._fast_eln import get_eln_solver
from ..corpus_state import CorpusState as CS
from .base import get_reg_params, get_poisson_targets_weights, _svi_update_fn, RateModel
from ._strand_transformer import StrandEncoder

import logging
logger = logging.getLogger(' LocusRegressor')


class ContextModel(RateModel):
     
    def __init__(self,
            corpuses,
            max_iter=100,
            tol=5e-4,
            reg : float = 0.0001, 
            conditioning_alpha : float = 5e-5,
            dtype = np.float32,
            init_components = None,
            *,
            n_components : int,
            random_state,
        ):
        self.n_corpuses = len(corpuses)
        self.n_components = n_components
        self.context_dim = corpuses[0].dims['context']
        self.context_names = corpuses[0].modality().CONTEXTS
        
        self.transformer = StrandEncoder(self.context_dim)\
                                    .fit(corpuses)
        

        self._coefs = self._init_params(random_state, n_components, self.context_dim, dtype)

        self._context_distribution = corpuses[0].regions.context_frequencies\
                                        .sum(dim=('locus','configuration'))\
                                        .data + 1
        self._context_distribution/=self._context_distribution.sum()

        if not init_components is None:
            self.init_from_signatures(
                corpuses[0].modality().load_components(*init_components)
            )

        is_intercept = np.array( self.transformer.intercept_mask_ + [True] )
        X = self.transformer.one_pad_right(self.transformer.get_encoding_matrix(1))
        
        eln_solver = partial(
            get_eln_solver,
            **get_reg_params(reg, conditioning_alpha),
            tol=tol,
            random_state=random_state,
        ) # f(X) -> f(z, w, beta) -> beta

        ridge_solver = partial(
            get_lsqr_solver,
            tol=tol,
            alpha=conditioning_alpha,
        ) # f(X) -> f(z, w, beta) -> beta

        mixed_solver = partial(
            setup_mixed_solver(X, ~is_intercept), # f(X) -> f( f(X) -> f(z, w, beta) -> beta, f(X) -> f(z, w, beta) -> beta ) -> f(z, w, beta) -> beta
            eln_solver, # f(X) -> f(z, w, beta) -> beta
            ridge_solver, # f(X) -> f(z, w, beta) -> beta
        ) # f() -> f(z, w, beta) -> beta
 
        self.context_models = [
            partial(
                make_optimizer(X, tol=tol, max_iter=max_iter), # f(X) -> f( f(z, w, beta) -> beta ) -> f(y, weight) -> f(beta) -> beta
                mixed_solver(), # f( solver, solver ) -> f(z, w, beta) -> beta
            ) # f(y, weight) -> f(beta) -> beta    
            for _ in range(n_components)
        ]

        '''reg_basemodel = ElasticNet(
            **get_reg_params(reg, conditioning_alpha),
            warm_start=True,
            fit_intercept=False,
            selection='random',
            tol=tol,
        )

        unreg_basemodel = partial(
            simple_ridge,
            alpha=conditioning_alpha,
            tol=tol,
            max_iter=None,
        )

        is_intercept = np.array( self.transformer.intercept_mask_ + [True] )

        self.context_models = [
            make_optimizer(
                regularized_model=SklearnWrapper(clone(reg_basemodel)),
                unregularized_model=unreg_basemodel,
                regularize_mask=~is_intercept,
                max_iter=max_iter,
                X = self.transformer.one_pad_right(
                    self.transformer.get_encoding_matrix(1)
                )
            )
            for _ in range(n_components)
        ]'''
    

    def init_from_signatures(self, signatures):
        
        k = signatures.shape[0]
        c=self.transformer.n_encoded_features_

        renormalized = (signatures.sum(axis = -1)*100 + 1)/self._context_distribution

        self._coefs[
                :k,
                0:c*self.context_dim:c
            ] = np.log( renormalized )
        
        return self
    

    def _init_params(self, random_state, n_components, n_contexts, dtype):
        return random_state.normal(0., 0.1, 
                            (
                                n_components, 
                                self.transformer.get_num_coefs() + self.n_corpuses
                            ),
                        ).astype(dtype, copy=False)


    @property
    def context_distribution_(self):
        return self._context_distribution
    

    def prepare_corpusstate(self, corpus):
        return dict(
                strand_idx = DataArray(
                    np.array([
                        self.transformer.encode(corpus),
                        self.transformer.encode(corpus, invert=True),
                    ]),
                    dims=('configuration', 'locus',),
                ),
                plus_strand_design = DataArray(
                    self.transformer.transform(corpus),
                    dims=('locus','strand_state'),
                ),
                minus_strand_design = DataArray(
                    self.transformer.transform(corpus, invert=True),
                    dims=('locus','strand_state'),
                ),
                log_context_distribution = DataArray(
                    self._get_log_context_distribution(corpus),
                    dims=('component','context','strand_state'),
                ),
            )
    

    def _get_log_context_distribution(self, corpus_state):
        return np.array([
            self._format_component(k)
            for k in range(self.n_components)
        ])
    

    def update_corpusstate(self, corpus, **kwargs):
        CS.fetch_val(corpus, 'log_context_distribution').data[:] = \
            self._get_log_context_distribution(corpus)
        
    
    def _get_sstats_dim(self):
        return (
            self.n_components, 
            self.context_dim, 
            self.transformer.n_states_
        )
    

    def spawn_sstats(self, corpus_state):
        return np.zeros(self._get_sstats_dim(), self._coefs.dtype)
            

    def predict(self, k, corpus_state):

        conditional_mutation_rates = self._format_component(k)
        # 2 x S x L
        (plus_idx, minus_idx) = CS.fetch_val(corpus_state, 'strand_idx').data
        
        # 2 x C x L => 2 x C x L
        context_effects = np.array([
                                conditional_mutation_rates[:, plus_idx], # K x C x L
                                conditional_mutation_rates[:, minus_idx]
                            ])

        return DataArray(
            context_effects,
            dims=('configuration','context','locus'),
        )
    

    @staticmethod
    def _run_regression(
        y,
        sample_weight,
        learning_rate=1.,
        *,
        update_vec,
        model,
    ):
        update_vec[:] = _svi_update_fn(
            update_vec,
            model(y, sample_weight)(update_vec),
            learning_rate=learning_rate
        )
    

    def partial_fit(self, 
                    sstats,
                    k, 
                    corpuses, 
                    log_mutation_rates,
                    learning_rate=1.,
                ):

        def aggregate_by_design(arr, design_matrix):
            #       SxL @ (LxC) => (SxC) => (CxS) => (C*S)
            return (design_matrix @ arr.T).T.ravel()

        def get_design(state, key):
            return CS.fetch_val(state, key).data.to_scipy_sparse().tocsc().T

        def get_context_exposure(state, log_mutation_rate):
            
            exp_offsets = np.exp(log_mutation_rate - self.predict(k, state))\
                            .transpose('configuration','context','locus')\
                            .data
            
            return aggregate_by_design(exp_offsets[0], get_design(state, 'plus_strand_design')) \
                + aggregate_by_design(exp_offsets[1], get_design(state, 'minus_strand_design'))
                
        eta = reduce(lambda x,y : x+y, (get_context_exposure(state, lmr) for state, lmr in zip(corpuses, log_mutation_rates)) ) # I*C*S
        target = reduce(lambda x,y : x+y, (sstats[state.attrs['name']][k].ravel() for state in corpuses) )

        yield partial(
            self._run_regression,
            *get_poisson_targets_weights(target, eta),
            model = self.context_models[k],
            update_vec = self._coefs[k],
            learning_rate = learning_rate,
        )


    '''def _make_design_matrix(self, *corpus_states):
        return sparse.hstack([
            self.transformer.get_encoding_matrix(len(corpus_states)),
            get_corpus_dummies(
                corpus_states, 
                self.corpus_dummy_encoder, 
                n_repeats = lambda _ : self.transformer.encoding_matrix_.shape[0]
            )
        ]).tocsr()'''
    
    @property
    def coefs_(self):
        return self._coefs[:,:-self.n_corpuses]
    

    def _calc_lambda(self, k, design_matrix):
        return (self.coefs_[k] @ design_matrix.T)\
                    .reshape((self.context_dim, -1))
    
    
    def _format_component(self, k):        
        return self._calc_lambda(
                    k, self.transformer.encoding_matrix_
                )
    
    def format_counterfactual(self, k):
        return self._calc_lambda(k,
                        self.transformer\
                            .independent_effects_encoding()
                    ).T
    

    def component_coef_summary(self, sig_idx):

        c = self.context_dim
        r = self.transformer.n_encoded_features_

        with_dummy = np.concatenate([
            self.coefs_[sig_idx, :c*r],
            np.ones(1),
            self.coefs_[sig_idx, c*r:],
        ])
        coef_matrix = with_dummy.reshape((c+1,-1)).T

        return DataFrame(
            coef_matrix,
            index=self.transformer.feature_names_out,
            columns=[self.context_names for i in range(c)] + ['Shared effect']
        )
