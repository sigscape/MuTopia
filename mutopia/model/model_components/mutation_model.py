from .base import get_reg_params, _svi_update_fn, RateModel
from ._strand_transformer import MutationStrandEncoder
#from ._poisson_elastic_net import make_optimizer, SklearnWrapper, simple_ridge
from ._glm_compiled import make_optimizer, setup_mixed_solver, \
    get_lsqr_solver, ls_partial_solver
from ._fast_eln import get_eln_solver
from functools import partial
from sklearn.base import clone
import numpy as np
from ..corpus_state import CorpusState as CS
from xarray import DataArray
from functools import reduce

class MutationModel(RateModel):

    def __init__(self,
            corpuses,
            reg : float = 0.0005,
            conditioning_alpha = 5e-5,
            tol = 5e-4,
            max_iter=100,
            dtype = float,
            init_components = None,
            *,
            n_components,
            random_state,
            ):
        self.n_components = n_components
        self.mutation_dim = corpuses[0].dims['mutation']
        self.context_dim = corpuses[0].dims['context']
        
        self.transformer = MutationStrandEncoder(self.mutation_dim)\
                                        .fit(corpuses)
        
        _mut_encoding_matrix = self.transformer.encoding_matrix_
        is_regularized = ~np.array(self.transformer.intercept_mask_)
        X = _mut_encoding_matrix.copy()

        eln_solver = partial(
            get_eln_solver,
            **get_reg_params(reg, conditioning_alpha),
            tol=tol,
            random_state=random_state,
        ) # f(X) -> f(z, w, beta) -> beta

        '''ridge_solver = partial(
            get_lsqr_solver,
            tol=tol,
            alpha=conditioning_alpha,
        ) # f(X) -> f(z, w, beta) -> beta'''

        ridge_solver = partial(
            ls_partial_solver,
            group_mask = np.array([True]*3 + [False]*(is_regularized.sum()-3)),
            tol=tol,
            max_iter=10000,
        )

        # f(X) -> f( f(X) -> f(z, w, beta) -> beta, f(X) -> f(z, w, beta) -> beta ) -> f(z, w, beta) -> beta
        mixed_solver = setup_mixed_solver(X, is_regularized)(
                            eln_solver, # f(X) -> f(z, w, beta) -> beta
                            ridge_solver, # f(X) -> f(z, w, beta) -> beta
                        ) # f(z, w, beta) -> beta
 
        self.mutation_models = [
            [
                partial(
                    make_optimizer(X, tol=tol, max_iter=max_iter), # f(X) -> f( f(z, w, beta) -> beta ) -> f(y, weight) -> f(beta) -> beta
                    mixed_solver, # f(z, w, beta) -> beta
                ) # f(y, weight) -> f(beta) -> beta
                for _ in range(self.context_dim)
            ]
            for _ in range(n_components)
        ]

        # convert to dense for other computations, but keep sparse for regression updates
        self._mut_encoding_matrix=_mut_encoding_matrix\
                                    .toarray()[:,:-self.transformer.n_states_]

        self._coefs = self._init_params(random_state, n_components, self.context_dim, dtype)

        if not init_components is None:
            self.init_from_signatures(
                corpuses[0].modality().load_components(*init_components)
            )

    @property
    def requires_normalization(self):
        return False

    def init_from_signatures(self, signatures):

        k = signatures.shape[0]
        c=self.transformer.n_encoded_features_

        renormalized = signatures*1000 + 1
        renormalized = renormalized/renormalized.sum(axis = -1, keepdims = True)

        self._coefs[
                :k,
                :,
                0:c*self.mutation_dim:c
            ] = np.log( renormalized )
    

    def _init_params(self, random_state, n_components, context_dim, dtype):
        return random_state.normal(
                    0, 0.1,
                    (
                        n_components, 
                        context_dim,
                        self.transformer.get_num_coefs()
                    )
                ).astype(dtype, copy = False)
    

    def _get_log_mutation_distribution(self, corpus_state):
        return np.array([
            self._format_component(k)
            for k in range(self.n_components)
        ])


    def prepare_corpusstate(self, corpus):
        return dict(
                    log_mutation_distribution = DataArray(
                        self._get_log_mutation_distribution(corpus),
                        dims=('component','context','mutation','strand_state')
                    )
                )
        
    def update_corpusstate(self, corpus, **kwargs):
        CS.fetch_val(corpus, 'log_mutation_distribution').data[:] = \
            self._get_log_mutation_distribution(corpus)


    def _get_sstats_dim(self):
        return (
            self.n_components, 
            self.context_dim, 
            self.transformer.n_states_, 
            self.mutation_dim
        )
    

    def spawn_sstats(self, corpus):
        return np.zeros(self._get_sstats_dim(), self._coefs.dtype)
    

    def predict(self,k, corpus_state):

        #CxMxS
        rho = self._format_component(k)

        (plus_idx, minus_idx) = CS.fetch_val(corpus_state, 'strand_idx').data
        
         # 2 x C x M x L
        mutation_effects = np.array([
                                rho[:,:,plus_idx], # C x M x L
                                rho[:,:,minus_idx]
                            ])

        return DataArray(
            mutation_effects,
            dims=('configuration','context','mutation','locus'),
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


    @staticmethod
    def get_exp_offset(offsets, corpus):
        return None
        
    
    def partial_fit(self, 
                    k,
                    sstats,
                    exp_offsets, 
                    corpuses, 
                    learning_rate=1.,
                ):

        corpus_names = [state.attrs['name'] for state in corpuses]
        # CxSxM
        stats_reduced = reduce(sum, [sstats[n][k] for n in corpus_names])

        for c, model in enumerate(self.mutation_models[k]):
            
            # CxSxM => MxS => M*S
            target=stats_reduced[c].T.ravel()
            target = target/target.mean()

            yield partial(
                self._run_regression,
                target,
                np.ones_like(target),
                update_vec=self._coefs[k,c],
                learning_rate=learning_rate,
                model=model
            )
    

    @property
    def coefs_(self):
        return self._coefs[:,:,:-self.transformer.n_states_]
    

    def _calc_rho(self,k, design_matrix):

        rho_tilde = np.exp( self.coefs_[k] @ design_matrix.T)\
            .reshape((self.context_dim, self.mutation_dim, -1))
        # CxMxS
        rho = rho_tilde/rho_tilde.sum(axis = -2, keepdims = True)

        return np.log(rho)


    def format_counterfactual(self, k):
        return self._calc_rho(k, 
                    self.transformer\
                        .independent_effects_encoding()
                ).transpose((2,0,1))


    def _format_component(self, k):
        return self._calc_rho(k, self._mut_encoding_matrix)
