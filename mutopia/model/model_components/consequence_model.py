from .base import get_reg_params, _svi_update_fn, RateModel, SparseDataBase, DenseDataBase
from ._strand_transformer import NormalizedMesoscaleEncoder
from ._glm_compiled import make_optimizer, setup_mixed_solver, \
    get_lsqr_solver, ls_partial_solver
from ._fast_eln import get_eln_solver
from functools import partial
from sklearn.base import clone
import numpy as np
from ..corpus_state import CorpusState as CS
from ..utils import dims_except_for
from xarray import DataArray
from functools import reduce

class StrandedConditionalConsequenceModel(RateModel, SparseDataBase, DenseDataBase):
    '''
    Stranded -> depends on the presence of the "configuration" dimension
    Conditional -> depends on the context of the event
    Consequence -> this model predicts the consequences of some event, not where it occurs.
    '''

    def __init__(
        self,
        dim_name,
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
        self.dim_name = dim_name
        super().__init__(*corpuses)

        self.n_components = n_components
        self.consequence_dim = corpuses[0].dims[dim_name]
        self.consequence_names = corpuses[0].coords[dim_name].values
        self.context_dim = corpuses[0].dims['context']
        self.context_names = corpuses[0].coords['context'].values

        self.transformer = NormalizedMesoscaleEncoder(self.consequence_dim)\
                                .fit(corpuses)
        
        _cons_encoding_matrix = self.transformer.encoding_matrix_
        self._is_regularized = ~np.array(self.transformer.intercept_mask_)
        X = _cons_encoding_matrix.copy()

        eln_solver = partial(
            get_eln_solver,
            **get_reg_params(reg, conditioning_alpha),
            tol=tol,
            random_state=random_state,
        ) # f(X) -> f(z, w, beta) -> beta

        ridge_solver = partial(
            ls_partial_solver,
            group_mask = np.array(
                [True]*self.consequence_dim \
                + [False]*( (~self._is_regularized).sum() - self.consequence_dim )
            ),
            tol=tol,
            max_iter=10000,
        )

        # f(X) -> f( f(X) -> f(z, w, beta) -> beta, f(X) -> f(z, w, beta) -> beta ) -> f(z, w, beta) -> beta
        mixed_solver = partial(
            setup_mixed_solver,
            is_regularized = self._is_regularized,
            reg_solver = eln_solver,
            unreg_solver = ridge_solver,
        )
 
        self.model = make_optimizer(
            X,
            mixed_solver,
            tol=tol,
            max_iter=max_iter,
        )

        # convert to dense for other computations, but keep sparse for regression updates
        self._cons_encoding_matrix=_cons_encoding_matrix\
                                    .toarray()[:,:-self.transformer.n_states_]

        self._coefs = self._init_params(
            random_state, 
            n_components, 
            self.context_dim, 
            dtype
        )

        if not init_components is None:
            self.init_from_signatures(
                corpuses[0].modality().load_components(*init_components)
            )

    @property
    def requires_normalization(self):
        return False
    

    @property
    def requires_dims(self):
        return ('configuration','context', self.dim_name, 'locus')
    
    def prepare_to_save(self):
        del self.model
        
    
    def predict_sparse(self, corpus,*,context, configuration, locus, **idx_dict):

        consequence = idx_dict[self.dim_name]

        mesoscale_state = CS.fetch_val(corpus, 'mesoscale_idx').data\
                            [configuration, locus]
        
        return CS.fetch_val(corpus, f'log_{self.dim_name}_distribution').data\
                [:, context, consequence, mesoscale_state]
    

    def reduce_sparse_sstats(
        self,
        sstats, 
        corpus,
        *,
        weighted_posterior,
        configuration,
        locus,
        context,
        **idx_dict,
    ):
        consequence = idx_dict[self.dim_name]
        mesoscale_states = CS.fetch_val(corpus, 'mesoscale_idx').data\
                            [configuration, locus]

        # Use numpy advanced indexing to update mutation_sstats
        np.add.at(sstats, (slice(None), context, consequence, mesoscale_states), weighted_posterior)

        return sstats
    

    def reduce_dense_sstats(self, sstats, corpus, *, weighted_posterior, **kw):

        to_dim = ('component', *self.requires_dims)

        if not to_dim[1] == 'configuration':
            raise ValueError('First dimension must be configuration!')
        
        # K x D x C x M x L
        weights = weighted_posterior\
            .sum(dim=dims_except_for(weighted_posterior.dims, *to_dim))\
            .transpose(*to_dim)\
            .data

        # 2 x L
        (plus_idx, minus_idx) = CS.fetch_val(corpus, 'mesoscale_idx').data
        
        # sstats : K x C x S x M
        np.add.at(sstats, (slice(None), slice(None), slice(None), plus_idx), weights[:,0])
        np.add.at(sstats, (slice(None), slice(None), slice(None), minus_idx), weights[:,1])

        return sstats


    def init_from_signatures(self, signatures):

        k = signatures.shape[0]
        c=self.transformer.n_encoded_features_

        renormalized = signatures + 1e-5
        renormalized = renormalized/renormalized.sum(axis = -1, keepdims = True)

        self._coefs[
                :k,
                :,
                0:c*self.consequence_dim:c
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
    

    def _get_log_cons_distribution(self, corpus_state):
        return np.array([
            self._format_component(k)
            for k in range(self.n_components)
        ])


    def prepare_corpusstate(self, corpus):
        return dict(
                    {f'log_{self.dim_name}_distribution' : DataArray(
                        self._get_log_cons_distribution(corpus),
                        dims=('component','context', self.dim_name, 'mesoscale_state')
                    ),
                    }
                )
        
    def update_corpusstate(self, corpus, **kwargs):
        CS.fetch_val(corpus, f'log_{self.dim_name}_distribution').data[:] = \
            self._get_log_cons_distribution(corpus)


    def _get_sstats_dim(self):
        return (
            self.n_components, 
            self.context_dim, 
            self.consequence_dim,
            self.transformer.n_states_, 
        )

    def spawn_sstats(self, corpus):
        return np.zeros(self._get_sstats_dim(), self._coefs.dtype)
    

    def predict(self,k, corpus_state):

        #CxMxS
        rho = self._format_component(k)

        (plus_idx, minus_idx) = CS.fetch_val(corpus_state, 'mesoscale_idx').data
        
         # 2 x C x M x L
        mutation_effects = np.array([
                                rho[:,:,plus_idx], # C x M x L
                                rho[:,:,minus_idx]
                            ])

        return DataArray(
            mutation_effects,
            dims=('configuration', 'context', self.dim_name, 'locus'),
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

        for c in range(self.context_dim):
            
            # CxMxS => MxS => M*S
            target=stats_reduced[c].ravel()
            target = target/target.mean()

            yield partial(
                self._run_regression,
                target,
                np.ones_like(target),
                update_vec=self._coefs[k,c],
                learning_rate=learning_rate,
                model=self.model
            )
    

    @property
    def coefs_(self):
        return self._coefs[:,:,:-self.transformer.n_states_]
    

    def _calc_rho(self,k, design_matrix):

        rho_tilde = np.exp( self.coefs_[k] @ design_matrix.T)\
            .reshape((self.context_dim, self.consequence_dim, -1))
        # CxMxS
        rho = rho_tilde/rho_tilde.sum(axis = -2, keepdims = True)

        return np.log(rho)


    def format_signature(self, k):
        # CxMxS
        rho = self._calc_rho(k, 
                    self.transformer\
                        .independent_effects_encoding()
                )
        
        return DataArray(
            rho,
            dims=('context', self.dim_name, 'mesoscale_state'),
            coords={
                'mesoscale_state' : self.transformer.feature_names_out,
                'context' : self.context_names,
                self.dim_name : self.consequence_names,
            }
        )
    

    def _format_component(self, k):
        return self._calc_rho(k, self._cons_encoding_matrix)

