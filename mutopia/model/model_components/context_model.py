
from functools import partial, reduce
import numpy as np
from xarray import DataArray
from pandas import DataFrame
from ._glm_compiled import make_optimizer, setup_mixed_solver, get_lsqr_solver
from ._fast_eln import get_eln_solver
from ..corpus_state import CorpusState as CS
from .base import get_reg_params, get_poisson_targets_weights, _svi_update_fn, \
    RateModel, SparseDataBase
from ._strand_transformer import StrandEncoder
from ..utils import dims_except_for

import logging
logger = logging.getLogger(' LocusRegressor')


class StrandedContextModel(RateModel, SparseDataBase):
     
    def __init__(self,
        corpuses,
        max_iter=100,
        tol=5e-4,
        reg : float = 0.0001, 
        conditioning_alpha : float = 5e-5,
        dtype = float,
        init_components = None,
        *,
        n_components : int,
        random_state,
    ):
        super().__init__(*corpuses)
        
        #self.n_corpuses = len(corpuses)
        self.n_components = n_components
        self.context_dim = corpuses[0].dims['context']
        self.context_names = corpuses[0].modality().coords['context']
        
        self.transformer = StrandEncoder(self.context_dim)\
                                    .fit(corpuses)
        

        self._coefs = self._init_params(random_state, n_components, self.context_dim, dtype)

        corpus=corpuses[0] # just grab the first one to use for initialization
        self._context_distribution = \
            corpus.regions.context_frequencies\
            .sum(dim=dims_except_for(corpus.regions.context_frequencies.dims, 'context'))\
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
            setup_mixed_solver,
            is_regularized = ~is_intercept,
            reg_solver = eln_solver, # f(X) -> f(z, w, beta) -> beta
            unreg_solver = ridge_solver, # f(X) -> f(z, w, beta) -> beta
        )
        
        self.model = make_optimizer(
            X,
            mixed_solver,
            tol=tol,
            max_iter=max_iter,
        )


    @property
    def requires_normalization(self):
        return True
    

    @property
    def requires_dims(self):
        return ('configuration','context','locus')
    

    def prepare_to_save(self):
        del self.model 
        # the model is not serializable because it has nested functions

    ##
    # Satisfaction of SparseDataBase interface
    ##
    @staticmethod
    def predict_sparse(corpus,*, configuration, context, locus, **idx_dict):

        strand_state = CS.fetch_val(corpus, 'strand_idx').data\
                            [configuration, locus]

        return CS.fetch_val(corpus, 'log_context_distribution').data\
                            [:, context, strand_state]
    

    @staticmethod
    def reduce_sparse_sstats(
        sstats, 
        corpus,*,
        weighted_posterior,
        locus,
        context,
        configuration,
        **kw,
    ):
        strand_states = CS.fetch_val(corpus, 'strand_idx').data\
                            [configuration, locus]

        # Use numpy advanced indexing to update context_sstats
        np.add.at(sstats, (slice(None), context, strand_states), weighted_posterior)

        return sstats
    ##
    # End of SparseDataBase interface
    ##


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
        return random_state.normal(
                    0., 0.1, 
                    (
                        n_components, 
                        self.transformer.get_num_coefs() + 1 #self.n_corpuses
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


    @staticmethod
    def get_exp_offset(offsets, corpus):
        
        def aggregate_by_design(arr, design_matrix):
            #       SxL @ (LxC) => (SxC) => (CxS) => (C*S)
            return (design_matrix @ arr.T).T.ravel()

        def get_design(state, key):
            return CS.fetch_val(state, key).data.to_scipy_sparse().tocsc().T

        exp_offsets = np.exp(offsets)\
                        .transpose('configuration','context','locus')\
                        .data
        
        return aggregate_by_design(exp_offsets[0], get_design(corpus, 'plus_strand_design')) \
            + aggregate_by_design(exp_offsets[1], get_design(corpus, 'minus_strand_design'))


    def partial_fit(self, 
                    k,
                    sstats,
                    exp_offsets, 
                    corpuses, 
                    learning_rate=1.,
                ):

        corpus_names = [state.attrs['name'] for state in corpuses]

        eta = reduce(sum, (exp_offsets[n][k].ravel() for n in corpus_names))
        target = reduce(sum, (sstats[n][k].ravel() for n in corpus_names))

        yield partial(
            self._run_regression,
            *get_poisson_targets_weights(target, eta),
            model = self.model,
            update_vec = self._coefs[k],
            learning_rate = learning_rate,
        )
    
    @property
    def coefs_(self):
        return self._coefs[:,:-1] #self.n_corpuses]
    

    def _calc_lambda(self, k, design_matrix):
        return (self.coefs_[k] @ design_matrix.T)\
                    .reshape((self.context_dim, -1))
    
    
    def _format_component(self, k):        
        return self._calc_lambda(
                    k, self.transformer.encoding_matrix_
                )
    

    def format_signature(self, k):
        # C x S
        signature = self._calc_lambda(k,
                        self.transformer\
                            .independent_effects_encoding()
                    ) + np.log(self._context_distribution)[:, None]
        
        return DataArray(
            signature,
            dims=('context','strand_state'),
            coords={
                'context' : self.context_names,
                'strand_state' : self.transformer.feature_names_out
            }
        )
    
    def get_save_data(self):
        return dict(
            coefs = self.coefs_,
            context_distribution = self.context_distribution_,
        )
    

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



class UnstrandedContextModel(StrandedContextModel, SparseDataBase):

    @property
    def requires_dims(self):
        return ('context','locus')
    

    @staticmethod
    def predict_sparse(corpus,*, context, locus, **idx_dict):

        strand_state = CS.fetch_val(corpus, 'strand_idx').data[locus]

        return CS.fetch_val(corpus, 'log_context_distribution').data\
                            [:, context, strand_state]
    

    @staticmethod
    def reduce_sparse_sstats(
        sstats, 
        corpus,
        *,
        weighted_posterior,
        locus,
        context,
        **kw,
    ):
        strand_states = CS.fetch_val(corpus, 'strand_idx').data[locus]

        # Use numpy advanced indexing to update context_sstats
        np.add.at(sstats, (slice(None), context, strand_states), weighted_posterior)

        return sstats
    

    def prepare_corpusstate(self, corpus):
        return dict(
                strand_idx = DataArray(
                    self.transformer.encode(corpus),
                    dims=('locus',),
                ),
                strand_design = DataArray(
                    self.transformer.transform(corpus),
                    dims=('locus','strand_state'),
                ),
                log_context_distribution = DataArray(
                    self._get_log_context_distribution(corpus),
                    dims=('component','context','strand_state'),
                ),
            )
    

    def predict(self, k, corpus_state):

        conditional_mutation_rates = self._format_component(k)
        # S x L
        idx = CS.fetch_val(corpus_state, 'strand_idx').data

        return DataArray(
            conditional_mutation_rates[:, idx],
            dims=('context','locus'),
        )
    

    @staticmethod
    def get_exp_offset(offsets, corpus):
        
        def aggregate_by_design(arr, design_matrix):
            #       SxL @ (LxC) => (SxC) => (CxS) => (C*S)
            return (design_matrix @ arr.T).T.ravel()

        def get_design(state, key):
            return CS.fetch_val(state, key).data.to_scipy_sparse().tocsc().T

        exp_offsets = np.exp(offsets)\
                        .transpose('context','locus')\
                        .data
        
        return aggregate_by_design(exp_offsets, get_design(corpus, 'strand_design'))