from xarray import DataArray
import numpy as np
from functools import partial, reduce
from ..model_components.base import _svi_update_fn, PrimitiveModel
from ..corpus_state import CorpusState as CS
from ._dirichlet_update import update_alpha
from .base import *
from numba import njit
import warnings

# enforcing contiguous memory layout for  the following functions
@njit('double[::1](double[::1], double[::1], double[:,::1], double[::1])', nogil=True)
def _update_step(
        gamma,
        alpha, 
        conditional_likelihood, 
        weights,
    ):

    exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma))
    
    # NxK @ K => N
    X_tild = exp_Elog_gamma @ conditional_likelihood
    #                                    KxN @ N => K
    gamma_sstats = exp_Elog_gamma*(conditional_likelihood @ (weights/X_tild))

    return alpha + gamma_sstats


@njit('double[::1](double[::1], double[:,::1], double[::1], int64, double, double[::1])', nogil=True)
def _iterative_update(
    alpha, 
    conditional_likelihood, 
    weights,
    iters,
    tol,
    gamma, # move gamma to the end so we can curry the function
):
    for _ in range(iters): # inner e-step loop
        
        old_gamma = gamma.copy()
        gamma = _update_step(
            gamma,
            alpha, 
            conditional_likelihood, 
            weights,
        )

        if (np.abs(gamma - old_gamma)/np.sum(old_gamma)).sum() < tol:
            break

    return gamma


@njit('double[:,::1](double[::1], double[:,::1], double[::1])', nogil=True)
def _calc_local_variables(
        gamma,
        conditional_likelihood,
        weights,
    ):
    
    exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma))
    # NxK @ Kx1 => Nx1
    X_tild = exp_Elog_gamma @ conditional_likelihood

    phi_matrix = np.outer(exp_Elog_gamma, weights/X_tild)*conditional_likelihood

    return phi_matrix


@njit('double(double[:,::1], double[::1], double[::1], double[:,::1], double[::1], double)', nogil=True)
def _bound(
    weighted_posterior,
    gamma,
    alpha,
    conditional_likelihood,
    weights,
    locals_weight,
    ):

    phi = weighted_posterior/weights[None,:]
    entropy_sstats = -np.sum(weighted_posterior * np.where(phi > 0, np.log(phi), 0.))
    entropy_sstats += dirichlet_bound(alpha, gamma)
    
    flattened_logweight = log_dirichlet_expectation(gamma)[:,None] + np.log(conditional_likelihood)
    
    return np.sum(
                np.where(
                    np.isfinite(flattened_logweight), 
                    weighted_posterior*flattened_logweight, 
                    0.
                )
            ) \
            + entropy_sstats*locals_weight



class LocalUpdateSparse(PrimitiveModel, LocalUpdate):

    def __init__(self,
            corpuses,
            n_components,
            prior_alpha=1.0,
            estep_iterations=300,
            difference_tol=1e-4,
            dtype=float,
            *,
            random_state,
        ):
        self.estep_iterations = estep_iterations
        self.difference_tol = difference_tol
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype

        self.alpha = {
            corpus.attrs['name'] : np.ones(n_components, dtype=dtype)*prior_alpha
            for corpus in corpuses
        }
    
    ##
    # E-step functionality to satisfy the LocalUpdate interface
    ##
    def convert_sample(self, sample):

        sample = sample.sparse_to_coo()
        weights = np.ascontiguousarray(sample.data.data.astype(self.dtype, copy=True))

        idx_dict = dict(zip(
            tuple(sample.coords['obs_indices'].data),
            sample.indices.data,
        ))

        return dict(
            **idx_dict,
            weights=weights,
        )


    def conditional_observation_likelihood(
        self,
        corpus,
        model_state,
        *,
        weights,
        locus,
        **idx_dict,
    ):

        ##
        # What's going on here: we have the normalized log mutation rate for each signature, configuration, context, locus.
        # For the mutations in this sample, we select over these axes.
        ##
        logp_normalizer = model_state.get_normalizers(corpus)[:,None]

        # np.log(corpus.regions.context_frequencies.data[context, locus]) \
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logp_X = reduce(
                        lambda x,y: x+y,
                        (
                            model.predict_sparse(corpus, locus=locus, **idx_dict)
                            for model in model_state.nonlocals.values()
                        ),
                        np.log(corpus.regions.exposures.data[locus]) \
                        + logp_normalizer
                    )

        return np.ascontiguousarray(
                    np.nan_to_num(np.exp(logp_X), nan=0)\
                        .astype(self.dtype, copy=False)
                )
    
    
    @staticmethod
    def _apply_update(
        gamma, 
        *,
        suffstat_fn,
        update_fn, 
        learning_rate
    ):
        return suffstat_fn(
            _svi_update_fn(
                gamma, 
                update_fn(gamma), 
                learning_rate
            )
        )
    

    def _get_update_fn(
        self,
        learning_rate=1.,
        subsample_rate=1.,
        *,
        corpus,
        sample,
        model_state,
    ):
        '''
        Why am I doing it this way? - one could pass 
        the update function pointfree to some multiprocessing
        generator.

        I took pains to prevent the large corpus and model_state 
        objects from being pulled into the scope of the update function.
        '''
        
        # 1. get the information we need from the sample
        sample_dict = self.convert_sample(sample)
        weights = sample_dict['weights']/subsample_rate
        alpha = np.ascontiguousarray(self.alpha[corpus.attrs['name']])

        conditional_likelihood = \
            self.conditional_observation_likelihood(
                corpus, 
                model_state,
                **sample_dict
            )
        
        map_update = partial(
            _iterative_update,
            alpha, 
            conditional_likelihood, 
            weights,
            self.estep_iterations,
            self.difference_tol,
        )
        
        suffstat_fn = partial(
            self._calc_sstats,
            sample_dict=sample_dict,
            conditional_likelihood=conditional_likelihood,
            weights=weights,
            alpha=alpha,
        )

        svi_update = partial(
            self._apply_update,
            update_fn=map_update,
            learning_rate=learning_rate,
            suffstat_fn=suffstat_fn
        )

        return svi_update
        

    @staticmethod
    def _calc_sstats(
        gamma,
        *,
        alpha,
        sample_dict,
        conditional_likelihood,
        weights,
        ):

        weighted_posterior = _calc_local_variables(
            gamma,
            conditional_likelihood,
            weights,
        )

        bound = _bound(
            weighted_posterior,
            gamma,
            alpha,
            conditional_likelihood,
            weights,
            1.
        )        

        suffstats = {
                **sample_dict,
                'weighted_posterior' : weighted_posterior, 
                'gamma' : gamma, 
            }

        return (suffstats, gamma, bound)
    

    def update_locals(self,
            gamma0,
            learning_rate=1.,
            subsample_rate=1.,
            *,
            corpus,
            sample,
            model_state,
        ):

        return self._get_update_fn(
                    learning_rate=learning_rate,
                    subsample_rate=subsample_rate,
                    corpus=corpus,
                    sample=sample,
                    model_state=model_state
                )(gamma0)
        
    
    def bound(self,
        gamma,
        subsample_rate=1.,
        locals_weight=1.,
        *,
        corpus,
        sample,
        model_state,
    ):
        
        sample_dict = self.convert_sample(sample)
        alpha = np.ascontiguousarray(self.alpha[corpus.attrs['name']])
        gamma = np.ascontiguousarray(gamma)
        weights = sample_dict['weights']/subsample_rate
        
        conditional_likelihood = \
            self.conditional_observation_likelihood(
                corpus, 
                model_state,
                **sample_dict
            )

        weighted_posterior = _calc_local_variables(
            gamma,
            conditional_likelihood,
            weights,
        )

        return _bound(
            weighted_posterior,
            gamma,
            alpha,
            conditional_likelihood,
            weights,
            locals_weight
        )
    
    ## 
    # M-step functionality to satisfy the PrimModel interface
    ##
    def init_locals(self, n_samples):
        return self.random_state.gamma(
                            100., 
                            1./100., 
                            size=(self.n_components, n_samples)
                        )
    

    def prepare_corpusstate(self, corpus):
        return dict(
                topic_compositions = DataArray(
                        self.init_locals( len(corpus.samples.keys()) ),
                        dims=('component','sample')
                    )
                )

    def spawn_sstats(self, corpus):
        return []
    

    def reduce_sstats(
        self, *args, **kw
    ):
        return self.reduce_sparse_sstats(*args, **kw) 
    

    @staticmethod
    def reduce_sparse_sstats(
        sstats, 
        corpus,
        *,
        gamma,
        **kw,
    ):
        sstats.append(gamma)
        return sstats


    def partial_fit(
        self, sstats, learning_rate
    ):
        
        for corpus_name, gammas in sstats.items():
            alpha0 = self.alpha[corpus_name]
            self.alpha[corpus_name] = _svi_update_fn(
                alpha0,
                update_alpha(alpha0, np.array(gammas)),
                learning_rate
            )
    