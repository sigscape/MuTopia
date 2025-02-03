from abc import ABC, abstractmethod
from joblib import delayed
from xarray import DataArray
from tqdm import tqdm
from numba import njit, vectorize
import numpy as np
from numba.extending import get_cython_function_address
import ctypes
from functools import wraps
from ...corpus.interfaces import *
from ...utils import ParContext
from ._dirichlet_update import update_alpha
from ..corpus_state import CorpusState as CS
from ..model_components.base import _svi_update_fn, PrimitiveModel

'''
Numba only works with 2D arrays, not arbitrary tensors. Luckily,
we don't need to know the shape of the tensor to do the updates.
These functions define wrappers around the update functions which
will reshape the tensor to 2D, perform the update, and then reshape
the tensor back to its original shape.

If the tensors passed are C-contiguous, the reshaping operation is 
free, since only the stride is changed. If reshaping the array 
requires copying the data, an error is raised.
'''
def reshape_same_mem(arr, shape):
    out = arr.reshape(shape, order='C')
    if not np.shares_memory(arr, out):
        raise ValueError('Memory was not shared')
    return out


def reshape_output(fn):
    @wraps(fn)
    def wrapper(alpha, conditional_likelihood, weights, *args):
        tensor_dim = conditional_likelihood.shape
        out = fn(alpha, conditional_likelihood, weights, *args)
        return reshape_same_mem(out, tensor_dim)
    return wrapper


def flatten_tensor_for_update(fn):
    @wraps(fn)
    def wrapper(alpha, conditional_likelihood, weights, *args):
        K = conditional_likelihood.shape[0]
        assert conditional_likelihood.shape[1:] == weights.shape, \
            'conditional_likelihood and weights must have the same trailing shape'
        return fn(
            alpha,
            reshape_same_mem(conditional_likelihood, (K,-1)),
            reshape_same_mem(weights, (-1,)),
            *args
        )
    return wrapper


def flatten_last_arg(fn):
    @wraps(fn)
    def wrapper(*args):
        K = args[-1].shape[0]
        return fn(*args[:-1], reshape_same_mem(args[-1], (K,-1)))
    return wrapper



'''
Here we're defining numba-compatible functions for the
psi and gammaln functions from scipy.special, referencing the 
cython code underlying the scipy.special module.
'''
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1psi")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
psi = functype(addr)

@vectorize(['double(double)'])
def psivec(x):
    return psi(x)


gammaln = functype(get_cython_function_address("scipy.special.cython_special", "gammaln"))
@vectorize(['double(double)'])
def gammalnvec(x):
    return gammaln(x)


'''
Helper functions for the dirichlet log likelihood and bound
'''
@njit('double[:](double[:])', nogil=True)
def log_dirichlet_expectation(alpha):
    return psivec(alpha) - psi(np.sum(alpha))


@njit('double(double[:], double[:])', nogil=True)
def dirichlet_bound(alpha, gamma):

    logE_gamma = log_dirichlet_expectation(gamma)
    
    return gammaln(np.sum(alpha)) - gammaln(np.sum(gamma)) + \
        np.sum(gammalnvec(gamma) - gammalnvec(alpha) + (alpha - gamma)*logE_gamma)



@njit('double[::1](double[::1], double[:,::1], double[::1], double[::1])', nogil=True)
def _update_step(
        alpha, 
        conditional_likelihood, 
        weights,
        gamma,
    ):

    exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma))
    
    '''# NxK @ K => N
    X_tild = exp_Elog_gamma @ conditional_likelihood
    #                                    KxN @ N => K
    gamma_sstats = exp_Elog_gamma*(conditional_likelihood @ (weights/X_tild))'''

    X_div_X_tild = np.where(
        weights>0.,
        weights/(exp_Elog_gamma @ conditional_likelihood),
        0.
    )
    gamma_sstats = exp_Elog_gamma*(conditional_likelihood @ X_div_X_tild)

    return alpha + gamma_sstats


@flatten_tensor_for_update
@njit('double[::1](double[::1], double[:,::1], double[::1], int64, double, double[::1])', nogil=True)
def iterative_update(
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
            alpha, 
            conditional_likelihood, 
            weights,
            gamma,
        )

        if (np.abs(gamma - old_gamma)/np.sum(old_gamma)).sum() < tol:
            break

    return gamma


@reshape_output
@flatten_tensor_for_update
@njit('double[:,::1](double[::1], double[:,::1], double[::1], double[::1])', nogil=True)
def calc_local_variables(
        alpha,
        conditional_likelihood,
        weights,
        gamma,
    ):
    exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma))

    X_tild = exp_Elog_gamma @ conditional_likelihood

    phi_matrix = np.outer(exp_Elog_gamma, weights/X_tild)*conditional_likelihood

    return phi_matrix


@flatten_tensor_for_update
@flatten_last_arg
@njit('double(double[::1], double[:,::1], double[::1], double[::1], double[:,::1])', nogil=True)
def bound(
    alpha,
    conditional_likelihood,
    weights,
    gamma,
    weighted_posterior,
    ):

    phi = weighted_posterior/weights[None,:]
    entropy_sstats = -np.sum(weighted_posterior * np.where(phi > 0, np.log(phi), 0.))
    entropy_sstats += dirichlet_bound(alpha, gamma)
    
    flattened_logweight = log_dirichlet_expectation(gamma)[:,None] + np.log(conditional_likelihood)
    
    return np.sum(weighted_posterior * np.nan_to_num(flattened_logweight, nan=0.)) + entropy_sstats


@flatten_tensor_for_update
@njit(nogil=True)
def mixture_update_step(
    gamma, # D*K
    delta, # D
    alpha, # D*K
    tau, # D
    conditional_likelihood,  # *(D*K)xI
    weights, # I
    fraction_map, # DxD*K,
):

    Elog_gamma = log_dirichlet_expectation(gamma)
    exp_Elog_gamma = np.exp(Elog_gamma) # D*K
    exp_Elog_delta = np.exp(log_dirichlet_expectation(delta)) # D
    
    # (D @ D*K) => D*K) * D*K => D*K
    exp_Elog_prior = (exp_Elog_delta @ fraction_map) * exp_Elog_gamma

    X_div_X_tild = np.where(
        weights>0.,
        weights/(exp_Elog_prior @ conditional_likelihood),
        0.
    ) # (I,)
    
    # (D*KxI @ I => D*K) * D*K => D*K
    sstats = (conditional_likelihood @ X_div_X_tild)*exp_Elog_prior

    weighted_phi = np.outer(exp_Elog_prior, X_div_X_tild)*conditional_likelihood
    phi = weighted_phi/weights[None,:]
    entropy_sstats = -np.sum(weighted_phi * np.where(phi > 0, np.log(phi), 0.))

    prior_elbo = dirichlet_bound(tau, delta)
    
    M = fraction_map.T
    prior_elbo += np.sum(
        gammalnvec(alpha @ M) - gammalnvec(gamma @ M) + \
            (gammalnvec(gamma) - gammalnvec(alpha) + (alpha - gamma) * Elog_gamma) @ M
    )

    flattened_logweight = np.log(exp_Elog_prior)[:,None] + np.log(conditional_likelihood)
    elbo = np.sum(weighted_phi * np.nan_to_num(flattened_logweight, nan=0.)) + entropy_sstats + prior_elbo

    return (
        alpha + sstats, 
        tau + sstats @ fraction_map.T,
        elbo,
    )


def random_locals(random_state, n_components):
    def init_locals(corpus, sample_name):
        return random_state.gamma(100., 1./100., size=(n_components,))
    
    return init_locals



class LocalUpdate(PrimitiveModel):

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
            CS.get_name(corpus) : np.ones(n_components, dtype=dtype)*prior_alpha
            for corpus in corpuses
        }


    def predict(
        self, 
        corpus, 
        model_state, 
        *, 
        parallel_context
    ):
        self.estep_iterations=10000
        self.difference_tol=5e-5

        subsample_rate = corpus.regions.context_frequencies.sum().item()/model_state.get_genome_size(corpus)

        samples, update_fns = self.get_update_fns(
            (corpus,),
            model_state,
            parallel_context=parallel_context,
            exposures_fn=random_locals(np.random.RandomState(1776), self.n_components),
            locus_subsample=subsample_rate,
        )
            
        gammas = []
        for s in tqdm(
            parallel_context(delayed(update_fn)() for update_fn in update_fns),
            total=len(samples),
            ncols=100,
            desc='Estimating contributions',
        ):
            gammas.append(s[1])
        
        return DataArray(
            np.array(gammas),
            dims=('sample', 'component'),
        )
    

    def predict_sample(
        self,
        sample,
        corpus,
        model_state,
    ):
        self.estep_iterations=10000
        self.difference_tol=5e-5

        with ParContext(1) as par:
            return self.predict(
                SampleCorpusFusion(CorpusInterface(corpus), sample),
                model_state,
                parallel_context=par,
            ).ravel()


    @abstractmethod
    def get_update_fns(
        self,
        corpuses,
        model_state,
        learning_rate=1.,
        subsample_rate=1.,
        exposures_fn=CS.fetch_topic_compositions,
        *,
        parallel_context
    ):
        raise NotImplementedError
    

    @abstractmethod
    def get_deviance_fns(
        self,
        corpuses,
        model_state,
        exposures_fn=None,
        *,
        parallel_context,
    ):
        raise NotImplementedError
    

    @staticmethod
    def reduce_model_sstats(
        model,
        carry,
        corpus,
        **sample_sstats,
    ):
        raise NotImplementedError
    

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
                self.init_locals( len(CS.list_samples(corpus)) ),
                dims=('component','sample')
            )
        )


    def spawn_sstats(self, corpus):
        return []
    

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