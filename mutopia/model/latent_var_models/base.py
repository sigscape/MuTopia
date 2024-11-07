
from abc import ABC, abstractmethod
from numba import njit, vectorize
import numpy as np
from numba.extending import get_cython_function_address
import ctypes

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


@njit('double[:](double[:])', nogil=True)
def log_dirichlet_expectation(alpha):
    return psivec(alpha) - psi(np.sum(alpha))


@njit('double(double[:], double[:])', nogil=True)
def dirichlet_bound(alpha, gamma):

    logE_gamma = log_dirichlet_expectation(gamma)
    
    return gammaln(np.sum(alpha)) - gammaln(np.sum(gamma)) + \
        np.sum(gammalnvec(gamma) - gammalnvec(alpha) + (alpha - gamma)*logE_gamma)


class LocalUpdate(ABC):

    @abstractmethod
    def bound(self, gamma, *, corpus, sample, model_state):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def conditional_observation_likelihood(corpus, model_state, **sample_dict):
        raise NotImplementedError
    
    @abstractmethod
    def update_locals(self, gamma0, learning_rate, *, corpus, sample, model_state):
        raise NotImplementedError
    

    @abstractmethod
    def _get_update_fn(
        self,
        learning_rate=1.,
        subsample_rate=1.,
        *,
        corpus,
        sample,
        model_state,
    ):
        raise NotImplementedError
    

    @staticmethod
    def reduce_model_sstats(
        model,
        carry,
        corpus,
        **sample_sstats,
    ):
        return model.reduce_sparse_sstats(
            carry, 
            corpus,
            **sample_sstats
        )