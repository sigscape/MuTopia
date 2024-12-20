from abc import ABC, abstractmethod
from joblib import delayed
from xarray import DataArray

from numba import njit, vectorize
import numpy as np
from numba.extending import get_cython_function_address
import ctypes
from functools import wraps

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
#functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
functype = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)
psi = functype(addr)

@vectorize(['float32(float32)'])
def psivec(x):
    return psi(x)


gammaln = functype(get_cython_function_address("scipy.special.cython_special", "gammaln"))
@vectorize(['float32(float32)'])
def gammalnvec(x):
    return gammaln(x)


'''
Helper functions for the dirichlet log likelihood and bound
'''
@njit('float32[:](float32[:])', nogil=True)
def log_dirichlet_expectation(alpha):
    return psivec(alpha) - psi(np.sum(alpha))


@njit('float32(float32[:], float32[:])', nogil=True)
def dirichlet_bound(alpha, gamma):

    logE_gamma = log_dirichlet_expectation(gamma)
    
    return gammaln(np.sum(alpha)) - gammaln(np.sum(gamma)) + \
        np.sum(gammalnvec(gamma) - gammalnvec(alpha) + (alpha - gamma)*logE_gamma)



@njit('float32[::1](float32[::1], float32[:,::1], float32[::1], float32[::1])', nogil=True)
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
        np.float32(0.),
    )
    gamma_sstats = exp_Elog_gamma*(conditional_likelihood @ X_div_X_tild)

    return alpha + gamma_sstats


@flatten_tensor_for_update
@njit('float32[::1](float32[::1], float32[:,::1], float32[::1], int64, double, float32[::1])', nogil=True)
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
@njit('float32[:,::1](float32[::1], float32[:,::1], float32[::1], float32[::1])', nogil=True)
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
@njit('float32(float32[::1], float32[:,::1], float32[::1], float32[::1], float32[:,::1])', nogil=True)
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



class LocalUpdate(ABC):

    def predict(
        self, 
        corpus, 
        model_state, 
        *, 
        parallel_context
    ):
        _, update_fns = self.get_update_fns(
            (corpus,),
            model_state,
            parallel_context=parallel_context,
        )
            
        gammas = map(
            lambda x : x[1],
            parallel_context(delayed(update_fn)() for update_fn in update_fns)
        )
        
        return DataArray(
            np.array(list(gammas)),
            dims=('sample', 'component'),
        )


    @abstractmethod
    def get_update_fns(
        self,
        corpuses,
        model_state,
        learning_rate=1.,
        subsample_rate=1.,
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