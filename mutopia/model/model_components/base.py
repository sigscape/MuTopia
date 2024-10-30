import numpy as np
from sparse import COO, GCXS
from abc import ABC, abstractmethod
import inspect
from numba import njit

class PrimitiveModel(ABC):

    @abstractmethod
    def spawn_sstats(self, corpus):
        raise NotImplementedError
    
    @abstractmethod
    def prepare_corpusstate(self, corpus):
        return dict()

    def update_corpusstate(self, corpus, **kwargs):
        pass
    

class RateModel(PrimitiveModel, ABC):

    @classmethod
    def list_params(cls):
        return inspect.getfullargspec(cls.__init__).args[1:]
    
    @abstractmethod
    def partial_fit(self, sstats, k, corpuses, log_mutation_rates, learning_rate=1.):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, k, corpus):
        raise NotImplementedError


## Shared methods for rate models ##
def get_reg_params(l1_rate, l2_rate):
        return dict(
            alpha = l1_rate+l2_rate,
            l1_ratio = l1_rate/(l1_rate+l2_rate),
        )


def get_poisson_targets_weights(target, eta):
      
    with np.errstate(all='ignore'):
        m = np.nanmean(target/eta)

        sample_weights = eta * m

        return (
                np.nan_to_num(target/sample_weights, nan=0.), 
                sample_weights/sample_weights[sample_weights > 0].mean(), 
            )


def idx_array_to_design(idx_array, n_cols):
    
    n_idx = len(idx_array)

    return GCXS(
                COO(
                    (np.arange(n_idx).astype(int), idx_array.astype(int)),
                    np.ones(n_idx),
                    shape=(n_idx, n_cols)
                ),
                compressed_axes=(0,),
            )


def get_corpus_intercepts(corpuses, encoder : dict, n_repeats = lambda x : 1):
    intercept_idx = np.concatenate([
        np.repeat(encoder[corpus.attrs['name']], n_repeats(corpus), axis=0)
        for corpus in corpuses
    ])    
    return intercept_idx


def get_corpus_design(corpuses, encoder : dict, n_repeats = lambda x : 1):
    return idx_array_to_design(
        get_corpus_intercepts(corpuses, encoder, n_repeats),
        len(encoder)
    )


@njit
def _svi_update_fn(old_value, new_value, learning_rate):
    return (1-learning_rate)*old_value + learning_rate*new_value


@njit
def csr_matmul(X, B):
    
    (ptr, idx, data) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += data[i] * B[idx[i]]

    return out


@njit
def weighted_csr_matmul(X, w, B):
    '''
    Computes X @ w @ B, where X and w are sparse matrices,
    and S is a diagonal matrix represented as a 1D array.
    '''
    (ptr, idx, data) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += data[i] * B[idx[i]] * w[j]

    return out


@njit
def transpose_weighted_csr_matmul(X, w, B):
    '''
    Computes (X @ w)^T @ B, where X and w are sparse matrices,
    and w is a diagonal matrix represented as a 1D array.
    '''
    (ptr, idx, data) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += data[i] * B[idx[i]] * w[idx[i]]

    return out


'''@njit
def rescale_Xy(X, y, weights):
    (ptr, idx, data) = X
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        y[j] *= weights[j]
        for i in range(s, e):
            data[i] *= weights[j]
'''

def jitpartial(func, /, *args):
    @njit
    def newfunc(*fargs):
        return func(*args, *fargs)
    return newfunc


def sp2tup(X):
    '''
    "Sparse to tuple" - converts a scipy.sparse matrix to a tuple of arrays
    '''
    return (X.indptr, X.indices, X.data)

