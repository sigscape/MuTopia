import numpy as np
from sparse import COO, GCXS
from abc import ABC, abstractmethod
from numba import njit
import sys
from ..gtensor_interface import GtensorInterface as CS


class PrimitiveModel(ABC):

    @abstractmethod
    def spawn_sstats(self, corpus):
        raise NotImplementedError

    @abstractmethod
    def prepare_corpusstate(self, corpus):
        return dict()

    def update_corpusstate(self, corpus, **kwargs):
        pass

    def post_fit(self, corpuses):
        pass

    def prepare_to_save(self):
        pass


class RateModel(PrimitiveModel, ABC):

    def __init__(self, *corpuses):
        if not all(d in corpuses[0].dims for d in self.requires_dims):
            raise ValueError(
                f"Corpus must have dimensions {self.requires_dims}! "
                f'You\'re missing: {",".join(set(self.requires_dims).difference(corpuses[0].dims))}'
            )

    @property
    @abstractmethod
    def requires_normalization(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def requires_dims(self):
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, k, sstats, exp_offsets, corpuses, learning_rate=1.0):
        raise NotImplementedError

    @abstractmethod
    def predict(self, k, corpus):
        raise NotImplementedError

    # @staticmethod
    # @abstractmethod
    # def get_exp_offset(offsets, corpus):
    #    raise NotImplementedError

    @abstractmethod
    def format_component(self, k, normalization="global"):
        raise NotImplementedError


class SparseDataBase(ABC):

    @abstractmethod
    def predict_sparse(corpus, **idx_dict):
        raise NotImplementedError

    @abstractmethod
    def reduce_sparse_sstats(sstats, corpus, **idx_dict):
        raise NotImplementedError


class DenseDataBase(ABC):

    @abstractmethod
    def reduce_dense_sstats(
        sstats,
        corpus,
        *,
        weighted_posterior,
    ):
        raise NotADirectoryError


def get_feature_classes(corpus, feature):
    feature = CS.get_features(corpus)[feature]
    try:
        return list(feature.attrs["classes"])
    except KeyError:
        raise ValueError(
            f"Feature {feature.name} in {CS.get_name(corpus)} does not have classes defined!"
        )


## Shared methods for rate models ##
def get_reg_params(l1_rate, l2_rate):
    return dict(
        alpha=l1_rate + l2_rate,
        l1_ratio=l1_rate / (l1_rate + l2_rate),
    )


def get_poisson_targets_weights(target, eta):

    with np.errstate(all="ignore"):
        m = np.nanmean(target / eta)

        sample_weights = eta * m

        return (
            np.nan_to_num(target / sample_weights, nan=0.0),
            sample_weights / sample_weights[sample_weights > 0].mean(),
        )


def idx_array_to_design(idx_array, n_cols):

    n_idx = len(idx_array)

    return GCXS(
        COO(
            (np.arange(n_idx).astype(int), idx_array.astype(int)),
            np.ones(n_idx),
            shape=(n_idx, n_cols),
        ),
        compressed_axes=(0,),
    )


def get_corpus_intercepts(corpuses, encoder: dict, n_repeats=lambda x: 1):
    intercept_idx = np.concatenate(
        [
            np.repeat(encoder[CS.get_name(corpus)], n_repeats(corpus), axis=0)
            for corpus in corpuses
        ]
    )
    return intercept_idx


def get_corpus_design(corpuses, encoder: dict, n_repeats=lambda x: 1):
    return idx_array_to_design(
        get_corpus_intercepts(corpuses, encoder, n_repeats), len(encoder)
    )


def _svi_update_fn(old_value, new_value, learning_rate):

    if np.isnan(new_value).any():
        print(
            "\n\rNaN value encountered in update! - if this happens repeatedly later in training, "
            "consider increasing `conditioning_alpha`, `locus_subsample`, or `batch_subsample`",
            file=sys.stderr,
        )
        return old_value

    return (1 - learning_rate) * old_value + learning_rate * new_value


@njit(nogil=True)
def csr_matmul(X, B):

    (ptr, idx, data) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += data[i] * B[idx[i]]

    return out


@njit(nogil=True)
def weighted_csr_matmul(X, w, B):
    """
    Computes X @ w @ B, where X and w are sparse matrices,
    and S is a diagonal matrix represented as a 1D array.
    """
    (ptr, idx, data) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += data[i] * B[idx[i]] * w[j]

    return out


@njit(nogil=True)
def transpose_weighted_csr_matmul(X, w, B):
    """
    Computes (X @ w)^T @ B, where X and w are sparse matrices,
    and w is a diagonal matrix represented as a 1D array.
    """
    (ptr, idx, data) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += data[i] * B[idx[i]] * w[idx[i]]

    return out


@njit(nogil=True)
def design_csr(X, B):

    (ptr, idx, _) = X

    out = np.zeros(len(ptr) - 1)
    for j, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        for i in range(s, e):
            out[j] += B[idx[i]]

    return out


def jitpartial(func, /, *args):
    @njit
    def newfunc(*fargs):
        return func(*args, *fargs)

    return newfunc


def sp2tup(X):
    """
    "Sparse to tuple" - converts a scipy.sparse matrix to a tuple of arrays
    """
    return (X.indptr, X.indices, X.data)


@njit(
    "float32[::1](float32[::1], float32[::1,:])",
    nogil=True,
)
def logsafe_vector_matmul(y, log_x):
    alpha = log_x.max()
    return alpha + np.log(y @ np.exp(log_x - alpha))


@njit(nogil=True)
def logsumexp(x):
    """
    Computes the log-sum-exp of a 1D array.
    """
    alpha = x.max()
    return alpha + np.log(np.sum(np.exp(x - alpha)))
