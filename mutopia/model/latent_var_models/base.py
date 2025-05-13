from abc import abstractmethod
from joblib import delayed
from xarray import DataArray
from tqdm import tqdm
from ...utils import logger, ParContext, parallel_map, parallel_gen
from numba import njit, vectorize, objmode
import numpy as np
from numba.extending import get_cython_function_address
import ctypes
from functools import wraps
from ...gtensor.interfaces import *
from ._dirichlet_update import update_alpha
from .. import gtensor_interface as GT
from ..model_components.base import _svi_update_fn, PrimitiveModel


"""
Numba only works with 2D arrays, not arbitrary tensors. Luckily,
we don't need to know the shape of the tensor to do the updates.
These functions define wrappers around the update functions which
will reshape the tensor to 2D, perform the update, and then reshape
the tensor back to its original shape.

If the tensors passed are C-contiguous, the reshaping operation is 
free, since only the stride is changed. If reshaping the array 
requires copying the data, an error is raised.
"""
def reshape_same_mem(arr, shape):
    out = arr.reshape(shape, order="C")
    if not np.shares_memory(arr, out):
        raise ValueError("Memory was not shared")
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
        remainder_shape = conditional_likelihood.shape[1:]
        assert (
            remainder_shape == weights.shape
        ), "conditional_likelihood and weights must have the same trailing shape"
        return fn(
            alpha,
            reshape_same_mem(conditional_likelihood, (K, -1)),
            reshape_same_mem(weights, (-1,)),
            *args,
        )

    return wrapper


def flatten_last_arg(fn):
    @wraps(fn)
    def wrapper(*args):
        K = args[-1].shape[0]
        return fn(*args[:-1], reshape_same_mem(args[-1], (K, -1)))

    return wrapper


"""
Here we're defining numba-compatible functions for the
psi and gammaln functions from scipy.special, referencing the 
cython code underlying the scipy.special module.
"""
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1psi")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
psi = functype(addr)


@vectorize(["double(double)"])
def psivec(x):
    return psi(x)


gammaln = functype(
    get_cython_function_address("scipy.special.cython_special", "gammaln")
)


@vectorize(["double(double)"])
def gammalnvec(x):
    return gammaln(x)

@njit("float32[:](float32[:])", nogil=True)
def psivec32(x):
    return psivec(x).astype(np.float32)

@njit("float32[::1](float32[::1])", nogil=True)
def exp_log_dirichlet_expectation(alpha):
    return np.exp(psivec(alpha) - psi(np.sum(alpha))).astype(np.float32)


@njit(
    "float32[::1](float32[::1], float32[:,::1], float32[::1], float32[::1])", 
    nogil=True
)
def _update_step(
    alpha,
    conditional_likelihood,
    weights,
    Nk,
):

    exp_Elog_prior = np.exp( psivec32(alpha + Nk) )

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    Nk = exp_Elog_prior * (conditional_likelihood @ X_div_X_tild)

    return Nk


@flatten_tensor_for_update
@njit(
    "float32[::1](float32[::1], float32[:,::1], float32[::1], int64, double, float32[::1])",
    nogil=True,
)
def iterative_update(
    alpha,
    conditional_likelihood,
    weights,
    iters,
    tol,
    Nk,  # move Nk to the end so we can curry the function
):
    for _ in range(iters):  # inner e-step loop

        old_Nk = Nk.copy()
        Nk = _update_step(
            alpha,
            conditional_likelihood,
            weights,
            Nk,
        )

        if (np.abs(Nk - old_Nk) / np.sum(old_Nk)).sum() < tol:
            break

    return Nk


@reshape_output
@flatten_tensor_for_update
@njit(
    "float32[:,::1](float32[::1], float32[:,::1], float32[::1], float32[::1])",
    nogil=True,
)
def calc_local_variables(
    alpha,
    conditional_likelihood,
    weights,
    Nk,
):
    exp_Elog_prior = np.exp( psivec32(alpha + Nk) )

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    phi_matrix = np.outer(exp_Elog_prior, X_div_X_tild) * conditional_likelihood

    return phi_matrix


"""
@njit("double(double[:], double[:])", nogil=True)
def dirichlet_bound(alpha, Nk):

    logE_Nk = log_dirichlet_expectation(Nk).astype(np.float64)

    return (
        Nkln(np.sum(alpha))
        - Nkln(np.sum(Nk))
        + np.sum(Nklnvec(Nk) - Nklnvec(alpha) + (alpha - Nk) * logE_Nk)
    )
    
@flatten_tensor_for_update
@flatten_last_arg
@njit(
    "double(double[::1], double[:,::1], double[::1], double[::1], double[:,::1])",
    nogil=True,
)
def bound(
    alpha,
    conditional_likelihood,
    weights,
    Nk,
    weighted_posterior,
):

    phi = weighted_posterior / weights[None, :]
    entropy_sstats = -np.sum(weighted_posterior * np.where(phi > 0, np.log(phi), 0.0))
    entropy_sstats += dirichlet_bound(alpha, Nk)

    flattened_logweight = log_dirichlet_expectation(Nk)[:, None] + np.log(
        conditional_likelihood
    )

    return (
        np.sum(weighted_posterior * np.nan_to_num(flattened_logweight, nan=0.0))
        + entropy_sstats
    )
"""

"""
Annealed importance sampling (AIS) for marginal likelihood estimation
"""


@njit(nogil=True)
def categorical_draw(logits):
    logits = np.exp(logits - logits.max())
    cdf = np.cumsum(logits)
    z = cdf[-1]
    u = np.random.uniform(0, 1)
    return np.searchsorted(cdf, u * z)


@njit(
    "Tuple((float32[::1],int64[::1],float32))(float32[::1], float32[:,::1], float32[::1], float32[::1], int64[::1], float32)",
    nogil=True,
)
def _gibbs_sample_step(
    alpha,
    log_conditional_likelihood,
    weights,
    Nk,
    z,
    temperature,
):

    log_weight = 0.0
    I = log_conditional_likelihood.shape[0]
    sampling_order = np.random.permutation(I)

    for i in sampling_order:
        Nk[z[i]] -= weights[i]
        logits = temperature * log_conditional_likelihood[i] + np.log(alpha + Nk)
        z[i] = categorical_draw(logits)
        Nk[z[i]] += weights[i]
        log_weight += weights[i] * log_conditional_likelihood[i, z[i]]

    return Nk, z, log_weight


@njit(nogil=True)
def _gibbs_sample_posterior(
    alpha,
    log_conditional_likelihood,
    weights,
    Nk,
    z,
    steps,
    warmup,
    quiet,
):

    posterior = np.zeros_like(log_conditional_likelihood)
    Nks = np.zeros((steps, Nk.shape[0]), dtype=np.float32)

    for t in range(steps + warmup):
        Nk, z, _ = _gibbs_sample_step(
            alpha, log_conditional_likelihood, weights, Nk, z, min(1.0, t / warmup)
        )

        if t >= warmup:
            for i in range(len(z)):
                posterior[i, z[i]] += 1 / steps

            # Nk += Nk/steps
            Nks[t - warmup] = Nk

        if not quiet and (t % 500 == 0) and t > warmup:
            with objmode():
                logger.info(
                    "Completed "
                    + str(t - warmup)
                    + "/"
                    + str(steps)
                    + " Gibb's sampling steps."
                )

    return (posterior, Nks)


def gibbs_sample_posterior(
    alpha,
    conditional_likelihood,
    weights,
    steps=5000,
    warmup=1000,
    seed=42,
    quiet=False,
):

    K, I = conditional_likelihood.shape
    np.random.seed(seed)

    log_conditional_likelihood = np.ascontiguousarray(
        np.log(conditional_likelihood).T, dtype=np.float32
    )
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)

    z = np.random.choice(K, size=I, p=alpha / np.sum(alpha))
    z = np.ascontiguousarray(z, dtype=np.int64)

    Nk = np.array(
        [weights[z == k].sum() for k in range(K)],
        dtype=np.float32,
    )
    Nk = np.ascontiguousarray(Nk, dtype=np.float32)

    posterior, Nk = _gibbs_sample_posterior(
        alpha,
        log_conditional_likelihood,
        weights,
        Nk,
        z,
        steps,
        warmup,
        quiet,
    )

    return posterior.T, Nk


@njit(nogil=True)
def _ais_inner(
    alpha,
    log_conditional_likelihood,
    weights,
    Nk,
    z,
    iters,
):
    logweights = np.zeros(iters, dtype=np.float32)
    ais_weight = 0

    for t in range(1, iters + 1):

        Nk, z, log_weight = _gibbs_sample_step(
            alpha,
            log_conditional_likelihood,
            weights,
            Nk,
            z,
            t / iters,
        )

        ais_weight += log_weight * 1 / iters
        logweights[t] = log_weight

    return (
        ais_weight,
        logweights,
        Nk,
    )


def AIS_marginal_ll(
    alpha,
    conditional_likelihood,
    weights,
    steps=100000,
    seed=42,
):

    K, I = conditional_likelihood.shape
    np.random.seed(seed)

    log_conditional_likelihood = np.ascontiguousarray(
        np.log(conditional_likelihood).T, dtype=np.float32
    )
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)

    z = np.random.choice(K, size=I, p=alpha / np.sum(alpha))
    z = np.ascontiguousarray(z, dtype=np.int64)

    Nk = np.array(
        [weights[z == k].sum() for k in range(K)],
        dtype=np.float32,
    )
    Nk = np.ascontiguousarray(Nk, dtype=np.float32)

    return _ais_inner(
        alpha,
        log_conditional_likelihood,
        weights,
        Nk,
        z,
        steps,
    )


def random_locals(random_state, n_components):
    def init_locals(dataset, sample_name):
        return random_state.gamma(100.0, 1.0 / 100.0, size=(n_components,))

    return init_locals


class LocalsModel(PrimitiveModel):

    def __init__(
        self,
        GT, # gtensor_interface
        datasets,
        prior_alpha=1.0,
        estep_iterations=1000,
        difference_tol=5e-5,
        dtype=np.float32,
        *,
        n_components,
        random_state,
        **kw,
    ):
        self.estep_iterations = estep_iterations
        self.difference_tol = difference_tol
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype
        self.GT = GT

        self.alpha = {
            name: np.ones(n_components, dtype=dtype) * prior_alpha
            for name, _ in self.GT.expand_datasets(*datasets)
        }

    def update_prior(self, datasets):

        sstats = {
            name: [
                self.GT.fetch_topic_compositions(dataset, sname)
                for sname in self.GT.list_samples(dataset)
            ]
            for name, dataset in self.GT.expand_datasets(*datasets)
        }

        self.partial_fit(sstats, learning_rate=1.0)

    def Estep(
        self,
        datasets,
        factor_model,
        learning_rate=1.0,
        locus_subsample=1.0,
        batch_subsample=1.0,
        *,
        par_context,
    ):
        """
        A suffstats dictionary with the following structure:
        sstats[parameter_name][corpus_name] <- suffstats
        """
        sstats = factor_model.get_sstats_dict(datasets)
        elbo = 0.0

        """
        Construct the update function for each sample from the dataset
        in a generator fashion. Curry the initial Nk value for the 
        update, but don't call the update function yet - we want to
        pass this expression to the multiprocessing pool.
        """

        kw = dict(
            learning_rate=learning_rate,
            locus_subsample=locus_subsample,
            batch_subsample=batch_subsample,
            par_context=par_context,
        )

        for dataset in datasets:
            
            update_fns = parallel_gen(
                self._get_update_fns(dataset, factor_model, **kw),
                par_context=par_context,
                ordered=True,
            )

            for sample_name, sample_suffstats in zip(
                self.GT.list_samples(dataset),
                update_fns,
            ):
                # Update the topic compositions for the sample
                self.GT.update_topic_compositions(dataset, sample_name, sample_suffstats["Nk"])

                for model_name, model in factor_model.models.items():
                    """
                    The latent variables model handle the observation data format
                    and the Nk updates. Here, the model state just delegates the
                    suffstat reduction back to the latent variables model, which
                    calls the model's reduce_sstats method depending on the data type.
                    """
                    self.reduce_model_sstats(
                        model,
                        sstats[model_name + "_sstats"],
                        dataset,
                        **sample_suffstats,
                    )

        return sstats

    def predict(self, dataset, factor_model, par_context=None):
        self.estep_iterations = 10000
        self.difference_tol = 5e-5

        subsample_rate = (
            dataset.regions.context_frequencies.sum().item()
            / factor_model.get_genome_size(dataset)
        )

        update_fns = self._get_update_fns(
            (dataset,),
            factor_model,
            par_context=par_context,
            exposures_fn=random_locals(np.random.RandomState(1776), self.n_components),
            locus_subsample=subsample_rate,
        )

        Nks = []
        for s in tqdm(
            parallel_map(update_fns, par_context),
            ncols=100,
            desc="Estimating contributions",
        ):
            Nks.append(s[1])

        return DataArray(
            np.array(Nks),
            dims=("sample", "component"),
        )

    def predict_sample(
        self,
        sample,
        dataset,
        factor_model,
    ):
        self.estep_iterations = 10000
        self.difference_tol = 5e-5

        with ParContext(1) as par:
            return self.predict(
                SampleCorpusFusion(CorpusInterface(dataset), sample),
                factor_model,
                par_context=par,
            )

    def deviance(
        self,
        factor_model,
        datasets,
        exposures_fn=None,
        par_context=None,
    ):

        factor_model.update_normalizers(datasets, par_context)

        kw = dict(
            factor_model=factor_model,
            exposures_fn=exposures_fn,
            par_context=par_context,
        )

        dev_fns = (
            fn
            for dataset in datasets
            for fn in self._get_deviance_fns(dataset, **kw)
        )

        d_fit, d_null = list(zip(*parallel_gen(dev_fns, par_context, ordered=False)))

        return 1 - sum(d_fit) / sum(d_null)

    @abstractmethod
    def _get_update_fns(
        self,
        dataset,
        factor_model,
        learning_rate=1.0,
        subsample_rate=1.0,
        exposures_fn=None,
        *,
        par_context,
    ):
        raise NotImplementedError

    @abstractmethod
    def _get_deviance_fns(
        self,
        dataset,
        factor_model,
        exposures_fn=None,
        par_context=None,
    ):
        raise NotImplementedError

    @staticmethod
    def reduce_model_sstats(
        model,
        carry,
        dataset,
        **sample_sstats,
    ):
        raise NotImplementedError

    ##
    # M-step functionality to satisfy the PrimModel interface
    ##
    def init_locals(self, n_samples):
        return self.random_state.gamma(
            100.0,
            1.0 / 100.0,
            size=(self.n_components, n_samples),
        ).astype(self.dtype)

    
    def prepare_corpusstate(self, dataset):

        n_observations = np.array([
            sample.X.sum().data.item()
            for _, sample in self.GT.iter_samples(dataset)
        ])

        if "ploidy" in dataset:
            weighted_ploidy = (
                (n_observations/n_observations.sum()) @ 
                dataset["ploidy"].transpose("sample", "locus").data
            ) + 1
        else:
            weighted_ploidy = np.ones(dataset.sizes["locus"], dtype=self.dtype)

        return dict(
            topic_compositions=DataArray(
                self.init_locals(len(n_observations)),
                dims=("component", "sample"),
            ),
            n_observations=DataArray(
                n_observations,
                dims=("sample",),
            ),
            weighted_ploidy=DataArray(
                weighted_ploidy,
                dims=("locus",),
            ),
        )

    def spawn_sstats(self, dataset):
        return []

    @staticmethod
    def reduce_sparse_sstats(
        sstats,
        dataset,
        *,
        Nk,
        **kw,
    ):
        sstats.append(Nk)
        return sstats

    @staticmethod
    def reduce_dense_sstats(
        sstats,
        dataset,
        *,
        Nk,
        **kw,
    ):
        sstats.append(Nk)
        return sstats

    def partial_fit(self, sstats, learning_rate):
        for corpus_name, Nks in sstats.items():
            alpha0 = self.alpha[corpus_name]
            self.alpha[corpus_name] = _svi_update_fn(
                alpha0, update_alpha(alpha0, np.array(Nks)), learning_rate
            )

    def to_contig(self, x):
        return np.ascontiguousarray(x, dtype=self.dtype)
