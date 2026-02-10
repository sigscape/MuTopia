from abc import abstractmethod
from xarray import DataArray
from tqdm import tqdm
from numba import njit, vectorize, objmode
import numpy as np
from numba.extending import get_cython_function_address
import ctypes
from functools import wraps, partial
from contextlib import contextmanager
from mutopia.utils import logger, parallel_map, parallel_gen
from ._dirichlet_update import update_alpha
from ..model_components.base import _svi_update_fn
from ..gtensor_interface import GtensorInterface


@contextmanager
def predict_mode(model, estep_iterations=100_000, difference_tol=1e-5):
    """
    A context manager to temporarily set the model to prediction mode
    by adjusting the E-step parameters.
    """
    logger.info("Setting model to prediction mode.")
    old_iters = model.estep_iterations
    old_tol = model.difference_tol

    model.estep_iterations = estep_iterations
    model.difference_tol = difference_tol
    try:
        yield model
    finally:
        model.estep_iterations = old_iters
        model.difference_tol = old_tol


def delayed(fn, *args, **kwargs):
    """
    A decorator to delay the execution of a function until it is called.
    This is useful for functions that are expensive to compute and should
    only be executed when needed.
    """
    run = partial(fn, *args, **kwargs)

    @wraps(fn)
    def wrapper(*wargs, **wkwargs):
        return run()

    return wrapper


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


@njit("float32[:](float32[:])", nogil=True)
def psivec32(x):
    return psivec(x).astype(np.float32)


@njit("float32(float32)", nogil=True)
def psi32(x):
    return np.float32(psi(x))


gammaln = functype(
    get_cython_function_address("scipy.special.cython_special", "gammaln")
)


@njit("float32(float32)", nogil=True)
def gammaln32(x):
    return np.float32(gammaln(x))


@vectorize(["double(double)"])
def gammalnvec(x):
    return gammaln(x)


@njit("float32[:](float32[:])", nogil=True)
def gammalnvec32(x):
    return gammalnvec(x).astype(np.float32)


@njit("float32[::1](float32[::1])", nogil=True)
def exp_log_dirichlet_expectation(alpha):
    return np.exp(psivec(alpha) - psi(np.sum(alpha))).astype(np.float32)


@njit("float32(float32[::1], float32[::1])", nogil=True)
def dirichlet_bound(alpha, x):

    logE_x = np.log(exp_log_dirichlet_expectation(x))

    return (
        gammaln32(np.sum(alpha))
        - gammaln32(np.sum(x))
        + np.sum(gammalnvec32(x) - gammalnvec32(alpha) + (alpha - x) * logE_x)
    )


@njit(
    "float32[::1](float32[::1], float32[:,::1], float32[::1], float32[::1])", nogil=True
)
def _update_step(
    alpha,
    conditional_likelihood,
    weights,
    Nk,
):
    exp_Elog_prior = np.exp(psivec32(alpha + Nk))

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

        check_tol = (np.abs(Nk - old_Nk) / np.sum(old_Nk)).sum()
        if check_tol < tol:
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
    exp_Elog_prior = np.exp(psivec32(alpha + Nk))

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    phi_matrix = np.outer(exp_Elog_prior, X_div_X_tild) * conditional_likelihood

    return phi_matrix


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


def _just_next_Nk(Nks):
    i = iter(Nks)

    def _next_Nk(*args, **kw):
        return next(i)

    return _next_Nk


class LocalsModel:

    def __init__(
        self,
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
        self.GT = GtensorInterface()

        self.alpha = {
            name: np.ones(n_components, dtype=dtype) * prior_alpha
            for name, _ in self.GT.expand_datasets(*datasets)
        }

    def get_alpha(self, dataset) -> np.ndarray:
        return self.alpha[self.GT.get_name(dataset)]
    
    def update_prior(self, datasets):

        sstats = {
            name: [
                self.GT.fetch_topic_compositions(dataset, sname)
                for sname in self.GT.list_samples(dataset)
            ]
            for name, dataset in self.GT.expand_datasets(*datasets)
        }

        self.Mstep(sstats, learning_rate=1.0)

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
                if sample_suffstats is None:
                    continue
                # Update the topic compositions for the sample
                self.GT.update_topic_compositions(
                    dataset, sample_name, sample_suffstats["Nk"]
                )

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
    
    def _predict(self, dataset, factor_model, threads=1, estep_iterations=100_000, difference_tol=1e-5):

        with predict_mode(self, estep_iterations=estep_iterations, difference_tol=difference_tol):

            Nk = self.init_locals(dataset)

            update_fns = self._get_update_fns(
                dataset,
                factor_model,
                exposures_fn=_just_next_Nk(Nk),
            )

            update_fns = tqdm(
                update_fns,
                total=len(dataset.list_samples()),
                ncols=100,
                desc="Estimating contributions",
            )

            Nks = np.array(
                [stats["Nk"] for stats in parallel_map(update_fns, threads=threads)]
            )
            return Nks

    def predict(
        self,
        dataset,
        factor_model,
        threads=1,
    ):
        Nks = self._predict(dataset, factor_model=factor_model, threads=threads)
        n_sources = self.GT.n_sources(dataset)
        Nks = Nks.reshape((-1, n_sources, self.n_components)).transpose((1, 0, 2))
        return DataArray(Nks, dims=("source", "sample", "component"))

    def score(
        self,
        factor_model,
        datasets,
        exposures_fn=None,
        par_context=None,
    ):
        kw = dict(
            factor_model=factor_model,
            exposures_fn=exposures_fn,
            par_context=par_context,
        )

        dev_fns = (
            fn for dataset in datasets for fn in self._get_deviance_fns(dataset, **kw)
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

    def _get_sample_init_fn(self, dataset):
        return delayed(
            self.random_state.gamma,
            100.0,
            1.0 / 100.0,
            size=(
                1,
                self.n_components,
            ),
        )

    def init_locals(self, dataset):
        init_fn = self._get_sample_init_fn(dataset)
        return self.to_contig(
            np.array([init_fn(sample) for _, sample in self.GT.iter_samples(dataset)])
        )

    def prepare_corpusstate(self, dataset):
        return dict(
            topic_compositions=DataArray(
                self.init_locals(dataset).astype(self.dtype),
                dims=("sample", "source", "component"),
            ),
        )

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

    def Mstep(self, datasets, learning_rate=1.0):

        for name, dataset in self.GT.expand_datasets(*datasets):

            Nks = (
                self.GT.fetch_val(dataset, "topic_compositions")
                .transpose("sample", ...)
                .data
            )
            Nks = np.squeeze(Nks, axis=1)

            alpha0 = self.alpha[name]

            self.alpha[name] = _svi_update_fn(
                alpha0, update_alpha(alpha0, Nks).astype(self.dtype), learning_rate
            )

        return self

    def to_contig(self, x):
        return np.ascontiguousarray(x, dtype=self.dtype)
