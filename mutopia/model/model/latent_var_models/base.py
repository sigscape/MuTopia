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


def polyak_predict(step_fn, init_nk, n_loci, locus_subsample,
                   min_steps=30, max_steps=500, relative_tol=0.01, seed=42,
                   sample_name=None):
    """
    Stochastic prediction with Polyak averaging and variance-based convergence.

    Repeatedly draws random locus subsets, runs E-step to convergence on each,
    and averages the resulting Nk estimates (Polyak averaging). Uses Welford's
    online algorithm to track the sampling variance of the estimator. Stops when
    the relative standard error of the mean drops below ``relative_tol``.

    Parameters
    ----------
    step_fn : callable(locus_indices, warm_start_nk) -> nk_estimate
        Given a 1-D array of locus indices and a warm-start Nk, returns the
        MAP Nk from running the E-step on that locus subset.
    init_nk : np.ndarray
        Initial Nk estimate (shape preserved in output).
    n_loci : int
        Total number of loci in the dataset.
    locus_subsample : float
        Fraction of loci to draw per step.
    min_steps : int
        Minimum number of steps before checking convergence.
    max_steps : int
        Hard upper bound on the number of steps.
    relative_tol : float
        Convergence threshold: ``||SE(mean)|| / ||mean|| < relative_tol``.
    seed : int
        Random seed for reproducibility.
    sample_name : str, optional
        Sample identifier for progress logging.

    Returns
    -------
    np.ndarray
        Polyak-averaged Nk estimate with the same shape as ``init_nk``.
    """
    rng = np.random.RandomState(seed)
    n_subset = max(1, int(locus_subsample * n_loci))
    original_shape = init_nk.shape
    d = init_nk.size
    label = sample_name or "sample"

    nk_mean = np.zeros(d, dtype=np.float32)
    # Welford accumulators for the *proportion* (Nk / sum(Nk))
    prop_mean = np.zeros(d, dtype=np.float32)
    prop_m2 = np.zeros(d, dtype=np.float32)
    warm_start = init_nk.copy()
    rel_se = float("inf")
    nk_history = []

    pbar = tqdm(
        range(1, max_steps + 1),
        desc=f"  {label}",
        ncols=100,
        leave=False,
        postfix={"rel_se": "n/a"},
    )

    try:
        for t in pbar:
            loci = rng.choice(n_loci, n_subset, replace=False)
            nk_t = step_fn(loci, warm_start).ravel()
            nk_history.append(nk_t.copy())

            # Running mean of raw Nk (used as the final estimate)
            nk_mean += (nk_t - nk_mean) / t

            # Welford's algorithm on the normalized proportions
            nk_sum = nk_t.sum()
            prop_t = nk_t / nk_sum if nk_sum > 0 else nk_t
            delta = prop_t - prop_mean
            prop_mean += delta / t
            delta2 = prop_t - prop_mean
            prop_m2 += delta * delta2

            warm_start = nk_mean.reshape(original_shape).copy()

            if t >= 2:
                # SE of the mean proportion
                sample_var = prop_m2 / (t - 1)
                se = np.sqrt(sample_var / t)
                rel_se = np.linalg.norm(se) / (np.linalg.norm(prop_mean) + 1e-10)
                pbar.set_postfix(rel_se=f"{rel_se:.4f}")
                if t >= min_steps and rel_se < relative_tol:
                    pbar.close()
                    logger.info(
                        f"{label}: converged after {t} steps (rel_se={rel_se:.6f})"
                    )
                    break
        else:
            pbar.close()
            logger.warning(
                f"{label}: reached max_steps={max_steps} without converging "
                f"(rel_se={rel_se:.6f}, tol={relative_tol})"
            )
    except KeyboardInterrupt:
        pbar.close()
        logger.info(
            f"{label}: interrupted after {t} steps (rel_se={rel_se:.6f})"
        )

    result = nk_mean.reshape(original_shape)
    # Attach history as a list of per-step Nk estimates for diagnostics.
    # Access via: polyak_predict.last_history after the call returns.
    polyak_predict.last_history = np.array(nk_history).reshape(-1, *original_shape)
    return result


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

    def _predict_svi(
        self,
        dataset,
        factor_model,
        threads=1,
        locus_subsample=1 / 128,
        min_steps=30,
        max_steps=500,
        relative_tol=0.01,
        seed=42,
    ):
        with predict_mode(self):
            predict_fns = self._get_svi_predict_fns(
                dataset,
                factor_model,
                locus_subsample=locus_subsample,
                min_steps=min_steps,
                max_steps=max_steps,
                relative_tol=relative_tol,
                seed=seed,
            )

            logger.info(
                f"SVI predict: {len(dataset.list_samples())} samples, "
                f"locus_subsample={locus_subsample}, tol={relative_tol}"
            )

            Nks = np.array(list(parallel_map(predict_fns, threads=threads)))
            return Nks

    def predict(
        self,
        dataset,
        factor_model,
        threads=1,
        locus_subsample=None,
        **svi_kw,
    ):
        if locus_subsample is not None:
            Nks = self._predict_svi(
                dataset,
                factor_model,
                threads=threads,
                locus_subsample=locus_subsample,
                **svi_kw,
            )
        else:
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
                alpha0, update_alpha(alpha0, Nks).astype(self.dtype), learning_rate,
                parameter_name="alpha"
            )

        return self

    def to_contig(self, x):
        return np.ascontiguousarray(x, dtype=self.dtype)
