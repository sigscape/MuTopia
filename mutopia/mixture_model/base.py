from numba import njit
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
from collections import defaultdict
from ..model.model_components.base import idx_array_to_design, _svi_update_fn
from ..model.latent_var_models.base import *
from .mixture_interface import MixtureInterface
from os.path import basename


def reshape_output(fn):
    @wraps(fn)
    def wrapper(alpha, tau, fraction_map, conditional_likelihood, weights, *args):
        tensor_dim = conditional_likelihood.shape
        out = fn(alpha, tau, fraction_map, conditional_likelihood, weights, *args)
        return reshape_same_mem(out, tensor_dim)

    return wrapper


def flatten_tensor_for_update(fn):
    @wraps(fn)
    def wrapper(alpha, tau, fraction_map, conditional_likelihood, weights, *args):
        K = conditional_likelihood.shape[0]
        remainder_shape = conditional_likelihood.shape[1:]
        assert (
            remainder_shape == weights.shape
        ), "conditional_likelihood and weights must have the same trailing shape"
        return fn(
            alpha,
            tau,
            fraction_map,
            reshape_same_mem(conditional_likelihood, (K, -1)),
            reshape_same_mem(weights, (-1,)),
            *args,
        )

    return wrapper


"""
Mixture model inference functions
"""


@njit(
    "float32[::1](float32[::1], float32[::1], float32[:,:], float32[:,::1], float32[::1], float32[::1])",
    nogil=True,
)
def _update_step(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    conditional_likelihood,  # D*K x I
    weights,  # I,
    Nk,
):

    # D*K
    exp_Elog_prior = np.exp(
        psivec32(alpha + Nk)
        - psivec32((alpha + Nk) @ fraction_map.T) @ fraction_map
        + psivec32(tau + (Nk @ fraction_map.T)) @ fraction_map
    )

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    Nk = exp_Elog_prior * (conditional_likelihood @ X_div_X_tild)

    return Nk


@flatten_tensor_for_update
@njit(
    "float32[::1](float32[::1], float32[::1], float32[:,:], float32[:,::1], float32[::1], int64, float32, float32[::1])",
    nogil=True,
)
def iterative_update(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    conditional_likelihood,
    weights,
    iters,
    tol,
    Nk,  # move gamma to the end so we can curry the function
):
    for _ in range(iters):  # inner e-step loop

        old_Nk = Nk.copy()
        Nk = _update_step(
            alpha,  # D*K
            tau,  # D
            fraction_map,  # Dx(D*K)
            conditional_likelihood,
            weights,
            Nk,
        )

        if (np.abs(Nk - old_Nk) / np.sum(Nk)).sum() < tol:
            break

    return Nk


@reshape_output
@flatten_tensor_for_update
@njit(
    "float32[:,::1](float32[::1], float32[::1], float32[:,:], float32[:,::1], float32[::1], float32[::1])",
    nogil=True,
)
def calc_local_variables(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    conditional_likelihood,
    weights,
    Nk,
):
    exp_Elog_prior = np.exp(
        psivec32(alpha + Nk)
        - psivec32((alpha + Nk) @ fraction_map.T) @ fraction_map
        + psivec32(tau + (Nk @ fraction_map.T)) @ fraction_map
    )

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    phi_matrix = np.outer(exp_Elog_prior, X_div_X_tild) * conditional_likelihood

    return phi_matrix


"""
Gibb's sampling for mixture models
"""


@njit(nogil=True)
def _parallel_categorical_draw(logits, u):
    logits = np.exp(logits - logits.max())
    cdf = np.cumsum(logits)
    z = cdf[-1]
    return np.searchsorted(cdf, u * z)


@njit(nogil=True)
def _gibbs_sample_mixture_step(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    log_conditional_likelihood,  # Ix(D*K)
    weights,  # I
    Nk,
    z,
    temperature,
):
    log_weight = 0.0
    I = log_conditional_likelihood.shape[0]
    sampling_order = np.random.permutation(I)

    for i in sampling_order:
        Nk[z[i]] -= weights[i]
        logits = (
            temperature * log_conditional_likelihood[i]
            + np.log(alpha + Nk)
            - np.log((alpha + Nk) @ fraction_map.T) @ fraction_map
            + np.log(tau + (Nk @ fraction_map.T)) @ fraction_map
        )
        z[i] = categorical_draw(logits)
        Nk[z[i]] += weights[i]
        log_weight += weights[i] * log_conditional_likelihood[i, z[i]]

    return Nk, z, log_weight


@njit(nogil=True)
def _gibbs_sample_mixture_step_implicit(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    log_conditional_likelihood,  # Ix(D*K)
    weights,  # I
    Nk,
    temperature,
):
    """
    This function performs Gibbs sampling for a mixture of MuTopia models,
    but with block sampling. The major advantage of block gibbs sampling is that
    one needs not explicitly store the latent variable assignments (z) for each mutation.

    For samples with large number of mutations, mutations are essientailly independent given the current
    mixture contributions, and so this estimator is not much less efficient than the full Gibbs sampler.
    """
    log_weight = 0.0
    I, _ = log_conditional_likelihood.shape
    new_Nk = np.zeros_like(Nk)
    u = np.random.uniform(0, 1, size=I)

    logits = (
        temperature * log_conditional_likelihood
        + np.log(alpha + Nk)
        - np.log((alpha + Nk) @ fraction_map.T) @ fraction_map
        + np.log(tau + (Nk @ fraction_map.T)) @ fraction_map
    )

    for i in range(I):
        k = _parallel_categorical_draw(logits[i], u[i])
        log_weight += weights[i] * log_conditional_likelihood[i, k]
        new_Nk[k] += weights[i]

    return new_Nk, log_weight


@njit(nogil=True)
def gibbs_sample(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    log_conditional_likelihood,  # Ix(D*K)
    weights,  # I
    Nk,
    warmup=100,
    steps=1000,
):

    out = np.zeros((Nk.shape[0], steps), dtype=Nk.dtype)
    lls = np.zeros(steps + warmup, dtype=Nk.dtype)

    for t in range(steps + warmup):
        # Perform a Gibbs sampling step
        Nk, ll = _gibbs_sample_mixture_step_implicit(
            alpha,
            tau,
            fraction_map,
            log_conditional_likelihood,
            weights,
            Nk,
            min(1, t / warmup),  # warm up for 100 steps
        )

        lls[t] = ll

        if t >= warmup:
            out[:, t - warmup] = Nk

    return out, lls


def iterative_update_gibbs(
    alpha,  # D*K
    tau,  # D
    fraction_map,  # Dx(D*K)
    conditional_likelihood,
    weights,
    iters,
    tol,
    Nk,
):
    args = (
        alpha,
        tau,
        fraction_map,
        np.log(conditional_likelihood),
        weights,
    )

    # renormalize Nk to match the weights
    Nk = Nk / np.sum(Nk) * np.sum(weights)

    Nks, _ = gibbs_sample(
        *args,
        Nk,
        warmup=100,
        steps=iters,
    )

    return Nks[:, -1]


class MixtureModel(LocalsModel):

    def __init__(
        self,
        GT: MixtureInterface,
        datasets,
        prior_alpha=1.0,
        prior_tau=1.0,
        estep_iterations=1000,
        difference_tol=5e-5,
        dtype=np.float32,
        *,
        random_state,
        n_components,
        **kw,
    ):
        self.estep_iterations = estep_iterations
        self.difference_tol = difference_tol
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype
        self.GT = GT

        self.alpha = {
            basename(name): np.ones(n_components, dtype=dtype) * prior_alpha
            for name, _ in self.GT.expand_datasets(*datasets)
        }

        self.tau = {
            self.GT.get_name(dataset): np.ones(
                len(self.GT.list_sources(dataset)), dtype=dtype
            )
            * prior_tau
            for dataset in datasets
        }

    def prepare_corpusstate(self, dataset):

        n_observations = np.array(
            [sample.X.sum().data.item() for _, sample in self.GT.iter_samples(dataset)]
        )

        return dict(
            topic_compositions=DataArray(
                self.init_locals(len(n_observations)),
                dims=("component", "sample"),
            ),
            n_observations=DataArray(
                n_observations,
                dims=("sample",),
            ),
        )

    def _get_mixture_kw(self, dataset):

        alpha = np.concatenate(
            [self.alpha[basename(name)] for name, _ in self.GT.expand_datasets(dataset)]
        )
        alpha = self.to_contig(alpha)

        tau = self.to_contig(self.tau[self.GT.get_name(dataset)])

        n_sources = len(self.GT.list_sources(dataset))

        fraction_map = (
            idx_array_to_design(
                np.repeat(np.arange(n_sources), self.n_components), n_sources
            )
            .todense()
            .T
        )
        fraction_map = self.to_contig(fraction_map)

        return {
            "alpha": alpha,
            "tau": tau,
            "fraction_map": fraction_map,
        }

    def Mstep(self, datasets, learning_rate=1):

        Nks_by_source = {}
        for name, dataset in self.GT.expand_datasets(*datasets):

            Nks = (
                self.GT.fetch_val(dataset, "topic_compositions")
                .transpose("sample", ...)
                .data
            )

            source_name = basename(name)

            if source_name not in Nks_by_source:
                Nks_by_source[source_name] = Nks
            else:
                Nks_by_source[source_name] = np.concatenate(
                    (Nks_by_source[source_name], Nks), axis=0
                )

        for source_name, alpha0 in self.alpha.items():

            self.alpha[source_name] = _svi_update_fn(
                alpha0, update_alpha(alpha0, Nks_by_source[source_name]), learning_rate
            )

        """for dataset in datasets:

            name = self.GT.get_name(dataset)
            Nds = (
                self.GT.fetch_val(dataset, "topic_compositions")
                .sum(dim="component")
                .transpose("sample", "source")
                .data
            )

            tau = self.tau[name]
            self.tau[name] = _svi_update_fn(
                tau,
                update_alpha(tau, Nds),
                learning_rate
            )"""

        return self
