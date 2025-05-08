from numba import njit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
from ..model.latent_var_models.base import *


@njit("float32[:](float32[:])", nogil=True)
def psivec32(x):
    return psivec(x).astype(np.float32)

"""
Mixture model inference functions
"""
@njit(nogil=True)
def _update_step(
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
        np.float32(0.0)
    )
    
    Nk = exp_Elog_prior * (conditional_likelihood @ X_div_X_tild)

    return Nk


#@flatten_tensor_for_update
@njit(nogil=True)
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


#@reshape_output
#@flatten_tensor_for_update
@njit(nogil=True)
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

    #X_tild = exp_Elog_gamma @ conditional_likelihood
    X_div_X_tild = weights / (exp_Elog_prior @ conditional_likelihood)

    phi_matrix = np.outer(exp_Elog_prior, X_div_X_tild) * conditional_likelihood

    return phi_matrix


'''@flatten_tensor_for_update
@flatten_last_arg
@njit(
    "double(double[::1], double[:,::1], double[::1], double[::1], double[:,::1])",
    nogil=True,
)
def bound(
    alpha,
    conditional_likelihood,
    weights,
    gamma,
    weighted_posterior,
):

    phi = weighted_posterior / weights[None, :]
    entropy_sstats = -np.sum(weighted_posterior * np.where(phi > 0, np.log(phi), 0.0))
    entropy_sstats += dirichlet_bound(alpha, gamma)

    flattened_logweight = log_dirichlet_expectation(gamma)[:, None] + np.log(
        conditional_likelihood
    )

    return (
        np.sum(weighted_posterior * np.nan_to_num(flattened_logweight, nan=0.0))
        + entropy_sstats
    )'''


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
            min(1, t / warmup)  # warm up for 100 steps
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
    Nk = Nk/np.sum(Nk) * np.sum(weights)

    Nks, _ = gibbs_sample(
        *args,
        Nk,
        warmup=100,
        steps=iters,
    )

    return Nks[:,-1]
