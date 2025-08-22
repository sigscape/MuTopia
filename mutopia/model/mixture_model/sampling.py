from .base import *

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
