import numpy as np
from numba import njit
from mutopia.utils import logger
from mutopia.model.model import (
    StrandedContextModel,
    FactorModel,
    DiagonalEncoder,
)
from .model import SBSModel

def _get_args(k, corpus):
    return (
        (
            corpus.sections["Regions"]
            .context_frequencies.transpose("locus", "context", "configuration")
            .data.reshape(-1, 192)
        ),
        corpus.sections["Regions"].exposures.data,
        (
            np.exp(corpus.sections["State"].log_context_distribution)
            .isel(component=k)
            .transpose(..., "context")
            .data
        ),
        corpus.sections["State"].mesoscale_idx.data.T,
    )


@njit(
    "Tuple((float32, float32[:,:],float32[:]))(float32[:,::1], float32[::1], float32[:,:], int64[:,::1])",
    nogil=True,
)
def _context_only_fast_exp_offsets(
    context_freqs,  # (L x C*D)
    exposures,  # L
    context_effects,  # S x C
    idx_selector,  # L,D
):

    D = 2
    L, CD = context_freqs.shape
    C = CD // D
    S, _ = context_effects.shape

    assert exposures.shape == (L,)
    assert idx_selector.shape == (L, D)
    assert context_effects.shape == (S, C)

    context_offsets = np.zeros((S, C), dtype=context_freqs.dtype)
    locus_offsets = np.zeros(L, dtype=context_freqs.dtype)
    normalizer = 0

    for l, (s_d0, s_d1) in enumerate(idx_selector):

        # (D, C)
        o = exposures[l] * context_freqs[l, :].reshape(C, D).T

        context_offsets[s_d0, :] += o[0]
        context_offsets[s_d1, :] += o[1]

        locus_offsets[l] += (context_effects[s_d0, :] * o[0]).sum() + (
            context_effects[s_d1, :] * o[1]
        ).sum()

        normalizer += locus_offsets[l]

    return (
        -np.log(normalizer),
        context_offsets.T,
        locus_offsets,
    )


def _get_exp_offsets_k_c(factor_model, k, corpus):

    (normalizer, context_offsets, locus_offsets) = _context_only_fast_exp_offsets(
        *_get_args(k, corpus)
    )

    return (
        normalizer,
        {"context_model": context_offsets, "theta_model": locus_offsets},
    )

class ContextOnlyModel(SBSModel):
    
    def _init_factor_model(
        self,
        train_corpuses,
        random_state,
        GT,  # gtensor interface
        *,
        num_components,
        init_components,
        fix_components,
        # context model
        context_reg,
        context_conditioning,
        conditioning_alpha,
        init_variance_context,
        max_iter,
        # locus model
        **kw,
    ):

        logger.info("Initializing model parameters and transformations...")

        context_model = StrandedContextModel(
            GT.to_datasets(*train_corpuses),
            DiagonalEncoder(),
            n_components=num_components,
            random_state=random_state,
            init_variance=init_variance_context,
            tol=5e-4,
            reg=context_reg,
            context_conditioning=context_conditioning,
            conditioning_alpha=conditioning_alpha,
            init_components=init_components,
            fix_components=fix_components,
            max_iter=max_iter,
        )

        factor_model = FactorModel(
            GT,
            train_corpuses,
            context_model=context_model,
            offsets_fn=_get_exp_offsets_k_c,
        )

        return factor_model
