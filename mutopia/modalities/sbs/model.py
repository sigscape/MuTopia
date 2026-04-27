from numba import njit
import numpy as np
import xarray as xr
from functools import partial
from mutopia.utils import logger
from mutopia.model.model import (
    StrandedContextModel,
    GBTThetaModel,
    LinearThetaModel,
    FactorModel,
    DiagonalEncoder,
)
from mutopia.model import TopographyModel


contig_f32 = partial(np.ascontiguousarray, dtype=np.float32)


def _get_args(k, corpus):

    return (
        (
            corpus.sections["Regions"]
            .context_frequencies.transpose("locus", "context", "configuration")
            .data.reshape(-1, 192)
        ),
        corpus.sections["Regions"].exposures.data,
        contig_f32(
            np.exp(corpus.sections["State"].log_locus_distribution)
            .isel(component=k)
            .data
        ),
        (
            np.exp(corpus.sections["State"].log_context_distribution)
            .isel(component=k)
            .transpose(..., "context")
            .data
        ),
        corpus.sections["State"].mesoscale_idx.data.T,
    )


@njit(
    "Tuple((float32, float32[:,:],float32[:]))(float32[:,::1], float32[::1], float32[::1], float32[:,:], int64[:,::1])",
    nogil=True,
)
def _fast_exp_offsets(
    context_freqs,  # (L x C*D)
    exposures,  # L
    locus_effects,  # L
    context_effects,  # S x C
    idx_selector,  # L,D
):

    D = 2
    L, CD = context_freqs.shape
    C = CD // D
    S, _ = context_effects.shape

    assert exposures.shape == (L,)
    assert locus_effects.shape == (L,)
    assert idx_selector.shape == (L, D)
    assert context_effects.shape == (S, C)

    context_offsets = np.zeros((S, C), dtype=context_freqs.dtype)
    locus_offsets = np.zeros(L, dtype=context_freqs.dtype)
    normalizer = 0

    for l, (s_d0, s_d1) in enumerate(idx_selector):

        # (D, C)
        o = exposures[l] * context_freqs[l, :].reshape(C, D).T

        context_offsets[s_d0, :] += o[0] * locus_effects[l]
        context_offsets[s_d1, :] += o[1] * locus_effects[l]

        locus_offsets[l] += (context_effects[s_d0, :] * o[0]).sum() + (
            context_effects[s_d1, :] * o[1]
        ).sum()

        normalizer += locus_offsets[l] * locus_effects[l]

    return (
        -np.log(normalizer),
        context_offsets.T,
        locus_offsets,
    )


def _get_exp_offsets_k_c(factor_model, k, corpus):

    # STABILITY: max-subtract in log-space before exp to avoid inf*0 -> NaN in the
    # numba kernel. The Poisson M-step absorbs uniform offset rescaling via
    # get_poisson_targets_weights, so we only have to undo the shift on the
    # scalar normalizer (in log-space, where it cannot overflow).
    state = corpus.sections["State"]
    log_locus = np.ascontiguousarray(
        state.log_locus_distribution.isel(component=k).data, dtype=np.float32
    )
    log_context = np.ascontiguousarray(
        state.log_context_distribution.isel(component=k)
        .transpose(..., "context")
        .data,
        dtype=np.float32,
    )

    finite_locus = log_locus[np.isfinite(log_locus)]
    finite_context = log_context[np.isfinite(log_context)]
    Lmax = np.float32(finite_locus.max()) if finite_locus.size else np.float32(0.0)
    Cmax = np.float32(finite_context.max()) if finite_context.size else np.float32(0.0)

    locus_effects = contig_f32(np.exp(log_locus - Lmax))
    context_effects = np.exp(log_context - Cmax).astype(np.float32, copy=False)

    context_freqs = (
        corpus.sections["Regions"]
        .context_frequencies.transpose("locus", "context", "configuration")
        .data.reshape(-1, 192)
    )
    exposures = corpus.sections["Regions"].exposures.data
    idx_selector = state.mesoscale_idx.data.T

    (normalizer, context_offsets, locus_offsets) = _fast_exp_offsets(
        context_freqs, exposures, locus_effects, context_effects, idx_selector,
    )

    # STABILITY: restore the max-subtraction purely in log-space.
    normalizer = float(normalizer) - float(Lmax) - float(Cmax)

    # STABILITY: fall back to the previous normalizer instead of poisoning state
    # if the slice was degenerate (e.g. empty normalizer, stale NaNs in inputs).
    if not (
        np.isfinite(normalizer)
        and np.all(np.isfinite(locus_offsets))
        and np.all(np.isfinite(context_offsets))
    ):
        logger.warning(
            f"Non-finite exp-offsets for component {k} on corpus "
            f"{factor_model.GT.get_name(corpus)}; falling back to previous normalizer."
        )
        normalizer = float(factor_model.get_normalizers(corpus)[k])
        locus_offsets = np.nan_to_num(
            locus_offsets, nan=0.0, posinf=0.0, neginf=0.0
        )
        context_offsets = np.nan_to_num(
            context_offsets, nan=0.0, posinf=0.0, neginf=0.0
        )

    return (
        normalizer,
        {"context_model": context_offsets, "theta_model": locus_offsets},
    )


@njit(
    "float32[:,:](float32[:,::1], float32[::1], float32[::1], float32[:,:], int64[:,::1], float32, bool)",
    nogil=True,
)
def _fast_component_predict(
    context_freqs,  # (L x C*D)
    exposures,  # L
    locus_effects,  # L
    context_effects,  # S x C
    idx_selector,  # L,D
    normalizer,  # float32
    with_context,
):
    D = 2
    L, CD = context_freqs.shape
    C = CD // D
    S, _ = context_effects.shape

    assert exposures.shape == (L,)
    assert locus_effects.shape == (L,)
    assert idx_selector.shape == (L, D)
    assert context_effects.shape == (S, C)

    out = np.zeros_like(context_freqs)
    ones = np.ones((D, C), dtype=context_freqs.dtype)

    for l, s in enumerate(idx_selector):

        # (DxC)
        exp_offsets = (
            exposures[l] * context_freqs[l, :].reshape(C, D).T if with_context else ones
        )

        # (DxC) * (DxC) * (1,) ==> (DxC).T ==> ravel(CxD) ==> C*D
        out[l, :] = (context_effects[s, :] * exp_offsets * locus_effects[l]).T.ravel()

    return np.log(out) + normalizer


def _predict(factor_model, k, corpus, with_context=True):

    out = _fast_component_predict(
        *_get_args(k, corpus),
        np.float32(factor_model.get_normalizers(corpus)[k]),
        with_context=with_context,
    )

    return xr.DataArray(
        out.reshape(-1, 96, 2).T,
        dims=("configuration", "context", "locus"),
    )


class SBSModel(TopographyModel):

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
        locus_model_type,
        tree_learning_rate,
        max_depth,
        max_trees_per_iter,
        max_leaf_nodes,
        min_samples_leaf,
        max_features,
        n_iter_no_change,
        use_groups,
        add_corpus_intercepts,
        convolution_width,
        l2_regularization,
        init_variance_theta,
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

        theta_model = (GBTThetaModel if not locus_model_type=="linear" else LinearThetaModel)(
            GT.to_datasets(*train_corpuses),
            init_variance=init_variance_theta,
            n_components=num_components,
            tree_learning_rate=tree_learning_rate,
            max_depth=max_depth,
            max_trees_per_iter=max_trees_per_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_iter_no_change=n_iter_no_change,
            use_groups=use_groups,
            random_state=random_state,
            add_corpus_intercepts=add_corpus_intercepts,
            convolution_width=convolution_width,
            l2_regularization=l2_regularization,
        )

        factor_model = FactorModel(
            GT,
            train_corpuses,
            context_model=context_model,
            theta_model=theta_model,
            predict_fn=_predict,
            offsets_fn=_get_exp_offsets_k_c,
        )

        return factor_model
