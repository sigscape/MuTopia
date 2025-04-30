from functools import partial, reduce
import numpy as np
from xarray import DataArray
from ._glm_compiled import (
    make_optimizer,
    setup_mixed_solver,
    right_intercept_solver,
    partial_ls_solver,
)
from ._fast_eln import get_eln_solver
from .. import corpus_state as CS
from .base import (
    get_reg_params,
    get_poisson_targets_weights,
    _svi_update_fn,
    RateModel,
    SparseDataBase,
    DenseDataBase,
)
from ._strand_transformer import MesoscaleEncoder, DesignMatrixHelper
from ...gtensor import dims_except_for


def _reduce_locus_exp_offsets(
    context_effects,
    exposures,
    idx_selector,
    conditional_rates,
):
    pass


class StrandedContextModel(RateModel, SparseDataBase, DenseDataBase):
    """
    The dimensions of context models are hard-coded because contexts are always
    part of the genome, and so are fixed. Additionally, strandedness
    imparts spectial behavior. This is hard-coded as well.
    """

    def __init__(
        self,
        corpuses,
        context_encoder,
        max_iter=100,
        tol=1e-3,
        reg: float = 0.0001,
        kmer_reg=0.005,
        init_variance: float = 0.1,
        conditioning_alpha: float = 1e-9,
        dtype=float,
        init_components=[],
        fix_components=[],
        context_conditioning=1e-5,
        *,
        n_components: int,
        random_state,
    ):
        super().__init__(*corpuses)

        self.n_components = n_components
        self.n_fixed_components = len(fix_components)
        self.context_dim = CS.get_dims(corpuses[0])["context"]
        self.context_names = list(corpuses[0].coords["context"].data)
        self.dtype = dtype

        corpus = corpuses[0]  # just grab the first one to use for initialization
        self._context_distribution = (
            CS.get_regions(corpus)
            .context_frequencies.sum(
                dim=dims_except_for(
                    CS.get_regions(corpus).context_frequencies.dims, "context"
                )
            )
            .data
            + 1
        )
        self._context_distribution /= self._context_distribution.sum()

        self.context_transformer = context_encoder.fit(self.context_names)
        self.transformer = MesoscaleEncoder().fit(corpuses)

        self.encoding_matrix_ = DesignMatrixHelper.compose_encoding_matrix(
            self.context_transformer.encoding_matrix,
            self.transformer.encoding_matrix,
            shared_effects=True,
        ).astype(dtype)

        X = DesignMatrixHelper.one_pad_right(self.encoding_matrix_.copy()).astype(dtype)

        self._coefs = self._init_params(
            random_state, n_components, init_variance, X.shape[1], dtype
        )

        self.is_intercept_ = np.array(
            [True] * self.context_transformer.n_coefs
            + [False]
            * self.transformer.n_coefs
            * (self.context_transformer.n_states + 1)
        )

        if len(fix_components) > 0 or len(init_components) > 0:
            self.init_from_signatures(
                corpuses[0]
                .modality()
                .load_components(*fix_components, *init_components)
            )

        optim_kw = dict(
            is_intercept=self.is_intercept_,
            tol=tol,
            max_iter=max_iter,
        )

        ridge_solver = partial(
            partial_ls_solver,
            alpha=conditioning_alpha,
        )  # f(X) -> f(z, w, beta) -> beta

        eln_solver = partial(
            get_eln_solver,
            **get_reg_params(reg, context_conditioning),
            tol=tol,
            random_state=random_state,
            max_iter=100,
        )  # f(X) -> f(z, w, beta) -> beta

        self.model = self._make_optimizer(
            X,
            ridge_solver,
            eln_solver,
            **optim_kw,
        )

        passthrough_solver = lambda X: lambda z, w, beta: beta

        self.fixed_model = self._make_optimizer(
            X,
            passthrough_solver,
            eln_solver,
            **optim_kw,
        )  # f(X) -> f(z, w, beta) -> beta

    def _make_optimizer(
        self,
        X,
        unreg_solver,
        reg_solver,
        *,
        is_intercept,
        tol,
        max_iter,
    ):
        split_solver = partial(
            setup_mixed_solver,
            is_regularized=~is_intercept,
            unreg_solver=unreg_solver,  # f(X) -> f(z, w, beta) -> beta
            reg_solver=reg_solver,  # f(X) -> f(z, w, beta) -> beta
        )

        solver = partial(
            right_intercept_solver,
            solver=split_solver,
        )

        return make_optimizer(
            X,
            solver,
            tol=tol,
            max_iter=max_iter,
        )

    @property
    def requires_normalization(self):
        return True

    @property
    def requires_dims(self):
        return ("configuration", "context", "locus")

    def post_fit(self, corpuses):

        corpus = corpuses[0]

        locus_effects = (
            CS.fetch_val(corpus, "log_locus_distribution").transpose("locus", ...).data
        )

        freqs = CS.get_regions(corpus).context_frequencies

        if "context" in freqs.dims:
            contexts = (
                freqs.sum(dim=dims_except_for(freqs.dims, "context", "locus"))
                .transpose("context", "locus")
                .data
            )

            # CxL @ LxK => CxK => KxC
            self._comp_context_distribution = (contexts @ np.exp(locus_effects)).T
            self._comp_context_distribution /= self._comp_context_distribution.sum(
                axis=1, keepdims=True
            )
        else:
            self._comp_context_distribution = np.expand_dims(
                self.context_distribution_, axis=0
            )

    def prepare_to_save(self):
        if hasattr(self, "model"):
            del self.model

        if hasattr(self, "fixed_model"):
            del self.fixed_model

    ##
    # Satisfaction of SparseDataBase interface
    ##
    def predict_sparse(self, corpus, *, configuration, context, locus, **idx_dict):

        mesoscale_state = CS.fetch_val(corpus, "mesoscale_idx").data[
            configuration, locus
        ]

        return CS.fetch_val(corpus, "log_context_distribution").data[
            :, context, mesoscale_state
        ]

    def reduce_sparse_sstats(
        self,
        sstats,
        corpus,
        *,
        weighted_posterior,
        locus,
        context,
        configuration,
        **kw,
    ):
        mesoscale_states = CS.fetch_val(corpus, "mesoscale_idx").data[
            configuration, locus
        ]

        # Use numpy advanced indexing to update context_sstats
        np.add.at(sstats, (slice(None), context, mesoscale_states), weighted_posterior)

        return sstats.astype(self.dtype, copy=False)

    ##
    # End of SparseDataBase interface
    ##

    def reduce_dense_sstats(self, sstats, corpus, *, weighted_posterior, **kw):

        to_dim = ("component", *self.requires_dims)

        if not to_dim[1] == "configuration":
            raise ValueError("First dimension must be configuration!")
        # 2 x C x L
        weights = (
            weighted_posterior.sum(
                dim=dims_except_for(weighted_posterior.dims, *to_dim)
            )
            .transpose(*to_dim)
            .data
        )

        # 2 x L
        (plus_idx, minus_idx) = CS.fetch_val(corpus, "mesoscale_idx").data

        np.add.at(sstats, (slice(None), slice(None), plus_idx), weights[:, 0])
        np.add.at(sstats, (slice(None), slice(None), minus_idx), weights[:, 1])

        return sstats.astype(self.dtype, copy=False)

    def init_from_signatures(self, signatures):

        k = signatures.shape[0]

        if k > self.n_components:
            raise ValueError(
                "You are trying to initialize with too many signatures! Increase `num_components/k`, then try again."
            )

        c = self.context_transformer.n_coefs

        context_effects = (
            signatures.sum(dim=dims_except_for(signatures.dims, "context", "component"))
            .transpose("component", "context")
            .data
        )

        context_effects /= context_effects.sum(axis=1, keepdims=True)
        renormalized = 100 * (context_effects + 1e-5)  # /self._context_distribution

        self._coefs[:k, :c] = np.log(renormalized)

        return self

    def _init_params(self, random_state, n_components, init_variance, n_coefs, dtype):
        return random_state.normal(
            0.0,
            init_variance,
            (n_components, n_coefs),
        ).astype(dtype, copy=False)

    @property
    def context_distribution_(self):
        return self._context_distribution

    def prepare_corpusstate(self, corpus):
        return dict(
            mesoscale_idx=DataArray(
                np.array(
                    [
                        self.transformer.encode(corpus),
                        self.transformer.encode(corpus, invert=True),
                    ]
                ),
                dims=(
                    "configuration",
                    "locus",
                ),
            ),
            plus_mesoscale_design=DataArray(
                self.transformer.transform(corpus),
                dims=("locus", "mesoscale_state"),
            ),
            minus_mesoscale_design=DataArray(
                self.transformer.transform(corpus, invert=True),
                dims=("locus", "mesoscale_state"),
            ),
            log_context_distribution=DataArray(
                self._get_log_context_distribution(corpus),
                dims=("component", "context", "mesoscale_state"),
            ),
        )

    def _get_log_context_distribution(self, corpus_state):
        return np.array([self._format_component(k) for k in range(self.n_components)])

    def update_corpusstate(self, corpus, **kwargs):
        CS.fetch_val(corpus, "log_context_distribution").data[:] = (
            self._get_log_context_distribution(corpus)
        )

    def _get_sstats_dim(self):
        return (self.n_components, self.context_dim, self.transformer.n_states_)

    def spawn_sstats(self, corpus_state):
        return np.zeros(self._get_sstats_dim(), self._coefs.dtype)

    def predict(self, k, corpus_state):

        conditional_mutation_rates = self._format_component(k)
        # 2 x S x L
        (plus_idx, minus_idx) = CS.fetch_val(corpus_state, "mesoscale_idx").data

        # 2 x C x L => 2 x C x L
        context_effects = np.array(
            [
                conditional_mutation_rates[:, plus_idx],  # K x C x L
                conditional_mutation_rates[:, minus_idx],
            ]
        )

        return DataArray(
            context_effects,
            dims=("configuration", "context", "locus"),
        )

    @staticmethod
    def _run_regression(
        y,
        sample_weight,
        learning_rate=1.0,
        *,
        update_vec,
        model,
    ):
        update_vec[:] = _svi_update_fn(
            update_vec, model(y, sample_weight)(update_vec), learning_rate=learning_rate
        )

    @staticmethod
    def get_exp_offset(offsets, corpus):

        def aggregate_by_design(arr, design_matrix):
            #       SxL @ (LxC) => (SxC) => (CxS) => (C*S)
            return (design_matrix @ arr.T).T.ravel()

        def get_design(state, key):
            return CS.fetch_val(state, key).data.to_scipy_sparse().tocsc().T

        exp_offsets = (
            np.exp(offsets).transpose("configuration", "context", "locus").data
        )

        return aggregate_by_design(
            exp_offsets[0], get_design(corpus, "plus_mesoscale_design")
        ) + aggregate_by_design(
            exp_offsets[1], get_design(corpus, "minus_mesoscale_design")
        )

    def partial_fit(
        self,
        k,
        sstats,
        exp_offsets,
        corpuses,
        learning_rate=1.0,
    ):

        corpus_names = [CS.get_name(state) for state in corpuses]

        eta = reduce(
            lambda x, y: x + y, (exp_offsets[n][k].ravel() for n in corpus_names)
        )
        target = reduce(
            lambda x, y: x + y, (sstats[n][k].ravel() for n in corpus_names)
        )

        yield partial(
            self._run_regression,
            *get_poisson_targets_weights(target, eta),
            model=self.fixed_model if k < self.n_fixed_components else self.model,
            update_vec=self._coefs[k],
            learning_rate=learning_rate,
        )

    @property
    def coefs_(self):
        return self._coefs[:, :-1]  # self.n_corpuses]

    def _calc_lambda(self, k, design_matrix):
        return (self.coefs_[k] @ design_matrix.T).reshape((self.context_dim, -1))

    def _format_component(self, k):
        return self._calc_lambda(k, self.encoding_matrix_)

    def _get_mesoscale_feature_name(self, raw_name):
        return raw_name.replace(":-1", ":A/G-centered").replace(":1", ":C/T-centered")

    def get_mesoscale_feature_names(self):
        return list(
            map(
                self._get_mesoscale_feature_name,
                self.transformer.get_feature_names_out(),
            )
        )

    def format_signature(self, k, normalization="global"):

        if not normalization in ("global", "weighted", "none"):
            raise ValueError(
                'Normalization must be one of "global", "weighted", or "none"'
            )

        encoding_matrix = DesignMatrixHelper.compose_encoding_matrix(
            self.context_transformer.encoding_matrix,
            self.transformer.independent_effects_encoding(),
        )

        # C x S
        signature = self._calc_lambda(k, encoding_matrix)

        if normalization == "weighted":
            signature += np.log(self._comp_context_distribution[k, :])[:, None]

        if normalization == "global":
            signature += np.log(self._context_distribution)[:, None]

        return DataArray(
            signature,
            dims=("context", "mesoscale_state"),
            coords={
                "context": self.context_names,
                "mesoscale_state": ["Baseline"] + self.get_mesoscale_feature_names(),
            },
        )

    def get_baseline_summary(self, k):
        return self.format_signature(k, normalization="none").sel(
            mesoscale_state="Baseline"
        )

    def get_interaction_summary(self, k):

        c = self.context_transformer.n_states + 1
        r = self.transformer.n_coefs

        return DataArray(
            data=self.coefs_[k][-r * c :].reshape((c, r)).T,
            dims=("feature", "context"),
            coords={
                "feature": self.get_mesoscale_feature_names(),
                "context": ["Shared effect"] + self.context_names,
            },
        )


class UnstrandedContextModel(StrandedContextModel, SparseDataBase):

    @property
    def requires_dims(self):
        return ("context", "locus")

    @staticmethod
    def predict_sparse(corpus, *, context, locus, **idx_dict):

        mesoscale_state = CS.fetch_val(corpus, "mesoscale_idx").data[locus]

        return CS.fetch_val(corpus, "log_context_distribution").data[
            :, context, mesoscale_state
        ]

    @staticmethod
    def reduce_sparse_sstats(
        sstats,
        corpus,
        *,
        weighted_posterior,
        locus,
        context,
        **kw,
    ):
        mesoscale_states = CS.fetch_val(corpus, "mesoscale_idx").data[locus]

        # Use numpy advanced indexing to update context_sstats
        np.add.at(sstats, (slice(None), context, mesoscale_states), weighted_posterior)

        return sstats

    def reduce_dense_sstats(self, sstats, corpus, *, weighted_posterior, **kw):

        to_dim = ("component", *self.requires_dims)

        # K x Cons x L
        weights = (
            weighted_posterior.sum(
                dim=dims_except_for(weighted_posterior.dims, *to_dim)
            )
            .transpose(*to_dim)
            .data
        )

        # 2 x L
        idx = CS.fetch_val(corpus, "mesoscale_idx").data

        # sstats : K x C x S x M
        np.add.at(sstats, (slice(None), slice(None), idx), weights)

        return sstats

    def prepare_corpusstate(self, corpus):
        return dict(
            mesoscale_idx=DataArray(
                self.transformer.encode(corpus),
                dims=("locus",),
            ),
            mesoscale_design=DataArray(
                self.transformer.transform(corpus),
                dims=("locus", "mesoscale_state"),
            ),
            log_context_distribution=DataArray(
                self._get_log_context_distribution(corpus),
                dims=("component", "context", "mesoscale_state"),
            ),
        )

    def predict(self, k, corpus_state):

        conditional_mutation_rates = self._format_component(k)
        # S x L
        idx = CS.fetch_val(corpus_state, "mesoscale_idx").data

        return DataArray(
            conditional_mutation_rates[:, idx],
            dims=("context", "locus"),
        )

    @staticmethod
    def get_exp_offset(offsets, corpus):

        def aggregate_by_design(arr, design_matrix):
            #       SxL @ (LxC) => (SxC) => (CxS) => (C*S)
            return (design_matrix @ arr.T).T.ravel()

        def get_design(state, key):
            return CS.fetch_val(state, key).data.to_scipy_sparse().tocsc().T

        exp_offsets = np.exp(offsets).transpose("context", "locus").data

        return aggregate_by_design(exp_offsets, get_design(corpus, "mesoscale_design"))
