import inspect
from .base import (
    get_corpus_intercepts,
    get_poisson_targets_weights,
    _svi_update_fn,
    RateModel,
    idx_array_to_design,
    SparseDataBase,
    DenseDataBase,
)
from ...utils import str_wrapped_list
from ...gtensor import dims_except_for
from .. import gtensor_interface as CS
from ._hist_gbt import CustomHistGradientBooster
from ._feature_tranformer import (
    get_feature_transformer,
    get_categorical_feature_idxs,
    get_feature_interaction_group_idxs,
    get_known_categories,
    StratifiedTransformer,
)

from functools import partial
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import clone
from xarray import DataArray
import logging

logger = logging.getLogger(" Mutopia")


class ThetaModel(RateModel, SparseDataBase, DenseDataBase):

    categorical_encoder = None

    def __init__(
        self,
        corpuses,
        add_corpus_intercepts=False,
        convolution_width=1,
        model_kw={},
        dtype=np.float32,
        init_variance=0.05,
        transformers=[],
        *,
        n_components: int,
        random_state,
    ):
        super().__init__(*corpuses)

        self.dtype = dtype
        self.n_components = n_components
        self.corpus_encoder_ = {
            CS.get_name(corpus): i for i, corpus in enumerate(corpuses)
        }

        (self.base_transformer_, feature_names_in) = get_feature_transformer(
            *corpuses,
            additional_transformers=transformers,
            categorical_encoder=self.categorical_encoder,
            add_corpus_intercepts=add_corpus_intercepts and len(corpuses) > 1,
            convolution_width=convolution_width,
        )

        # Fit the transformer to get an example
        # on which to fit the GBT binning.
        corpus = corpuses[0]
        X = self.base_transformer_.fit(corpus, corpus=corpus).transform(
            corpus, corpus=corpus
        )

        self.n_features_ = X.shape[1]
        self.feature_names_out_ = self.base_transformer_.get_feature_names_out(
            feature_names_in
        )

        categorical_features = get_categorical_feature_idxs(self.base_transformer_)
        self.n_categorical_features_ = len(categorical_features)

        known_categories = get_known_categories(
            self.base_transformer_, len(self.feature_names_out_)
        )

        interaction_groups = get_feature_interaction_group_idxs(
            corpus, self.base_transformer_.get_feature_names_out(feature_names_in)
        )

        self.rate_models = [
            self._make_model(
                **model_kw,
                X=X,
                interaction_groups=interaction_groups,
                categorical_features=categorical_features,
                known_categories=known_categories,
                random_state=random_state,
            )
            for _ in range(n_components)
        ]

        self.locus_transformer_ = StratifiedTransformer(self.base_transformer_)

        """
        We want to initialize the locus effects with something instead of just 0s
        (if you initialize with 0s, the model can only use exposure differences to 
        start disentangling the components). Since we expect the locus effects
        to follow feature distributions, let's just initialize them with a simple
        Laplace projection of the feature matrix.
        """
        self.init_projection_ = random_state.normal(
            0.0,
            init_variance,
            (self.n_components, self.n_features_ - self.n_categorical_features_),
        ).astype(self.dtype)

    @property
    def requires_normalization(self):
        return True

    @property
    def requires_dims(self):
        return ("locus",)

    def _init_log_locus_distribution(self, locus_features):

        X = locus_features[:, self.n_categorical_features_ :]

        return DataArray(
            np.asfortranarray(
                (np.nan_to_num(X, nan=-3.0) @ self.init_projection_.T).T,
                dtype=self.dtype,
            ),
            dims=("component", "locus"),
        )

    def prepare_corpusstate(self, corpus):

        X = self.locus_transformer_.transform(corpus, corpus=corpus)

        return dict(
            locus_features=DataArray(
                np.ascontiguousarray(X, dtype=self.dtype),
                dims=("locus", "feature"),
                coords={
                    "feature": self.feature_names_out_,
                },
            ),
            log_locus_distribution=self._init_log_locus_distribution(X),
        )

    def _check_input(self, X):
        return self.rate_models[0]._preprocess_X(X, reset=False)

    """
    Because of the way the GBT model works, we want to cache and store the predictions
    rather than generate them from the model.
    """

    def update_corpusstate(self, corpus, from_scratch=False):

        X = self._check_input(self._fetch_feature_matrix(corpus))

        for k in range(self.n_components):
            CS.fetch_val(corpus, "log_locus_distribution")[k].data[:] = self._predict(
                k, corpus, from_scratch=from_scratch, X=X, check_input=False
            )

    """
    The predict method just returns the cached predictions,
    while _predict generates them from the model.
    """

    def predict(self, k, corpus):
        return CS.fetch_val(corpus, "log_locus_distribution")[k]

    def spawn_sstats(self, corpus):
        return np.zeros((self.n_components, CS.get_dims(corpus)["locus"]), self.dtype)

    def _get_intercept_idx(self, *corpuses):
        return get_corpus_intercepts(
            corpuses,
            self.corpus_encoder_,
            n_repeats=lambda corpus: CS.get_dims(corpus)["locus"],
        )

    def _get_intercept_design(self, *corpuses):
        return (
            idx_array_to_design(
                self._get_intercept_idx(*corpuses),
                len(corpuses),
            )
            .to_scipy_sparse()
            .tocsc()
        )

    def _make_model(self, **kw):
        raise NotImplementedError

    def _fetch_feature_matrix(self, *corpuses):
        return np.vstack(
            [CS.fetch_val(state, "locus_features").data for state in corpuses]
        )

    def _get_targets(self, k, sstats, exp_offsets, corpuses):

        corpus_names = [CS.get_name(corpus) for corpus in corpuses]
        target = np.concatenate([sstats[n][k] for n in corpus_names])
        eta = np.concatenate([exp_offsets[n][k] for n in corpus_names])

        current_prediction = np.concatenate(
            [
                CS.fetch_val(state, "log_locus_distribution")[k].data
                for state in corpuses
            ]
        )

        return (*get_poisson_targets_weights(target, eta), current_prediction)

    def partial_fit(self, k, sstats, exp_offsets, corpuses, learning_rate=1.0):
        raise NotImplementedError

    @staticmethod
    def predict_sparse(corpus, *, locus, **idx_dict):
        return CS.fetch_val(corpus, "log_locus_distribution").data[:, locus]

    @staticmethod
    def reduce_sparse_sstats(
        sstats,
        corpus,
        *,
        weighted_posterior,
        locus,
        **kw,
    ):
        np.add.at(sstats, (slice(None), locus), weighted_posterior)

        return sstats

    def reduce_dense_sstats(self, sstats, corpus, *, weighted_posterior, **kw):

        to_dim = ("component", *self.requires_dims)

        weights = (
            weighted_posterior.sum(
                dim=dims_except_for(weighted_posterior.dims, *to_dim)
            )
            .transpose(*to_dim)
            .data
        )

        sstats += weights
        return sstats

    def format_signature(self, k, normalization="none"):
        return 0.0


class LinearThetaModel(ThetaModel):

    categorical_encoder = (OneHotEncoder(sparse_output=False, drop="first"),)

    def __init__(self, corpuses, l2_regularization=0.1, **kw):
        super().__init__(
            corpuses, model_kw={"l2_regularization": l2_regularization}, **kw
        )

    def _make_model(self, *, l2_regularization, **kw):
        return PoissonRegressor(
            alpha=l2_regularization,
            solver="lbfgs",
            warm_start=True,
            fit_intercept=False,
        )

    """def _make_design_matrix(self, *corpuses):
        
        design_matrix = self._get_design_matrix(*corpuses)
        X = self._fetch_feature_matrix(*corpuses)

        X = np.hstack([np.nan_to_num(X, nan=0), design_matrix.toarray()])

        return X"""

    def _predict(self, k, corpus, from_scratch=False):

        X = self._make_design_matrix(corpus)

        return np.log(self.rate_models[k].predict(X))

    def partial_fit(
        self,
        sstats,
        k,
        corpuses,
        log_mutation_rates,
        learning_rate=1.0,
    ):

        X = self._make_design_matrix(*corpuses)

        (y, sample_weights, lograte_prediction) = self._get_targets(
            k, sstats, corpuses, log_mutation_rates
        )

        # store the current model state (ignore the intercept fits)
        try:
            old_coef = self.rate_models[k].coef_.copy()
        except AttributeError:
            old_coef = np.zeros(X.shape[1])

        # update the model with the new suffstats
        self.rate_models[k].fit(
            X,
            y,
            sample_weight=sample_weights,
        )

        # merge the new model state with the old
        self.rate_models[k].coef_ = _svi_update_fn(
            old_coef, self.rate_models[k].coef_, learning_rate
        )

        return self


class GBTThetaModel(ThetaModel):

    categorical_encoder = partial(
        OrdinalEncoder,
        max_categories=254,
        dtype=np.float32,
    )

    @classmethod
    def list_params(cls):
        return list(
            set(super().list_params() + inspect.getfullargspec(cls.__init__).args[1:])
        )

    def __init__(
        self,
        corpuses,
        tree_learning_rate=0.1,
        max_depth=5,
        max_trees_per_iter=25,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        max_features=0.5,
        n_iter_no_change=1,
        use_groups=True,
        l2_regularization=1.0,
        **kw,
    ):
        super().__init__(
            corpuses,
            model_kw={
                "tree_learning_rate": tree_learning_rate,
                "max_depth": max_depth,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "n_iter_no_change": n_iter_no_change,
                "use_groups": use_groups,
                "l2_regularization": l2_regularization,
            },
            **kw,
        )
        self.max_trees_per_iter = max_trees_per_iter
        self.predict_from = np.zeros(self.n_components, dtype=int)

    def _make_model(
        self,
        *,
        X,
        interaction_groups,
        tree_learning_rate,
        use_groups,
        random_state,
        known_categories,
        **tree_kw,
    ):
        model = CustomHistGradientBooster(
            loss="poisson",
            scoring="loss",
            warm_start=True,
            early_stopping=True,
            validation_fraction=0.2,
            interaction_cst=interaction_groups if use_groups else None,
            verbose=False,
            learning_rate=tree_learning_rate,
            random_state=1776,
            **tree_kw,
        )

        model.fit_binning(X, known_categories)

        return model

    def partial_fit(self, k, sstats, exp_offsets, corpuses, learning_rate=1.0):

        intercept_idx = self._get_intercept_idx(*corpuses)
        X = self._fetch_feature_matrix(*corpuses)

        rate_model = self.rate_models[k]

        (y, sample_weights, lograte_prediction) = self._get_targets(
            k, sstats, exp_offsets, corpuses
        )

        try:
            n_fit_trees = rate_model.n_iter_
        except AttributeError:
            n_fit_trees = 0

        self.predict_from[k] = n_fit_trees

        rate_model = rate_model.set_params(
            max_iter=n_fit_trees + self.max_trees_per_iter
        )

        yield partial(
            rate_model.fit,
            X,
            y,
            sample_weight=sample_weights,
            intercept_idx=intercept_idx,
            raw_predictions=lograte_prediction[:, None],
            svi_shrinkage=learning_rate,
        )

    def _predict(self, k, corpus, from_scratch=False, X=None, check_input=True):

        if X is None:
            X = self._fetch_feature_matrix(corpus)

        ##
        # TODO:
        # Make this "from_scratch" more robust, like define a private method
        # for initializing the mr estimates.
        ##
        if from_scratch:
            init_prediction = self._init_log_locus_distribution(X)

            theta = np.log(self.rate_models[k].predict(X)) + init_prediction[k].data
        else:
            theta = (
                self.rate_models[k]
                ._raw_predict_from(
                    X,
                    CS.fetch_val(corpus, "log_locus_distribution")[k].data[:, None],
                    from_iteration=self.predict_from[k],
                    check_input=check_input,
                )
                .ravel()
            )

        return theta
