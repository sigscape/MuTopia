from timeit import default_timer as time

from sklearn.base import check_is_fitted
import numpy as np
from sklearn._loss.loss import BaseLoss
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import (
    _check_monotonic_cst,
    _check_sample_weight,
    check_consistent_length,
    _check_y,
)
from sklearn.ensemble._hist_gradient_boosting._gradient_boosting import (
    _update_raw_predictions,
)
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_DTYPE
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    BaseHistGradientBoosting,
    _update_leaves_values,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from numba import njit
import warnings


class ShrunkTreePredictor:

    def __init__(self, tree_predictor, shrinkage):
        self.tree_predictor = tree_predictor
        self.shrinkage = shrinkage

    def predict(self, *args, **kwargs):
        return self.shrinkage * self.tree_predictor.predict(*args, **kwargs)

    def predict_binned(self, *args, **kwargs):
        return self.shrinkage * self.tree_predictor.predict_binned(*args, **kwargs)

    def get_n_leaf_nodes(self):
        return self.tree_predictor.get_n_leaf_nodes()

    def get_max_depth(self):
        return self.tree_predictor.get_max_depth()

    def compute_partial_dependence(self, *args):
        return self.tree_predictor.compute_partial_dependence(*args)

    @property
    def nodes(self):
        return self.tree_predictor.nodes


@njit(nogil=True)
def _fit_bias(*, y_train, raw_predictions, intercept_train, intercept_val):
    """
    Parameters
    ----------

    y_train : array-like of shape (n_samples,)
        Target values.
    raw_predictions : array-like of shape (n_samples, n_trees_per_iteration)
        The raw predictions of the model so far, speeds up calculations
        since these are not recomputed.
    intercept_train : array-like of shape (n_samples,)
        A vector of int-encoded labels which indicate from which corpus
        a given observation is from.
    intercept_val : array-like of shape (n_samples,)
        A vector of int-encoded labels which indicate from which corpus
        a given observation is from.
    """
    n_train = len(y_train)
    n_corpuses = max(intercept_train) + 1
    mean_y = np.zeros(n_corpuses, dtype=y_train.dtype)
    mean_pred = np.zeros(n_corpuses, dtype=raw_predictions.dtype)

    for i in range(n_train):
        mean_y[intercept_train[i]] += y_train[i]
        mean_pred[intercept_train[i]] += np.exp(raw_predictions[i, 0])

    bias = np.log(mean_y) - np.log(mean_pred)

    return (
        np.expand_dims(bias[intercept_train], -1),
        np.expand_dims(bias[intercept_val], -1),
    )


class BaseCustomBinnedGradientBooster(BaseHistGradientBoosting):
    """Base class for histogram-based gradient boosting estimators."""

    """@staticmethod
    def get_log_weight(*, 
                       y_train, 
                       raw_predictions, 
                       intercept_train, 
                       intercept_val
                    ):

        log_mean_y = np.log(y_train[None,:] @ intercept_train)
        log_mean_predictions = np.log(np.exp(raw_predictions.ravel()) @ intercept_train)

        log_w_t = log_mean_y - log_mean_predictions

        print(log_mean_y, log_mean_predictions, log_w_t)

        bias_t_train = intercept_train.multiply(log_w_t).sum(1)
        bias_t_val = intercept_val.multiply(log_w_t).sum(1)

        return bias_t_train, bias_t_val"""

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        *,
        raw_predictions,
        sample_weight,
        intercept_idx,
        svi_shrinkage=1.0,
    ):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) default=None
            Weights of training data.

        raw_predictions : array-like of shape (n_samples, n_trees_per_iteration)
            The raw predictions of the model so far, speeds up calculations
            since these are not recomputed.

        desing_matrix : array-like of shape (n_samples, n_distributions)
            A matrix of {0,1}, where a 1 in some column indicates that the
            observation is from the corresponding multinomial distribution.

        svi_shrinkage : float, default=1.
            The shrinkage factor for the stochastic variational inference.
            algorithm. If 1., then no shrinkage is applied.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        fit_start_time = time()
        acc_find_split_time = 0.0  # time spent finding the best splits
        acc_apply_split_time = 0.0  # time spent splitting nodes
        acc_compute_hist_time = 0.0  # time spent computing histograms
        # time spent predicting X for gradient and hessians update
        acc_prediction_time = 0.0
        # X, y = self._validate_data(X, y, dtype=[X_DTYPE], force_all_finite=False)
        X = self._preprocess_X(X, reset=False)
        y = _check_y(y, estimator=self)
        y = self._encode_y(y)
        check_consistent_length(X, y)
        # Do not create unit sample weights by default to later skip some
        # computation
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            # TODO: remove when PDP supports sample weights
            self._fitted_with_sw = True

        sample_weight = self._finalize_sample_weight(sample_weight, y)

        rng = check_random_state(self.random_state)

        raw_predictions = np.asarray(raw_predictions, dtype=X_DTYPE, order="F")
        if raw_predictions.shape != (X.shape[0], self.n_trees_per_iteration_):
            raise ValueError(
                "raw_predictions must have shape (n_samples, n_trees_per_iteration_)"
            )

        # When warm starting, we want to reuse the same seed that was used
        # the first time fit was called (e.g. for subsampling or for the
        # train/val split).
        # When warm starting, we want to reuse the same seed that was used
        # the first time fit was called (e.g. train/val split).
        # For feature subsampling, we want to continue with the rng we started with.
        if not self.warm_start or not self._is_fitted():
            self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
            feature_subsample_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
            self._feature_subsample_rng = np.random.default_rng(feature_subsample_seed)

        # self._validate_parameters()
        monotonic_cst = _check_monotonic_cst(self, self.monotonic_cst)

        # used for validation in predict
        n_samples, self._n_features = X.shape

        # Encode constraints into a list of sets of features indices (integers).
        interaction_cst = self._check_interaction_cst(self._n_features)

        # we need this stateful variable to tell raw_predict() that it was
        # called from fit() (this current method), and that the data it has
        # received is pre-binned.
        # predicting is faster on pre-binned data, so we want early stopping
        # predictions to be made on pre-binned data. Unfortunately the _scorer
        # can only call predict() or predict_proba(), not raw_predict(), and
        # there's no way to tell the scorer that it needs to predict binned
        # data.
        self._in_fit = True

        # `_openmp_effective_n_threads` is used to take cgroups CPU quotes
        # into account when determine the maximum number of threads to use.
        n_threads = _openmp_effective_n_threads()

        if isinstance(self.loss, str):
            self._loss = self._get_loss(sample_weight=sample_weight)
        elif isinstance(self.loss, BaseLoss):
            self._loss = self.loss

        if self.early_stopping == "auto":
            self.do_early_stopping_ = n_samples > 10000
        else:
            self.do_early_stopping_ = self.early_stopping

        # create validation data if needed
        self._use_validation_data = self.validation_fraction is not None
        if self.do_early_stopping_ and self._use_validation_data:
            # stratify for classification
            # instead of checking predict_proba, loss.n_classes >= 2 would also work
            stratify = y if hasattr(self._loss, "predict_proba") else None

            # Save the state of the RNG for the training and validation split.
            # This is needed in order to have the same split when using
            # warm starting.
            (
                X_train,
                X_val,
                y_train,
                y_val,
                sample_weight_train,
                sample_weight_val,
                raw_predictions,
                raw_predictions_val,
                intercept_train,
                intercept_val,
            ) = train_test_split(
                X,
                y,
                sample_weight,
                raw_predictions,
                intercept_idx,
                test_size=self.validation_fraction,
                stratify=stratify,
                random_state=self._random_seed,
            )
        else:
            raise NotImplementedError()
            X_train, y_train, sample_weight_train = X, y, sample_weight
            X_val = y_val = sample_weight_val = None

        X_binned_train = self._bin_data(
            X_train, n_threads=n_threads, is_training_data=True
        )
        if X_val is not None:
            X_binned_val = self._bin_data(
                X_val, n_threads=n_threads, is_training_data=False
            )
        else:
            X_binned_val = None

        # Uses binned data to check for missing values
        has_missing_values = (
            (X_binned_train == self._bin_mapper.missing_values_bin_idx_)
            .any(axis=0)
            .astype(np.uint8)
        )

        n_bins = self.max_bins + 1

        if self.verbose:
            print("Fitting gradient boosted rounds:")

        n_samples = X_binned_train.shape[0]

        # First time calling fit, or no warm start
        if not (self._is_fitted() and self.warm_start):
            # Clear random state and score attributes
            self._clear_state()

            # initialize raw_predictions: those are the accumulated values
            # predicted by the trees for the training data. raw_predictions has
            # shape (n_samples, n_trees_per_iteration) where
            # n_trees_per_iterations is n_classes in multiclass classification,
            # else 1.
            # self._baseline_prediction has shape (1, n_trees_per_iteration)

            # if not np.isclose(raw_predictions, 0.).all():
            #    warnings.warn('Initial raw predictions are not set to 0. This will bias the predictions of the model.')

            self._baseline_prediction = np.zeros_like(raw_predictions[0])

            # predictors is a matrix (list of lists) of TreePredictor objects
            # with shape (n_iter_, n_trees_per_iteration)
            self._predictors = predictors = []

            # Initialize structures and attributes related to early stopping
            self._scorer = None  # set if scoring != loss
            # raw_predictions_val = None  # set if scoring == loss and use val
            self.train_score_ = []
            self.validation_score_ = []

            if self.do_early_stopping_:
                # populate train_score and validation_score with the
                # predictions of the initial model (before the first tree)

                if self.scoring == "loss":
                    # we're going to compute scoring w.r.t the loss. As losses
                    # take raw predictions as input (unlike the scorers), we
                    # can optimize a bit and avoid repeating computing the
                    # predictions of the previous trees. We'll reuse
                    # raw_predictions (as it's needed for training anyway) for
                    # evaluating the training loss, and create
                    # raw_predictions_val for storing the raw predictions of
                    # the validation data.
                    self._check_early_stopping_loss(
                        raw_predictions=raw_predictions,
                        y_train=y_train,
                        # intercept_train=intercept_train,
                        sample_weight_train=sample_weight_train,
                        raw_predictions_val=raw_predictions_val,
                        # intercept_val=intercept_val,
                        y_val=y_val,
                        sample_weight_val=sample_weight_val,
                        n_threads=n_threads,
                    )
                else:
                    raise NotImplementedError()

            begin_at_stage = 0

        # warm start: this is not the first time fit was called
        else:
            # Check that the maximum number of iterations is not smaller
            # than the number of iterations from the previous fit
            if self.max_iter < self.n_iter_:
                raise ValueError(
                    "max_iter=%d must be larger than or equal to "
                    "n_iter_=%d when warm_start==True" % (self.max_iter, self.n_iter_)
                )

            # Convert array attributes to lists
            self.train_score_ = self.train_score_.tolist()
            self.validation_score_ = self.validation_score_.tolist()

            """if self.do_early_stopping_ and self.scoring != "loss":
                # Compute the subsample set
                (
                    X_binned_small_train,
                    y_small_train,
                    sample_weight_small_train,
                ) = self._get_small_trainset(
                    X_binned_train, y_train, sample_weight_train, self._random_seed
                )"""

            # Get the predictors from the previous fit
            predictors = self._predictors

            begin_at_stage = self.n_iter_

        # initialize gradients and hessians (empty arrays).
        # shape = (n_samples, n_trees_per_iteration).
        gradient, hessian = self._loss.init_gradient_and_hessian(
            n_samples=n_samples, dtype=G_H_DTYPE, order="F"
        )

        assert self.n_trees_per_iteration_ == 1, "n_trees_per_iteration must be 1"

        n_trees_fit = 0
        for iteration in range(begin_at_stage, self.max_iter):
            if self.verbose:
                iteration_start_time = time()
                print(
                    "[{}/{}] ".format(iteration + 1, self.max_iter), end="", flush=True
                )

            # 1: fit the bias term here
            # this code finds the mean y and prediction for each distribution according to the design matrix.
            bias_t_train, bias_t_val = _fit_bias(
                y_train=y_train,
                raw_predictions=raw_predictions,
                intercept_train=intercept_train,
                intercept_val=intercept_val,
            )
            # There is no need to save the bias term, since it cancels out in the multinomial ll
            # The bias is only needed to ensure that poisson regression produces the same parameter estimates.
            # get bias term for each observation
            raw_predictions += bias_t_train
            raw_predictions_val += bias_t_val
            #

            # Update gradients and hessians, inplace
            # Note that self._loss expects shape (n_samples,) for
            # n_trees_per_iteration = 1 else shape (n_samples, n_trees_per_iteration).
            if self._loss.constant_hessian:
                self._loss.gradient(
                    y_true=y_train,
                    raw_prediction=raw_predictions,
                    sample_weight=sample_weight_train,
                    gradient_out=gradient,
                    n_threads=n_threads,
                )
            else:
                self._loss.gradient_hessian(
                    y_true=y_train,
                    raw_prediction=raw_predictions,
                    sample_weight=sample_weight_train,
                    gradient_out=gradient,
                    hessian_out=hessian,
                    n_threads=n_threads,
                )

            # Append a list since there may be more than 1 predictor per iter
            predictors.append([])

            # 2-d views of shape (n_samples, n_trees_per_iteration_) or (n_samples, 1)
            # on gradient and hessian to simplify the loop over n_trees_per_iteration_.
            if gradient.ndim == 1:
                g_view = gradient.reshape((-1, 1))
                h_view = hessian.reshape((-1, 1))
            else:
                g_view = gradient
                h_view = hessian

            # Build `n_trees_per_iteration` trees.
            for k in range(self.n_trees_per_iteration_):
                grower = TreeGrower(
                    X_binned=X_binned_train,
                    gradients=g_view[:, k],
                    hessians=h_view[:, k],
                    n_bins=n_bins,
                    n_bins_non_missing=self._bin_mapper.n_bins_non_missing_,
                    has_missing_values=has_missing_values,
                    is_categorical=self.is_categorical_,
                    monotonic_cst=monotonic_cst,
                    interaction_cst=interaction_cst,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    l2_regularization=self.l2_regularization,
                    shrinkage=self.learning_rate,
                    feature_fraction_per_split=self.max_features,
                    rng=self._feature_subsample_rng,
                    n_threads=n_threads,
                )

                grower.grow()

                acc_apply_split_time += grower.total_apply_split_time
                acc_find_split_time += grower.total_find_split_time
                acc_compute_hist_time += grower.total_compute_hist_time

                if self._loss.need_update_leaves_values:
                    _update_leaves_values(
                        loss=self._loss,
                        grower=grower,
                        y_true=y_train,
                        raw_prediction=raw_predictions[:, k],
                        sample_weight=sample_weight_train,
                    )

                predictor = grower.make_predictor(
                    binning_thresholds=self._bin_mapper.bin_thresholds_
                )

                predictors[-1].append(predictor)

                # Update raw_predictions with the predictions of the newly
                # created tree.
                tic_pred = time()
                _update_raw_predictions(raw_predictions[:, k], grower, n_threads)
                toc_pred = time()
                acc_prediction_time += toc_pred - tic_pred

            should_early_stop = False
            if self.do_early_stopping_:
                if self.scoring == "loss":
                    # Update raw_predictions_val with the newest tree(s)
                    if self._use_validation_data:
                        for k, pred in enumerate(self._predictors[-1]):
                            raw_predictions_val[:, k] += pred.predict_binned(
                                X_binned_val,
                                self._bin_mapper.missing_values_bin_idx_,
                                n_threads,
                            )

                    should_early_stop = self._check_early_stopping_loss(
                        raw_predictions=raw_predictions,
                        y_train=y_train,
                        # intercept_train=intercept_train,
                        sample_weight_train=sample_weight_train,
                        raw_predictions_val=raw_predictions_val,
                        y_val=y_val,
                        # intercept_val=intercept_val,
                        sample_weight_val=sample_weight_val,
                        n_threads=n_threads,
                    )

                else:
                    raise NotImplementedError()

            if self.verbose:
                self._print_iteration_stats(iteration_start_time)

            n_trees_fit += 1

            # maybe we could also early stop if all the trees are stumps?
            ## DELETE THIS
            # should_early_stop = False

            if should_early_stop:
                break

        # change most recently-added predictors to shrunk predictors
        for i in range(1, n_trees_fit + 1):
            for k in range(self.n_trees_per_iteration_):
                self._predictors[-i][k] = ShrunkTreePredictor(
                    self._predictors[-i][k], svi_shrinkage
                )

        if self.verbose:
            duration = time() - fit_start_time
            n_total_leaves = sum(
                predictor.get_n_leaf_nodes()
                for predictors_at_ith_iteration in self._predictors
                for predictor in predictors_at_ith_iteration
            )
            n_predictors = sum(
                len(predictors_at_ith_iteration)
                for predictors_at_ith_iteration in self._predictors
            )
            print(
                "Fit {} trees in {:.3f} s, ({} total leaves)".format(
                    n_predictors, duration, n_total_leaves
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent computing histograms:", acc_compute_hist_time
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent finding best splits:", acc_find_split_time
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent applying splits:", acc_apply_split_time
                )
            )
            print(
                "{:<32} {:.3f}s".format("Time spent predicting:", acc_prediction_time)
            )

        self.train_score_ = np.asarray(self.train_score_)
        self.validation_score_ = np.asarray(self.validation_score_)
        del self._in_fit  # hard delete so we're sure it can't be used anymore

        return self

    def _preprocess_X(self, X, *, reset):
        """Preprocess and validate X.

        Parameters
        ----------
        X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
            Input data.

        reset : bool
            Whether to reset the `n_features_in_` and `feature_names_in_ attributes.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Validated input data.

        known_categories : list of ndarray of shape (n_categories,)
            List of known categories for each categorical feature.
        """
        # If there is a preprocessor, we let the preprocessor handle the validation.
        # Otherwise, we validate the data ourselves.
        check_X_kwargs = dict(dtype=[X_DTYPE], force_all_finite=False)

        if reset:
            self.is_categorical_ = self._check_categorical_features(X)
            self.n_features_in_ = X.shape[1]

        return self._validate_data(X, reset=False, **check_X_kwargs)

    def fit_binning(self, X, known_categories):

        n_threads = _openmp_effective_n_threads()
        # self.is_categorical_, known_categories = self._check_categories(X)
        X = self._preprocess_X(X, reset=True)

        known_categories = [
            np.array(c, dtype=X_DTYPE) if c is not None else None
            for c in known_categories
        ]

        n_bins = self.max_bins + 1  # + 1 for missing values
        self._bin_mapper = _BinMapper(
            n_bins=n_bins,
            is_categorical=self.is_categorical_,
            known_categories=known_categories,
            random_state=42,
            n_threads=n_threads,
        ).fit(X)

        return self

    def _bin_data(self, X, n_threads=1, is_training_data=True):
        """Bin data X.

        If is_training_data, then fit the _bin_mapper attribute.
        Else, the binned data is converted to a C-contiguous array.
        """

        # Bin the data
        # For ease of use of the API, the user-facing GBDT classes accept the
        # parameter max_bins, which doesn't take into account the bin for
        # missing values (which is always allocated). However, since max_bins
        # isn't the true maximal number of bins, all other private classes
        # (binmapper, histbuilder...) accept n_bins instead, which is the
        # actual total number of bins. Everywhere in the code, the
        # convention is that n_bins == max_bins + 1
        try:
            self._bin_mapper
        except AttributeError:
            raise AttributeError(
                "The binning step should be performed before calling this method."
            )

        description = "training" if is_training_data else "validation"
        if self.verbose:
            print(
                "Binning {:.3f} GB of {} data: ".format(X.nbytes / 1e9, description),
                end="",
                flush=True,
            )
        tic = time()

        X_binned = self._bin_mapper.transform(X)

        if not is_training_data:
            X_binned = np.ascontiguousarray(X_binned)

        toc = time()
        if self.verbose:
            duration = toc - tic
            print("{:.3f} s".format(duration))

        return X_binned

    def _raw_predict_from(
        self,
        X,
        raw_predictions,
        from_iteration=0,
        check_input=True,
    ):

        is_binned = getattr(self, "_in_fit", False)

        if check_input:
            X = self._preprocess_X(X, reset=False)

        check_is_fitted(self)

        if X.shape[1] != self._n_features:
            raise ValueError(
                "X has {} features but this estimator was trained with "
                "{} features.".format(X.shape[1], self._n_features)
            )

        # We intentionally decouple the number of threads used at prediction
        # time from the number of threads used at fit time because the model
        # can be deployed on a different machine for prediction purposes.
        n_threads = _openmp_effective_n_threads()
        self._predict_iterations(
            X, self._predictors[from_iteration:], raw_predictions, is_binned, n_threads
        )
        return raw_predictions


class CustomHistGradientBooster(
    BaseCustomBinnedGradientBooster, HistGradientBoostingRegressor
):
    pass
