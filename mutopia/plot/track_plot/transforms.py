"""
Transform helpers and accessors for genome track plotting.

This module provides small utilities used by track plotting, including
array transforms that preserve xarray objects, dataset accessors, and
clustering helpers.
"""
from __future__ import annotations
from functools import partial
import numpy as np
import warnings
from numpy._core._multiarray_umath import _array_converter
from mutopia.gtensor import fetch_features
from typing import Any, Callable, Mapping, Optional, Sequence, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray, Dataset
    from pandas import DataFrame


def _xarr_op(
    fn: Callable[[np.ndarray], np.ndarray]
) -> Callable[[np.ndarray | "DataArray"], np.ndarray | "DataArray"]:
    """
    Wrap a NumPy transform to preserve xarray objects.

    Parameters
    ----------
    fn : callable
        Function that accepts and returns a numpy array with the same shape.

    Returns
    -------
    callable
        A function that can be applied to either numpy arrays or xarray
        DataArray objects, returning the same type as the input.
    """

    def run_fn(x):
        conv = _array_converter(x)
        out = fn(x)
        return conv.wrap(out)

    return run_fn


def _moving_average(
    bin_width: Optional[np.ndarray],
    arr: np.ndarray,
    alpha: int = 10,
) -> np.ndarray:
    """
    Moving average with optional per-bin weighting.

    Parameters
    ----------
    bin_width : array-like or None
        If provided, each value in ``arr`` is weighted by the corresponding
        bin width in a window of size ``alpha``. If None, a simple unweighted
        moving average is used.
    arr : ndarray
        Input 1D array.
    alpha : int, default 10
        Window size.

    Returns
    -------
    ndarray
        Smoothed array with the same shape as ``arr``.
    """

    if bin_width is None:
        weights = np.ones(alpha) / alpha
        ema = np.convolve(arr, weights, mode="same")
    else:
        # Fix moving average rate to weighted average rate to use sum(bin width * rate)/ (total bin width)
        window = np.ones(alpha)
        weighted_sum = np.convolve(arr * bin_width, window, mode="same")
        total_weight = np.convolve(bin_width, window, mode="same")

        # Compute the weighted moving average
        ema = weighted_sum / total_weight

    return ema


def passthrough(data: Any) -> Callable[..., Any]:
    """
    Create a passthrough function that returns input data unchanged.

    This function creates a closure that ignores any arguments passed to it
    and always returns the original data object. Useful in data processing
    pipelines where certain steps should be bypassed.

    Parameters
    ----------
    data : any
        Input data to be returned unchanged by the generated function

    Returns
    -------
    callable
        Function that accepts any arguments but always returns the original data
    """

    def _passthrough(*args: Any, **kwargs: Any) -> Any:
        return data

    return _passthrough


def pipeline(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Create a data processing pipeline from a sequence of functions.

    This function composes multiple functions into a single pipeline function
    that applies each function in sequence. The output of each function becomes
    the input to the next function in the pipeline.

    Parameters
    ----------
    *fns : callable
        Variable number of functions to compose into a pipeline.
        Each function should accept one argument (the data) and return
        the transformed data for the next function.

    Returns
    -------
    callable
        Composed function that applies all input functions in sequence
        from first to last

    Examples
    --------
    >>> normalize = lambda x: x / x.max()
    >>> log_transform = lambda x: np.log(x + 1)
    >>> process = pipeline(normalize, log_transform)
    >>> result = process(data)
    """

    def _pipeline(data: Any) -> Any:
        for fn in fns:
            data = fn(data)
        return data

    return _pipeline


def select(var_name: str, **sel: Any) -> Callable[["Dataset"], "DataArray"]:
    """
    Create an accessor function to extract variables from datasets.

    This function creates a closure that extracts a specific variable from
    a dataset and optionally applies selection criteria. The extracted
    variable is transposed to ensure 'locus' is the last dimension.

    Parameters
    ----------
    var_name : str
        Name of the variable to access from the dataset
    **sel : dict
        Additional selection criteria passed to .sel() method.
        Keys should be dimension names and values should be selection criteria.

    Returns
    -------
    callable
        Function that takes a dataset and returns the specified variable
        with 'locus' as the last dimension

    Examples
    --------
    >>> get_feature = select("Features/gc_content", sample=0)
    >>> feature_data = get_feature(dataset)
    """

    def _accessor(dataset: "Dataset") -> "DataArray":
        return dataset[var_name].sel(**sel).squeeze().transpose(..., "locus")
    
    return _accessor


def feature_matrix(
    *feature_names: str,
    source: Optional[str] = None,
) -> Callable[["Dataset"], "DataArray"]:
    """
    Accessor function to retrieve multiple features from a dataset as a matrix.

    This function creates an accessor that extracts multiple features from a
    dataset and stacks them into a 2D matrix with features as rows and loci
    as columns. If no feature names are provided, it automatically selects
    all numeric features from the dataset.

    Parameters
    ----------
    *feature_names : str or iterable
        Names of the features to access. Can be:
        - Multiple string arguments: feature_matrix("feat1", "feat2", "feat3")
        - Single iterable: feature_matrix(["feat1", "feat2", "feat3"])
        - Empty: automatically selects all numeric features

    Parameters
    ----------
    source : str, optional
        Optional feature source or namespace passed through to ``fetch_features``.

    Returns
    -------
    callable
        Function that retrieves the specified features from the dataset
        and returns them as a DataArray with dimensions (feature, locus).
        If only one feature is selected, the 'feature' dimension is squeezed.

    Examples
    --------
    >>> get_features = feature_matrix("gc_content", "cpg_density")
    >>> matrix = get_features(dataset)  # Shape: (2, n_loci)

    >>> get_all_features = feature_matrix()
    >>> all_matrix = get_all_features(dataset)  # All numeric features
    """
    return lambda dataset: fetch_features(dataset, *feature_names, source=source)


def clip(min_quantile: float = 0.0, max_quantile: float = 1.0) -> Callable[[np.ndarray | "DataArray"], np.ndarray | "DataArray"]:
    """
    Create a clipping function based on quantiles.

    Parameters
    ----------
    min_quantile : float, default 0.0
        Lower quantile for clipping (0-1)
    max_quantile : float, default 1.0
        Upper quantile for clipping (0-1)

    Returns
    -------
    callable
        Function that clips input arrays to specified quantiles
    """

    def _clip(arr: np.ndarray) -> np.ndarray:
        return np.clip(
            arr, np.nanquantile(arr, min_quantile), np.nanquantile(arr, max_quantile)
        )

    return _xarr_op(_clip)


def renorm(x: np.ndarray) -> np.ndarray:
    """
    Renormalize array to sum to 1.

    Parameters
    ----------
    x : array-like
        Input array

    Returns
    -------
    array-like
        Normalized array that sums to 1
    """

    return x / np.nansum(x)


def minmax_scale(x: np.ndarray) -> np.ndarray:
    """
    Scale array to [0, 1] using min-max normalization.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    ndarray
        Rescaled array with values in [0, 1].
    """
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def apply_rows(fn: Callable[[np.ndarray], Any]) -> Callable[..., np.ndarray]:
    """
    Create function to apply operation along rows (axis=1).

    Parameters
    ----------
    fn : callable
        Function to apply to each row

    Returns
    -------
    callable
        Function that applies fn along axis 1
    """

    return partial(np.apply_along_axis, fn, 1)


def _get_optimal_row_order(data: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Compute an order of rows using hierarchical clustering with optimal leaf ordering.

    Parameters
    ----------
    data : ndarray
        2D numeric array. NaNs and infs will be replaced with zeros for clustering.
    **kwargs
        Additional keyword args passed to ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    ndarray
        Indices representing an ordering of rows.
    """

    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list

    if (~np.isfinite(data)).any():
        warnings.warn(
            "Data contains NaN or infinite values. Filling with zeros for clustering."
        )
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return leaves_list(optimal_leaf_ordering(linkage(data, **kwargs), data))


def reorder_df(df: "DataFrame") -> "DataFrame":
    """
    Reorder a DataFrame's rows using hierarchical clustering optimal order.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with numeric values.

    Returns
    -------
    pandas.DataFrame
        Reordered DataFrame according to optimal leaf ordering.
    """
    optimal_order = _get_optimal_row_order(df.values)
    return df.iloc[optimal_order]


class TopographyTransformer:
    """
    Transformer for mutational topography matrices.

    Fetches and standardizes a configuration x mutation x context matrix from
    a dataset, computes an informative row ordering, and provides labels for
    grouped mutation categories.
    """

    def __init__(
        self,
        mutation_order: Sequence[str] = ("C>G", "C>A", "T>A", "T>C", "T>G", "C>T"),
        data_key: str = "predicted_marginal",
    ) -> None:
        self.data_key = data_key
        self.mutation_order = mutation_order

    @staticmethod
    def _fetch_matrix(data_key: str, dataset: "Dataset") -> "DataFrame":
        topography_matrix = (
            np.log(dataset[data_key] / dataset["predicted_marginal_locus"])
            .stack(observation=("context", "configuration"))
            .transpose(..., "locus")
            .to_pandas()
        )

        mutation = topography_matrix.index.get_level_values("context").str.slice(2, 5)
        topography_matrix["mutation"] = mutation
        topography_matrix = topography_matrix.reset_index().set_index(
            ["configuration", "mutation", "context"]
        )
        return topography_matrix.T

    @staticmethod
    def _transform(topography_matrix: "DataFrame", mu: np.ndarray, std: np.ndarray) -> "DataFrame":
        return (topography_matrix - mu) / std

    def _get_order(self, topography_matrix: "DataFrame") -> list[tuple[str, str, str]]:
        topography_matrix = topography_matrix.T
        context_order = []
        for mutation_type in self.mutation_order:
            df = topography_matrix.loc["C/T-centered", mutation_type]
            context_order.extend(reorder_df(df).index.values)

        topography_order = [
            (configuration, context[2:5], context)
            for configuration in ["C/T-centered", "A/G-centered"]
            for context in (
                context_order[::-1]
                if configuration == "C/T-centered"
                else context_order
            )
        ]
        return topography_order

    def _axis_labels_from_order(self, ordering: Sequence[tuple[str, str, str]]) -> list[str]:
        complement = {"A": "T", "C": "G", "T": "A", "G": "C"}

        def _get_label(configuration: str, mutation: str) -> str:
            l, r = mutation.split(">")
            if configuration == "A/G-centered":
                return f"{complement[l]}>{complement[r]}"
            return mutation

        labels = [
            _get_label(configuration, mutation)
            for configuration, mutation, _ in ordering[8::16]
        ]

        return labels

    @property
    def labels(self) -> list[str]:
        return self.labels_

    def fit(self, dataset: "Dataset") -> "TopographyTransformer":
        x = self._fetch_matrix(self.data_key, dataset)

        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)

        self.ordering_ = self._get_order(self._transform(x, self.mean_, self.std_))
        self.labels_ = self._axis_labels_from_order(self.ordering_)

        return self

    def transform(self, dataset: "Dataset") -> np.ndarray:
        x = self._fetch_matrix(self.data_key, dataset)
        x = self._transform(x, self.mean_, self.std_)
        return x[self.ordering_].T.values
