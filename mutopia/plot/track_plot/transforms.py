from functools import partial
import numpy as np
from mutopia.gtensor import fetch_features
from numpy._core._multiarray_umath import _array_converter


def _xarr_op(fn):
    """
    Wrap a numpy function to work with xarray objects.

    Parameters
    ----------
    fn : callable
        Function that operates on numpy arrays

    Returns
    -------
    callable
        Wrapped function that preserves xarray structure
    """

    def run_fn(x):
        conv = _array_converter(x)
        out = fn(x)
        return conv.wrap(out)

    return run_fn


def _moving_average(bin_width, arr, alpha=10):
    """
    Compute moving average with optional bin width weighting.

    Parameters
    ----------
    bin_width : array-like or None
        Width of each bin for weighted averaging.
        If None, uses simple moving average.
    arr : array-like
        Input array to smooth
    alpha : int, default 10
        Window size for moving average

    Returns
    -------
    numpy.ndarray
        Smoothed array of same length as input
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


def passthrough(data):
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

    def _passthrough(*args, **kwargs):
        return data

    return _passthrough


def pipeline(*fns):
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

    def _pipeline(data):
        for fn in fns:
            data = fn(data)
        return data

    return _pipeline


def select(var_name, **sel):
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

    def _accessor(dataset):
        return dataset[var_name].sel(**sel).transpose(..., "locus")

    return _accessor


def feature_matrix(*feature_names, source=None):
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


def clip(min_quantile=0.0, max_quantile=1.0):
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

    def _clip(arr):
        return np.clip(
            arr, np.nanquantile(arr, min_quantile), np.nanquantile(arr, max_quantile)
        )

    return _xarr_op(_clip)


def renorm(x):
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


def apply_rows(fn):
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
