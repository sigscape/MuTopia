"""
GTensor module for genomic tensor analysis.
This module provides functionality for creating, manipulating, and analyzing genomic tensors,
including loading datasets, applying transformations, and generating explanations for model components.
"""

from os import path
import xarray as xr
from pandas import concat as pd_concat
import numpy as np
from typing import Union, List
from numpy.typing import NDArray
from pandas import IntervalIndex, Index, Interval, DataFrame
from ..utils import logger
from ..genome_utils.bed12_utils import unstack_regions as _unstack_regions
from functools import reduce
import os
from tqdm import tqdm
from .interfaces import *
import mutopia.gtensor.disk_interface as disk
from ..utils import parse_region

__all__ = [
    "GTensor",
    "apply_to_samples",
    "fetch_features",
    "load_dataset",
    "train_test_split",
    "lazy_load",
    "eager_load",
    "lazy_train_test_load",
    "eager_train_test_load",
    "num_sources",
    "is_mixture_dataset",
    "list_sources",
    "fetch_source",
    "get_explanation",
    "equal_size_quantiles",
    "slice_regions",
    "annot_empirical_marginal",
    "make_mixture_dataset",
    "match_dims",
    "dims_except_for",
    "unstack_regions",
    "mutate_method",
    "BED_COLS",
]

BED_COLS = [
    "Regions/chrom",
    "Regions/start",
    "Regions/end",
]


def GTensor(
    modality,
    *,
    name: str,
    chrom: List[str],
    start: List[int],
    end: List[int],
    context_frequencies: xr.DataArray,
    exposures: Union[None, NDArray[np.number]] = None,
    dtype=None,
):
    """
    Create a GTensor dataset for genomic tensor analysis.

    This function constructs an xarray Dataset with the standardized structure
    required for genomic tensor operations, including region coordinates,
    context frequencies, and metadata.

    Parameters
    ----------
    modality : object
        Modality object containing coordinate information and mode configuration
    name : str
        Name identifier for the dataset
    chrom : List[str]
        List of chromosome names for each genomic region
    start : List[int]
        List of start positions for each genomic region
    end : List[int]
        List of end positions for each genomic region
    context_frequencies : xr.DataArray
        Array containing context frequency data for each region
    exposures : Union[None, NDArray[np.number]], optional
        Exposure values for each region. If None, defaults to ones
    dtype : optional
        Data type for the dataset. If None, uses modality.MODE_ID

    Returns
    -------
    xr.Dataset
        Structured dataset with regions, coordinates, and metadata
    """

    locus_coords = Index(np.arange(len(chrom)))

    shared_coords = {
        **modality.coords,
        "locus": locus_coords,
        "sample": [],
    }

    region_lengths = np.sum(
        context_frequencies.data, axis=tuple(range(context_frequencies.data.ndim - 1))
    )

    if exposures is None:
        exposures = np.ones(len(locus_coords), dtype=np.float32)

    return xr.Dataset(
        {
            "Regions/context_frequencies": context_frequencies,
            "Regions/length": xr.DataArray(np.array(region_lengths), dims=("locus",)),
            "Regions/exposures": xr.DataArray(np.squeeze(exposures), dims=("locus",)),
            "Regions/chrom": xr.DataArray(np.array(chrom), dims=("locus",)),
            "Regions/start": xr.DataArray(np.array(start), dims=("locus",)),
            "Regions/end": xr.DataArray(np.array(end), dims=("locus",)),
        },
        coords=shared_coords,
        attrs={
            "name": name,
            "dtype": dtype or modality.MODE_ID,
        },
    )


def apply_to_samples(data, func, bar=True):
    """
    Apply a function to each sample in a dataset with parallel processing.

    This function applies a given function to each sample (region) in the dataset,
    handling the parallelization and aggregation of results. It's designed for
    operations that need to process each genomic region independently.

    Parameters
    ----------
    data : object
        Input dataset or data loader containing samples to process
    func : callable
        Function to apply to each sample. Should accept a dataset slice and
        return a result that can be concatenated
    bar : bool, default=True
        Whether to display a progress bar during processing

    Returns
    -------
    xr.Dataset
        Dataset containing the concatenated results from all sample applications
    """

    if not hasattr(data, "X"):
        data = LazySampleLoader(data)

    return xr.concat(
        [
            func(data.fetch_sample(sample_name))
            for sample_name in (
                data.list_samples()
                if not bar
                else tqdm(data.list_samples(), desc="Applying function to samples")
            )
        ],
        dim="sample",
    )


def mutate(func):
    """
    Decorator function to modify a dataset in place.

    This decorator allows running mutations on a dataset without disrupting
    the interface chains. It wraps a function to work with the dataset's
    mutate method.

    Parameters
    ----------
    func : callable
        Function that takes a dataset as first argument and returns
        a modified dataset

    Returns
    -------
    callable
        Wrapped function that can be used with dataset.mutate()
    """

    def wrapper(dataset, *args, **kwargs):
        return dataset.mutate(lambda x: func(x, *args, **kwargs))

    return wrapper


def mutate_method(func):
    """
    Decorator function to modify a dataset in place for class methods.

    This decorator allows running mutations on a dataset without disrupting
    the interface chains, specifically for methods that take 'self' as the
    first parameter.

    Parameters
    ----------
    func : callable
        Method that takes self and dataset as first two arguments and
        returns a modified dataset

    Returns
    -------
    callable
        Wrapped method that can be used with dataset.mutate()
    """

    def wrapper(self, dataset, *args, **kwargs):
        return dataset.mutate(lambda x: func(self, x, *args, **kwargs))

    return wrapper


def load_dataset(dataset, with_samples=True, with_state=True):
    """
    Load a dataset from disk with configurable loading options.

    This function loads a dataset from disk storage. The loading behavior can be customized
    based on whether samples and state information should be included.

    Parameters
    ----------
    dataset : str or path-like
        Path or identifier for the dataset to load
    with_samples : bool, default=True
        Whether to load sample data along with the dataset structure
    with_state : bool, default=True
        Whether to load state information (model parameters, etc.)

    Returns
    -------
    LazySampleLoader or CorpusInterface
        Loaded dataset interface. Returns LazySampleLoader if with_samples=False,
        otherwise returns CorpusInterface
    """
    return (LazySampleLoader if not with_samples else CorpusInterface)(
        disk.load_dataset(dataset, with_samples=with_samples, with_state=with_state)
    )


def train_test_split(dataset, *test_chroms: Union[str, List[str]], lazy=False):
    """
    Split a dataset into training and testing sets based on chromosomes.

    This function splits the dataset by chromosomes, with specified chromosomes
    reserved for testing and the remainder used for training. The split can be
    performed eagerly (loading all data) or lazily (for memory efficiency).

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to split
    *test_chroms : Union[str, List[str]]
        Chromosome names to reserve for the test set. Can be provided as
        multiple string arguments or lists of strings
    lazy : bool, default=False
        Whether to perform lazy splitting. If True, returns LazySlicer objects
        that don't load data until accessed

    Returns
    -------
    tuple[CorpusInterface or LazySlicer, CorpusInterface or LazySlicer]
        Training and testing dataset interfaces

    Raises
    ------
    ValueError
        If no test chromosomes are provided or none of the specified
        chromosomes are found in the dataset
    """

    if not len(test_chroms) > 0:
        raise ValueError("No test chromosomes provided.")

    test_mask = dataset.sections["Regions"].chrom.isin(test_chroms)
    if test_mask.sum() == 0:
        raise ValueError(
            f'None of the chromosomes in {",".join(test_chroms)} are present in the dataset. '
        )

    lazy = lazy or not "X" in dataset.data_vars

    if lazy:
        logger.warning(
            "The dataset is lazy, so the train/test split will be lazy as well. "
            "This may cause latency issues on systems with slow file IO."
        )
        train = LazySlicer(dataset, locus=~test_mask)
        test = LazySlicer(dataset, locus=test_mask)

        drop_vars = (
            dataset.sections.groups["Features"] + dataset.sections.groups["Regions"]
        )
        train._base_corpus.corpus = train._base_corpus.drop_vars(drop_vars)
        # test._base_corpus.corpus = test._base_corpus.drop_vars(drop_vars)

        return train, test

    else:
        train = CorpusInterface(dataset.isel(locus=~test_mask))
        test = CorpusInterface(dataset.isel(locus=test_mask))

    return train, test


def lazy_load(dataset):
    """
    Load a dataset lazily without samples or state information.

    This is a convenience function that loads a dataset with minimal memory
    footprint by excluding sample data and state information.

    Parameters
    ----------
    dataset : str or path-like
        Path or identifier for the dataset to load

    Returns
    -------
    LazySampleLoader
        Lazy dataset interface that loads data on demand
    """
    return load_dataset(dataset, with_samples=False, with_state=False)


def eager_load(dataset):
    """
    Load a dataset eagerly with samples but without state information.

    This is a convenience function that loads a dataset with sample data
    but excludes state information for faster access patterns.

    Parameters
    ----------
    dataset : str or path-like
        Path or identifier for the dataset to load

    Returns
    -------
    CorpusInterface
        Eager dataset interface with samples loaded into memory
    """
    return load_dataset(dataset, with_samples=True, with_state=False)


def lazy_train_test_load(dataset, *test_chroms):
    """
    Load a dataset and perform lazy train/test split by chromosomes.

    This convenience function combines lazy loading with train/test splitting,
    providing memory-efficient access to training and testing data.

    Parameters
    ----------
    dataset : str or path-like
        Path or identifier for the dataset to load
    *test_chroms : str
        Chromosome names to reserve for the test set

    Returns
    -------
    tuple[LazySlicer, LazySlicer]
        Training and testing dataset slicers
    """
    return train_test_split(lazy_load(dataset), *test_chroms, lazy=True)


def eager_train_test_load(dataset, *test_chroms):
    """
    Load a dataset and perform eager train/test split by chromosomes.

    This convenience function combines eager loading with train/test splitting,
    loading all data into memory for fast access.

    Parameters
    ----------
    dataset : str or path-like
        Path or identifier for the dataset to load
    *test_chroms : str
        Chromosome names to reserve for the test set

    Returns
    -------
    tuple[CorpusInterface, CorpusInterface]
        Training and testing dataset interfaces
    """
    return train_test_split(eager_load(dataset), *test_chroms, lazy=False)


def num_sources(dataset):
    return len(list_sources(dataset))


def is_mixture_dataset(dataset):
    return num_sources(dataset) > 1


def list_sources(dataset):
    if "source" in dataset.coords:
        return dataset.coords["source"].values.tolist()
    return []


def fetch_source(dataset, source):
    sources = list_sources(dataset)
    if not source in sources:
        raise ValueError(f"Source {source} not found in dataset")

    groups = dataset.sections.groups
    state = groups.pop("State", [])
    features = groups.pop("Features", [])

    use_features = [path.basename(v) for v in features if v.split("/")[1] == source]
    use_state = [path.basename(v) for v in state if v.split("/")[1] == source]

    rename_map = {
        os.path.join("Features", source, v): os.path.join("Features", v)
        for v in use_features
    }
    rename_map.update(
        {os.path.join("State", source, v): os.path.join("State", v) for v in use_state}
    )

    other_dvars = [v for g in groups.values() for v in g]
    shared_features = [v for v in features if len(v.split("/")) == 2]
    shared_state = [v for v in state if len(v.split("/")) == 2]

    source_corpus = dataset[
        list(rename_map.keys()) + other_dvars + shared_features + shared_state
    ].rename(rename_map)

    source_corpus.attrs["name"] = dataset.attrs["name"] + "/" + source

    if "source" in source_corpus.dims:
        source_corpus = source_corpus.sel(source=source, drop=True)

    source_corpus = source_corpus.assign_coords(**dataset.coords).drop_dims("source")

    return source_corpus


def get_explanation(dataset, component):
    """
    Generate SHAP explanations for a specific model component.

    This function extracts and formats SHAP values for interpretability analysis,
    creating an explanation object that can be used with SHAP visualization tools.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing SHAP values and feature information
    component : str
        Name of the model component to explain

    Returns
    -------
    shap.Explanation
        SHAP explanation object with values, features, and display data

    Raises
    ------
    ImportError
        If SHAP library is not installed
    ValueError
        If the specified component doesn't have SHAP values in the dataset
    """

    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required to calculate SHAP values")

    if not component in dataset["SHAP_values"].shap_component.values:
        raise ValueError(
            f"The dataset does not have SHAP values for component {component}."
        )

    def _get_shap_from_source(dataset, component):

        shap_values = dataset["SHAP_values"]
        locus_dim = "locus" if "locus" in shap_values.dims else "shap_locus"

        shap_df = (
            shap_values.sel(shap_component=component)
            .to_pandas()
            .melt(ignore_index=False, var_name="feature", value_name="value")
            .reset_index()
        )

        # handle this case to remove the convolution
        if any(shap_df.feature.str.contains(":")):
            shap_df[["feature", "convolution"]] = shap_df.feature.str.split(
                ":", expand=True, n=1
            ).rename(columns={0: "feature", 1: "convolution"})

        locus_dim = "locus" if "locus" in shap_df.columns else "shap_locus"

        shap_df = (
            shap_df.groupby(["feature", locus_dim])["value"].sum().unstack().fillna(0).T
        )

        data = (
            dataset["State/locus_features"]
            .sel(locus=shap_df.index)
            .sel(
                feature=[
                    f"{s}:0" if f"{s}:0" in dataset.coords["feature"].values else s
                    for s in shap_df.columns
                ]
            )
        ).to_pandas()

        display_features = (
            dataset.sections["Features"]
            .assign_coords(locus=dataset.locus.data)
            .sel(locus=shap_df.index)
        )

        display_data = DataFrame(
            [display_features[s].data for s in shap_df.columns],
            index=shap_df.columns,
        ).T

        return (
            shap_df,
            data,
            display_data,
        )

    if is_mixture_dataset(dataset):
        if "ploidy" in dataset.data_vars:
            dataset = dataset.drop_vars("ploidy")

        shap_data = [
            _get_shap_from_source(fetch_source(dataset, source_name), component)
            for source_name in list_sources(dataset)
        ]
    else:
        shap_data = [_get_shap_from_source(dataset, component)]

    shap_df, data, display_data = [pd_concat(x) for x in zip(*shap_data)]

    expl = shap.Explanation(
        shap_df.values,
        feature_names=shap_df.columns,
        data=data.values,
        display_data=display_data,
    )

    return expl


def equal_size_quantiles(dataset, var_name, n_bins=10, key=None):
    """
    Create equal-size quantile bins for a variable in the dataset.

    This function bins the values of a specified variable into quantiles of equal
    cumulative region length, which is useful for creating balanced genomic bins.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the variable to bin
    var_name : str
        Name of the variable to create quantile bins for
    n_bins : int, default=10
        Number of quantile bins to create
    key : str, optional
        Custom name for the output bin variable. If None, generates name as
        '{var_name_base}_qbins_{n_bins}' where var_name_base is the last part
        of var_name after splitting on '/'

    Returns
    -------
    xarray.Dataset
        The input dataset with quantile bins added as a new variable
    """

    bin_nums = np.arange(dataset.sizes["locus"])
    sorted_vals = DataFrame(
        {
            "length": dataset.sections["Regions"].length.values,
            "value": dataset[var_name].values,
        },
        index=bin_nums,
    )
    sorted_vals = sorted_vals.sort_values(by="value", ascending=True)

    sorted_vals["cumm_fraction"] = sorted_vals["length"].cumsum()
    sorted_vals["cumm_fraction"] /= sorted_vals["cumm_fraction"].iloc[-1]

    sorted_vals["bin"] = (sorted_vals.cumm_fraction // (1 / (n_bins - 1))).astype(int)

    if key is None:
        key = f'{var_name.rsplit("/", 1)[-1]}_qbins_{n_bins}'

    dataset[key] = xr.DataArray(
        sorted_vals["bin"].loc[bin_nums].values,
        dims="locus",
    )

    logger.info("Added key: " + key)

    return dataset


def slice_regions(dataset, *regions, lazy=False):
    """
    Extract genomic regions that overlap with specified intervals.

    This function filters the dataset to include only regions that overlap
    with any of the specified genomic intervals. Intervals can be specified in
    multiple formats: "chr:start-end", "chr" (entire chromosome), or a comma-separated
    list of such specifications.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset containing genomic regions
    regions : str
        Region specification(s) in formats:
        - "chr:start-end" (e.g., "chr1:1000-2000")
        - "chr" (entire chromosome, e.g., "chr1")
        - List of any of the above
    lazy : bool, default=False
        Whether to return a lazy slicer instead of materializing the data

    Returns
    -------
    xr.Dataset or LazySlicer
        Filtered dataset containing only overlapping regions

    Raises
    ------
    ValueError
        If no regions match the specified query intervals
    """
    lazy = lazy or not "X" in dataset.data_vars

    parsed_regions = list(map(parse_region, regions))

    # Create mask for regions overlapping with any of the parsed regions
    ds_regions = dataset.sections["Regions"]
    regions_mask = np.zeros(len(ds_regions.chrom), dtype=bool)

    for chrom, start, end in parsed_regions:
        chrom_mask = ds_regions.chrom.values == chrom
        if np.any(chrom_mask):
            interval_mask = IntervalIndex.from_arrays(
                ds_regions.start.values[chrom_mask], ds_regions.end.values[chrom_mask]
            ).overlaps(Interval(start, end))

            # Update the overall mask
            regions_mask[chrom_mask] |= interval_mask

    if not np.any(regions_mask):
        raise ValueError(f"No regions match the specified query: {regions}")

    logger.info(
        f"Found {np.sum(regions_mask)}/{len(regions_mask)} regions matching query."
    )

    if lazy:
        return LazySlicer(dataset, locus=regions_mask)

    return dataset.isel(locus=regions_mask)


def annot_empirical_marginal(dataset, key="empirical_marginal"):
    """
    Calculate and add empirical marginal mutation rates to a dataset.

    This method computes empirical marginal mutation rates by aggregating observed mutations
    across all samples in the dataset and normalizing by context frequencies and region lengths.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing mutation data to analyze
    key : str, default="empirical_marginal"
        Base name for the mutation rate variables to be added to the dataset.
        Creates two variables: `{key}` and `{key}_locus`

    Returns
    -------
    xarray.Dataset
        The input dataset with empirical marginal rates added as new variables:
        - {key}: Marginal mutation rates normalized by context frequencies
        - {key}_locus: Per-locus marginal rates normalized by region length
    """
    todense = lambda x: x.asdense() if x.is_sparse() else x
    coo_or_dense = lambda x: x.ascoo() if x.is_sparse() else x

    X_emp = reduce(
        lambda x, y: x + y,
        (
            coo_or_dense(dataset.fetch_sample(sample_name).X)
            for sample_name in tqdm(
                dataset.list_samples()[1:],
                desc="Reducing samples",
            )
        ),
        todense(dataset.fetch_sample(dataset.list_samples()[0]).X),
    )

    X_emp = todense(X_emp)

    logger.info(f'Added key: "{key}"')
    dataset[key] = (X_emp / dataset.sections["Regions"].context_frequencies).fillna(0.0)

    locus_key = f"{key}_locus"
    logger.info(f'Added key: "{locus_key}"')
    dataset[locus_key] = (
        (
            X_emp.sum(dim=dims_except_for(X_emp.dims, "locus"))
            / dataset.sections["Regions"].length
        )
        .fillna(0.0)
        .astype(np.float32)
    )

    return dataset


def make_mixture_dataset(**datasets):
    """
    Create a mixed dataset by combining multiple source datasets.

    This function merges multiple datasets, renaming their features and state
    variables to include source identifiers, enabling comparative analysis
    across different data sources.

    Parameters
    ----------
    **datasets : dict
        Named datasets to combine. Keys become source identifiers.

    Returns
    -------
    CorpusInterface
        Combined dataset with source-specific feature namespaces
    """

    source_names = list(datasets.keys())
    merge_dsets = []
    for source_name, dataset in datasets.items():

        rename_map = {
            old_name: f"{level}/{source_name}/{os.path.basename(old_name)}"
            for level in ["Features", "State"]
            for old_name in dataset.sections.groups[level]
        }

        merge_dsets.append(dataset[rename_map.keys()].rename(rename_map))

    first_dataset = list(datasets.values())[0]

    transfer_vars = [
        var_name
        for level, vars in first_dataset.sections.groups.items()
        if not level in ["Features", "State"]
        for var_name in vars
    ]
    merge_dsets.append(first_dataset[transfer_vars])

    merged = xr.merge(merge_dsets)
    merged["source"] = xr.DataArray(
        source_names,
        dims=("source",),
    )
    merged = merged.set_coords("source")

    return CorpusInterface(merged)


def dims_except_for(dims, *keepdims):
    return tuple({*dims}.difference({*keepdims}))


def match_dims(X, **dim_sizes):
    return X.expand_dims(
        {d: dim_sizes[d] for d in dims_except_for(dim_sizes.keys(), *X.dims)}
    )


def get_regions_filename(dataset):

    return os.path.join(
        os.path.dirname(dataset.attrs["filename"]), dataset.attrs["regions_file"]
    )


def unstack_regions(dataset):
    """
    Unstack regions from a compressed format to full coordinate arrays.

    This function expands region data from a compact representation to
    full coordinate arrays, using external region file information to
    reconstruct chromosome, start, and end coordinates.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with stacked region representation

    Returns
    -------
    xr.Dataset
        Dataset with unstacked region coordinates
    """
    n_regions = dataset.coords["locus"].size

    chrom, start, end, idx = _unstack_regions(
        dataset.coords["locus"].values,
        get_regions_filename(dataset),
        n_regions,
    )

    return (
        dataset.drop_vars(dataset.sections.groups["Regions"])
        .isel(locus=idx)
        .update(
            {
                "Regions/chrom": xr.DataArray(chrom, dims=("locus",)),
                "Regions/start": xr.DataArray(start, dims=("locus",)),
                "Regions/end": xr.DataArray(end, dims=("locus",)),
            }
        )
    )


def fetch_features(
    dataset: xr.Dataset,
    *feature_names: Union[str, List[str], tuple, set],
    source: Union[str, None] = None,
):
    """
    Extract numerical features from a dataset.

    This function selects numerical features from the dataset's 'Features' section
    based on the provided feature names and optional source filter.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset containing the features in its 'Features' section.
    feature_names : Union[str, List[str], tuple, set]
        Names of features to extract. If empty, all numerical features are returned.
        Feature names can be full paths or just the basename.
    source : Union[str, None], optional
        If provided, only features from this source directory will be returned.
        Default is None, which includes features from all sources.

    Returns
    -------
    xr.DataArray
        A DataArray containing the selected features with the following structure:
        - Values: Feature values arranged as a matrix
        - Dimensions: 'feature' and 'locus'
        - Coordinates:
            - 'locus': locus values from the dataset
            - 'feature': full feature paths
            - 'feature_name': basename of each feature
            - 'source': directory name of each feature

    Notes
    -----
    Features are filtered to include only those with numerical data types,
    since these can be stuck together in an xarray DataArray.
    """

    fnames = [
        name
        for name, arr in dataset.sections["Features"].items()
        if (
            np.issubdtype(arr.dtype, np.number)
            and (
                name in feature_names
                or os.path.basename(name) in feature_names
                or len(feature_names) == 0
            )
            and source is None
            or os.path.dirname(name) == source
        )
    ]

    features = [os.path.basename(name) for name in fnames]
    sources = [os.path.dirname(name) for name in fnames]

    feature_matrix = xr.DataArray(
        np.vstack([dataset.sections["Features"][name].values for name in fnames]),
        dims=("feature", "locus"),
        coords={
            "locus": dataset.coords["locus"].values,
            "feature": fnames,
            "feature_name": ("feature", features),
            "source": ("feature", sources),
        },
        name="Features",
    )

    return feature_matrix.squeeze()
