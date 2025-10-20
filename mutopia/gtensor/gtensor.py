"""
The GTensor module for genomic tensor analysis.
This module provides functionality for creating, manipulating, and analyzing genomic tensors,
including loading datasets, applying transformations, and generating explanations for model components.

GTensors are hierarchical, multi-dimensional arrays designed to represent complex genomic data structures.
They are sliceable along multiple dimensions, and support lazy loading for memory efficiency.

Use the Gtensor CLI tool to interact with and build GTensor datasets from the command line - the 
python API is mostly intended for analysis and visualization.
"""

from __future__ import annotations

import xarray as xr
#xr.set_options(use_new_combine_kwarg_defaults=True)
import pandas as pd
import numpy as np
from typing import Union, List, Any, Callable, Optional, Iterable, TYPE_CHECKING
from numpy.typing import NDArray
from functools import reduce
import os
from tqdm import tqdm
from mutopia.utils import logger, parse_region
from mutopia.genome_utils.bed12_utils import unstack_regions as _unstack_regions
import mutopia.gtensor.disk_interface as disk
from .interfaces import (
    CorpusInterface,
    LazySampleLoader,
    LocusSlice,
    SampleSlice,
)
if TYPE_CHECKING:
    import shap

GTensorDataset = Union[
    xr.Dataset,
    CorpusInterface,
    LazySampleLoader,
    LocusSlice,
    SampleSlice,
]

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
    "get_shap_summary",
    "equal_size_quantiles",
    "slice_regions",
    "slice_samples",
    "annot_empirical_marginal",
    "make_mixture_dataset",
    "match_dims",
    "dims_except_for",
    "unstack_regions",
    "mutate_method",
    "BED_COLS",
    "list_components",
    "fetch_component",
    "fetch_interactions",
    "fetch_shared_effects",
    "rename_components",
    "excel_report",
    "infer_source_celltypes",
]

BED_COLS = [
    "Regions/chrom",
    "Regions/start",
    "Regions/end",
]

def GTensor(
    modality: Any,
    *,
    name: str,
    chrom: List[str],
    start: List[int],
    end: List[int],
    context_frequencies: xr.DataArray,
    exposures: Union[None, NDArray[np.number]] = None,
    dtype: Any = None,
) -> GTensorDataset:
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

    locus_coords = pd.Index(np.arange(len(chrom)))

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

def infer_source_celltypes(dataset: GTensorDataset) -> GTensorDataset:
    """
    Infer source cell types from feature names and assign to dataset coordinates.

    This function examines the feature names in the dataset to identify unique
    source cell types based on directory structure. It then assigns these
    inferred cell types to the 'source' coordinate of the dataset.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset containing features with potential source information

    Returns
    -------
    GTensorDataset
        Dataset with 'source' coordinate added, reflecting inferred cell types

    Raises
    ------
    ValueError
        If no features are found in the dataset to infer sources from
    """
    return disk.infer_source_celltypes(dataset)


def apply_to_samples(data: GTensorDataset, func: Callable, bar: bool = True) -> GTensorDataset:
    """
    Apply a function to each sample in a dataset with parallel processing.

    This function applies a given function to each sample (region) in the dataset,
    handling the parallelization and aggregation of results. It's designed for
    operations that need to process each genomic region independently.

    Parameters
    ----------
    data : GTensorDataset
        Input dataset or data loader containing samples to process
    func : callable
        Function to apply to each sample. Should accept a dataset slice and
        return a result that can be concatenated
    bar : bool, default=True
        Whether to display a progress bar during processing

    Returns
    -------
    GTensorDataset
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


def mutate(func: Callable) -> Callable:
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


def mutate_method(func: Callable) -> Callable:
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


def load_dataset(
    dataset: Union[str, os.PathLike],
    with_samples: bool = True,
    with_state: bool = True,
) -> GTensorDataset:
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
    GTensorDataset
        Loaded dataset interface. Returns LazySampleLoader if with_samples=False,
        otherwise returns CorpusInterface
    """
    return (LazySampleLoader if not with_samples else CorpusInterface)(
        disk.load_dataset(dataset, with_samples=with_samples, with_state=with_state)
    )


def train_test_split(
    dataset: GTensorDataset, *test_chroms: Union[str, List[str]], lazy: bool = False
) -> tuple[GTensorDataset, GTensorDataset]:
    """
    Split a dataset into training and testing sets based on chromosomes.

    This function splits the dataset by chromosomes, with specified chromosomes
    reserved for testing and the remainder used for training. The split can be
    performed eagerly (loading all data) or lazily (for memory efficiency).

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset to split
    *test_chroms : Union[str, List[str]]
        Chromosome names to reserve for the test set. Can be provided as
        multiple string arguments or lists of strings
    lazy : bool, default=False
        Whether to perform lazy splitting. If True, returns LazySlicer objects
        that don't load data until accessed

    Returns
    -------
    tuple[GTensorDataset, GTensorDataset]
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
        train = LocusSlice(dataset, locus=~test_mask)
        test = LocusSlice(dataset, locus=test_mask)

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


def lazy_load(dataset: Union[str, os.PathLike]) -> GTensorDataset:
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
    GTensorDataset
        Lazy dataset interface that loads data on demand
    """
    return load_dataset(dataset, with_samples=False, with_state=False)


def eager_load(dataset: Union[str, os.PathLike]) -> GTensorDataset:
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
    GTensorDataset
        Eager dataset interface with samples loaded into memory
    """
    return load_dataset(dataset, with_samples=True, with_state=False)


def lazy_train_test_load(
    dataset: Union[str, os.PathLike], *test_chroms: str
) -> tuple[GTensorDataset, GTensorDataset]:
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


def eager_train_test_load(
    dataset: Union[str, os.PathLike], *test_chroms: str
) -> tuple[GTensorDataset, GTensorDataset]:
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


def num_sources(dataset: GTensorDataset) -> int:
    """
    Get the number of distinct sources in a dataset.

    This function counts the number of unique sources present in the dataset's
    'source' coordinate, which is useful for determining if the dataset contains
    data from multiple cell types or conditions.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset to query for sources

    Returns
    -------
    int
        Number of distinct sources in the dataset. Returns 0 if no sources
        are defined.
    """
    return len(list_sources(dataset))


def is_mixture_dataset(dataset: GTensorDataset) -> bool:
    """
    Check if a dataset contains data from multiple sources.

    This function determines whether the dataset is a mixture dataset by checking
    if it contains more than one source. Mixture datasets have source-specific
    features and require special handling for analysis.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset to check

    Returns
    -------
    bool
        True if the dataset contains multiple sources, False otherwise
    """
    return num_sources(dataset) > 1


def list_sources(dataset: GTensorDataset) -> List[str]:
    """
    List all source identifiers in the dataset.

    This function extracts and returns the names of all sources present in the
    dataset. Sources typically represent different cell types, tissues, or
    experimental conditions.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset containing source information

    Returns
    -------
    List[str]
        List of source names. Returns an empty list if the dataset has no
        'source' coordinate defined.
    """
    if "source" in dataset.coords:
        return dataset.coords["source"].values.tolist()
    return []


def fetch_source(dataset: GTensorDataset, source: str) -> GTensorDataset:
    """
    Extract and restructure data for a specific source from a multi-source dataset.

    This function filters and reorganizes a dataset to contain only data relevant to a 
    specified source, while maintaining shared features and state variables that are 
    common across all sources.

    Parameters
    ----------
    dataset : GTensorDataset
        The input dataset containing data from multiple sources, organized with 
        hierarchical variable names (e.g., "Features/source/variable", "State/source/variable").
    source : str
        The name of the source to extract data for. Must be present in the dataset.

    Returns
    -------
    GTensorDataset
        A new dataset containing:
        - Source-specific features and state variables (with paths flattened)
        - Shared features and state variables (common to all sources)
        - Other data variables from the original dataset
        - Updated name attribute reflecting the source
        - Source dimension removed if present

    Raises
    ------
    ValueError
        If the specified source is not found in the dataset.

    Notes
    -----
    The function performs the following transformations:
    1. Validates that the source exists in the dataset
    2. Separates source-specific and shared variables from Features and State groups
    3. Creates a rename mapping to flatten source-specific variable paths
    4. Combines source-specific, shared, and other variables into a new dataset
    5. Updates dataset attributes and coordinates while removing source dimension
    """
    sources = list_sources(dataset)
    if not source in sources:
        raise ValueError(f"Source {source} not found in dataset")

    groups = dataset.sections.groups
    state = groups.pop("State", [])
    features = groups.pop("Features", [])

    use_features = [os.path.basename(v) for v in features if v.split("/")[1] == source]
    use_state = [os.path.basename(v) for v in state if v.split("/")[1] == source]

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


def get_explanation(dataset: GTensorDataset, component: str) -> "shap.Explanation":
    """
    Generate SHAP explanations for a specific model component.

    This function extracts and formats SHAP values for interpretability analysis,
    creating an explanation object that can be used with SHAP visualization tools.

    Parameters
    ----------
    dataset : GTensorDataset
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
            .to_dataframe()
            .reset_index()
            .rename(
                columns={
                    "shap_component": "component",
                    locus_dim: "locus",
                    "shap_features": "feature",
                    "SHAP_values": "value",
                }
            )
        )

        # handle this case to remove the convolution
        if any(shap_df.feature.str.contains(":")):
            shap_df[["feature", "convolution"]] = shap_df.feature.str.split(
                ":", expand=True, n=1
            ).rename(columns={0: "feature", 1: "convolution"})

        shap_df = (
            shap_df.groupby(["feature", "locus"])["value"].sum().unstack().fillna(0).T
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

        display_data = pd.DataFrame(
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

    shap_df, data, display_data = [pd.concat(x) for x in zip(*shap_data)]

    expl = shap.Explanation(
        shap_df.values,
        feature_names=shap_df.columns,
        data=data.values,
        display_data=display_data,
    )

    return expl


def get_shap_summary(data: GTensorDataset, source: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a summary of SHAP values for model components.

    This function computes summary statistics for SHAP values across all components,
    including effect sizes (97th percentile of absolute SHAP values) and correlations
    between SHAP values and feature values. This provides a high-level view of which
    features have the strongest associations with each component.

    Parameters
    ----------
    data : GTensorDataset
        Dataset containing SHAP values and feature information
    source : str, optional
        Source identifier to analyze. Required if the dataset is a mixture dataset
        with multiple sources.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - component: Component name
        - feature: Feature name
        - effect_size: 97th percentile of absolute SHAP values
        - correlation: Pearson correlation between SHAP values and feature values

    Raises
    ------
    ValueError
        If the dataset is a mixture dataset and no source is specified
    """
    
    if is_mixture_dataset(data) and source is None:
        raise ValueError("Must specify source when dataset is a mixture.")
    
    source = "State/" + source if source is not None else "State"

    shap_values = data["SHAP_values"]
    locus_dim = "locus" if "locus" in shap_values.dims else "shap_locus"

    shap_values = (
        shap_values.to_dataframe()
        .reset_index()
        .rename(
            columns={
                "shap_component": "component",
                locus_dim: "locus",
                "shap_features": "feature",
                "SHAP_values": "shap_value",
            }
        )
    )
    #shap_values["feature"] = shap_values["feature"].str.split(":", expand=True)[0]
    #hap_values = shap_values.groupby(["component", "locus", "feature"])["shap_value"].sum().reset_index()

    shap_values = shap_values.merge(
        data[f"{source}/locus_features"]
        .sel(locus=shap_values.locus.unique())
        .to_dataframe()
        .reset_index()
        .rename(columns={f"{source}/locus_features": "feature_value"}),
        on=["locus", "feature"],
        how="inner",
    )

    effect_size = (
        shap_values.groupby(["component", "feature"])["shap_value"]
        .apply(lambda x: np.quantile(np.abs(x), 0.97))
        .rename("effect_size")
    )

    def nan_corr(x, y):
        """Compute correlation, ignoring NaNs."""
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            return np.nan
        return np.corrcoef(x[mask], y[mask])[0, 1]

    correlation = (
        shap_values.groupby(["component", "feature"])[["shap_value", "feature_value"]]
        .apply(lambda x: nan_corr(x["shap_value"], x["feature_value"]))
        .rename("correlation")
    )

    component_summary = effect_size.to_frame().join(correlation).reset_index()

    return component_summary


def equal_size_quantiles(
    dataset: GTensorDataset, var_name: str, n_bins: int = 10, key: Optional[str] = None
) -> GTensorDataset:
    """
    Create equal-size quantile bins for a variable in the dataset.

    This function bins the values of a specified variable into quantiles of equal
    cumulative region length, which is useful for creating balanced genomic bins.

    Parameters
    ----------
    dataset : GTensorDataset
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
    GTensorDataset
        The input dataset with quantile bins added as a new variable
    """

    bin_nums = np.arange(dataset.sizes["locus"])
    sorted_vals = pd.DataFrame(
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


def slice_samples(dataset: GTensorDataset, samples: List[str]) -> GTensorDataset:
    """
    Extract a subset of samples from the dataset.

    This function filters the dataset to include only the specified samples,
    enabling focused analysis on particular samples of interest while maintaining
    all other dataset dimensions and attributes.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset containing multiple samples
    samples : List[str]
        List of sample names to extract from the dataset. Sample names must
        exist in the dataset's sample coordinate.

    Returns
    -------
    GTensorDataset
        Filtered dataset containing only the specified samples wrapped in a
        SampleSlice interface

    Raises
    ------
    KeyError
        If any of the specified samples are not found in the dataset

    Notes
    -----
    If an empty list is provided, the original dataset is returned unchanged.
    The function uses the mutate pattern to maintain interface chain compatibility.
    """
    d = mutate(lambda d: d.sel(sample=list(samples)) if len(samples) > 0 else d)(dataset)
    return SampleSlice(d, samples)


def slice_regions(
    dataset: GTensorDataset, *regions: str, lazy: bool = False
) -> GTensorDataset:
    """
    Extract genomic regions that overlap with specified intervals.

    This function filters the dataset to include only regions that overlap
    with any of the specified genomic intervals. Intervals can be specified in
    multiple formats: "chr:start-end", "chr" (entire chromosome), or a comma-separated
    list of such specifications.

    Parameters
    ----------
    dataset : GTensorDataset
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
    GTensorDataset
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
            interval_mask = pd.IntervalIndex.from_arrays(
                ds_regions.start.values[chrom_mask], ds_regions.end.values[chrom_mask]
            ).overlaps(pd.Interval(start, end))

            # Update the overall mask
            regions_mask[chrom_mask] |= interval_mask

    if not np.any(regions_mask):
        raise ValueError(f"No regions match the specified query: {regions}")

    logger.info(
        f"Found {np.sum(regions_mask)}/{len(regions_mask)} regions matching query."
    )

    if lazy:
        return LocusSlice(dataset, locus=regions_mask)

    return dataset.isel(locus=regions_mask)


def annot_empirical_marginal(
    dataset: GTensorDataset, key: str = "empirical_marginal"
) -> GTensorDataset:
    """
    Calculate and add empirical marginal mutation rates to a dataset.

    This method computes empirical marginal mutation rates by aggregating observed mutations
    across all samples in the dataset and normalizing by context frequencies and region lengths.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing mutation data to analyze
    key : str, default="empirical_marginal"
        Base name for the mutation rate variables to be added to the dataset.
        Creates two variables: `{key}` and `{key}_locus`

    Returns
    -------
    GTensorDataset
        The input dataset with empirical marginal rates added as new variables:
        - {key}: Marginal mutation rates normalized by context frequencies
        - {key}_locus: Per-locus marginal rates normalized by region length
    """
    todense = lambda x: x.asdense() if x.is_sparse() else x
    coo_or_dense = lambda x: x.ascoo() if x.is_sparse() else x
    reduce_samples = list(dataset.list_samples()[1:])

    X_emp = reduce(
        lambda x, y: x + y,
        (
            coo_or_dense(sample.X)
            for sample in tqdm(
                dataset.iter_samples(subset=reduce_samples),
                desc="Reducing samples",
                total=len(reduce_samples),
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


def make_mixture_dataset(**datasets: GTensorDataset) -> GTensorDataset:
    """
    Create a mixed dataset by combining multiple source datasets.

    This function merges multiple datasets, renaming their features and state
    variables to include source identifiers, enabling comparative analysis
    across different data sources.

    Parameters
    ----------
    **datasets : GTensorDataset
        Named datasets to combine. Keys become source identifiers.

    Returns
    -------
    GTensorDataset
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


def dims_except_for(dims: Iterable, *keepdims: str) -> tuple:
    return tuple({*dims}.difference({*keepdims}))


def match_dims(X: xr.DataArray, **dim_sizes: int) -> xr.DataArray:
    return X.expand_dims(
        {d: dim_sizes[d] for d in dims_except_for(dim_sizes.keys(), *X.dims)}
    )


def get_regions_filename(dataset: GTensorDataset) -> str:

    return os.path.join(
        os.path.dirname(dataset.attrs["filename"]), dataset.attrs["regions_file"]
    )


def unstack_regions(dataset: GTensorDataset) -> GTensorDataset:
    """
    Unstack regions from a compressed format to full coordinate arrays.

    This function expands region data from a compact representation to
    full coordinate arrays, using external region file information to
    reconstruct chromosome, start, and end coordinates.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset with stacked region representation

    Returns
    -------
    GTensorDataset
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
    dataset: GTensorDataset,
    *feature_names: str,
    source: Union[str, None] = None,
) -> xr.DataArray:
    """
    Extract numerical features from the dataset's "Features" section.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing feature variables under the "Features" group.
    *feature_names : str
        Glob patterns or basenames of features to select. When empty, all
        numeric features are returned.
    source : str, optional
        Restrict selection to features within this source directory. When None,
        features from all sources are considered.

    Returns
    -------
    xarray.DataArray
        A DataArray with dims ("feature", "locus") and coords "locus",
        "feature" (full paths), "feature_name" (basenames), and "source".

    Notes
    -----
    All selected features must share a compatible numeric dtype.
    """
    from fnmatch import fnmatch

    fnames = [
        name
        for name, _ in dataset.sections["Features"].items()
        if (
            any(
                fnmatch(name, pattern) or fnmatch(os.path.basename(name), pattern)
                for pattern in feature_names
            )
            or len(feature_names) == 0
        )
        and (source is None or fnmatch(os.path.dirname(name), source))
    ]

    if not fnames:
        raise ValueError("No matching features found.")

    # Check that all features have inherit from same numpy data type (and so can be concatenated without unexpected type conversions)
    dtypes = {dataset.sections["Features"][name].dtype for name in fnames}

    check_dtype = lambda _dtype : all(np.issubdtype(dtype, _dtype) for dtype in dtypes)
    if not (check_dtype(np.number) or check_dtype(np.str_)):
        raise ValueError("All features must have a numeric data type.")

    # Reorder features to match the order in feature_names
    if len(feature_names) > 0:
        fnames = sorted(
            fnames,
            key=lambda x: (
                feature_names.index(os.path.basename(x))
                if os.path.basename(x) in feature_names
                else len(feature_names)
            ),
        )

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


class ComponentWrapper:

    def __init__(self, dataset):
        if not "component" in dataset.coords:
            raise ValueError("Dataset does not contain 'component' coordinate.")

        self.dataset = dataset

    def _get_k(self, component_name):
        if isinstance(component_name, int):
            return component_name
        try:
            return list(self.dataset.coords["component"].values).index(component_name)
        except ValueError:
            raise ValueError(f"Component {component_name} not found in model.")

    @property
    def n_components(self):
        return len(self.dataset.coords["component"].values)

    @property
    def component_names(self):
        return list(self.dataset.coords["component"].values)

    def get_spectrum(self, idx: Union[str, int]) -> xr.DataArray:
        k = self._get_k(idx)
        return self.dataset["Spectra/spectra"].isel(component=k)

    def get_interactions(self, idx: Union[str, int]) -> xr.DataArray:
        k = self._get_k(idx)
        return self.dataset["Spectra/interactions"].isel(component=k)

    def get_shared_effects(self, idx: Union[str, int]) -> xr.DataArray:
        return self.dataset["Spectra/shared_effects"].isel(component=self._get_k(idx))


def rename_components(dataset: GTensorDataset, names: List[str]) -> GTensorDataset:
    """
    Rename the components of the model and update the dataset coordinates accordingly.

    Parameters
    ----------
    dataset : GTensorDataset
        The dataset containing model components to be renamed.
    names : typing.List[str]
        New names for the components. Must have the same length as the number of components in the model.

    Returns
    -------
    GTensorDataset
        The dataset with updated component names in coordinates.

    Raises
    ------
    ValueError
        If the number of provided names doesn't match the number of components.
    KeyError
        If some components in the dataset's "shap_component" coordinate don't match the model components.

    Notes
    -----
    This method also updates the internal _component_names attribute of the model.
    """
    components = ComponentWrapper(dataset)
    if not len(names) == components.n_components:
        raise ValueError("The number of names must match the number of components")

    name_map = dict(zip(components.component_names, names))
    new_coords = {"component": names}

    if "shap_component" in dataset.coords:
        try:
            new_coords["shap_component"] = [
                name_map[c] for c in dataset.coords["shap_component"].data
            ]
        except KeyError:
            raise KeyError(
                "Some components in dataset do not match the model components. Just delete the SHAP_values and try again."
            )

    dataset = dataset.mutate(lambda ds : ds.assign_coords(new_coords))
    return dataset


def _fetch_component_data(
    dataset: GTensorDataset, component_name: Union[str, int], fetch_fn
) -> xr.DataArray:
    components = ComponentWrapper(dataset)
    d = getattr(components, fetch_fn)(component_name)
    d.attrs["dtype"] = dataset.attrs["dtype"]
    return d


def list_components(dataset: GTensorDataset) -> List[str]:
    """
    List all component names in the dataset.

    This function extracts and returns the names of all model components
    (mutational signatures or processes) present in the dataset.

    Parameters
    ----------
    dataset : GTensorDataset
        Input dataset containing model components

    Returns
    -------
    List[str]
        List of component names

    Raises
    ------
    ValueError
        If the dataset does not contain a 'component' coordinate
    """
    components = ComponentWrapper(dataset)
    return components.component_names


def fetch_component(dataset: GTensorDataset, component_name: Union[str, int]) -> xr.DataArray:
    """
    Retrieve the mutational spectrum for a specific component.

    This function extracts the signature spectrum (mutational profile) for a
    specified component from the dataset. The spectrum describes the relative
    frequency of different mutation types for this component.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing component spectra
    component_name : Union[str, int]
        Name or index of the component to retrieve

    Returns
    -------
    xr.DataArray
        DataArray containing the component's mutational spectrum with
        appropriate dimensions and coordinates

    Raises
    ------
    ValueError
        If the specified component is not found in the dataset
    """
    return _fetch_component_data(dataset, component_name, "get_spectrum")


def fetch_interactions(dataset: GTensorDataset, component_name: Union[str, int]) -> xr.DataArray:
    """
    Retrieve interaction effects for a specific component.

    This function extracts the interaction matrix for a specified component,
    showing how the mutational spectrum varies across different genomic contexts
    (e.g., strand orientation, replication timing, gene regions).

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing component interaction data
    component_name : Union[str, int]
        Name or index of the component to retrieve

    Returns
    -------
    xr.DataArray
        DataArray containing the component's interaction effects with
        appropriate dimensions and coordinates

    Raises
    ------
    ValueError
        If the specified component is not found in the dataset
    """
    return _fetch_component_data(dataset, component_name, "get_interactions")


def fetch_shared_effects(dataset: GTensorDataset, component_name: Union[str, int]) -> xr.DataArray:
    """
    Retrieve shared effects for a specific component.

    This function extracts the shared effects matrix for a specified component,
    representing effects that are common across different contexts or conditions.
    Shared effects capture baseline mutational patterns that don't vary with
    genomic features.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing component shared effects data
    component_name : Union[str, int]
        Name or index of the component to retrieve

    Returns
    -------
    xr.DataArray
        DataArray containing the component's shared effects with
        appropriate dimensions and coordinates

    Raises
    ------
    ValueError
        If the specified component is not found in the dataset
    """
    return _fetch_component_data(dataset, component_name, "get_shared_effects")


def excel_report(self, dataset: GTensorDataset, output: str, normalization="global"):
    """
    Generate a comprehensive Excel report with model results.

    This method creates an Excel file containing signature data, sample contributions,
    and SHAP values (if available) across multiple worksheets.

    Parameters
    ----------
    dataset : GTensorDataset
        Dataset containing the model results to export
    output : str
        Output file path for the Excel report

    Raises
    ------
    ImportError
        If openpyxl is not installed for Excel writing support

    Notes
    -----
    The Excel file will contain the following sheets:
    - Signature_{name}: Normalized signature data for each component
    - Sample_contributions: Component contributions per sample (if available)
    - SHAP_transformed_features: SHAP feature data (if available)
    - SHAP_original_features: Original feature data for SHAP (if available)
    - SHAP_values_{component}: SHAP values for each component (if available)

    Requires openpyxl to be installed: pip install openpyxl
    """

    try:
        from pandas import ExcelWriter
    except ImportError:
        raise ImportError(
            "openpyxl is required to save excel reports, install with `pip install openpyxl`"
        )

    renorm = lambda x: x / x.sum() * 1000

    with ExcelWriter(output) as writer:

        for sig in self.component_names:
            (
                renorm(self.format_component(sig, normalization=normalization))
                .to_pandas()
                .T.to_excel(
                    writer,
                    sheet_name=f"Signature_{sig}",
                )
            )

        if hasattr(dataset, "contributions"):
            (
                dataset.contributions.stack(observations=("source", "component"))
                .transpose("sample", ...)
                .to_pandas()
                .to_excel(
                    writer,
                    sheet_name="Sample_contributions",
                )
            )

        if hasattr(dataset, "SHAP_values"):

            shap_components = dataset.SHAP_values.coords["shap_component"].values
            expl = get_explanation(dataset, shap_components[0])

            pd.DataFrame(
                expl.data,
                columns=expl.feature_names,
            ).to_excel(
                writer,
                sheet_name="SHAP_transformed_features",
                index=False,
            )

            display_data = expl.display_data.copy()
            display_data.columns = expl.feature_names
            display_data.to_excel(
                writer,
                sheet_name="SHAP_original_features",
                index=False,
            )

            for component in shap_components:

                expl = get_explanation(dataset, component)

                pd.DataFrame(
                    expl.values,
                    columns=expl.feature_names,
                ).to_excel(
                    writer,
                    sheet_name="SHAP_values_{}".format(component),
                    index=False,
                )
