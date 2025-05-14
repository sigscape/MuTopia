import xarray as xr
import numpy as np
from typing import Union, List
from numpy.typing import NDArray
from pandas import IntervalIndex, Index, Interval, DataFrame
from ..utils import FeatureType, logger, str_wrapped_list
from collections import defaultdict
from functools import reduce, partial
from tqdm import tqdm
from .interfaces import *
import mutopia.gtensor.disk_interface as disk


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

    if not hasattr(data, "X"):
        data = LazySampleLoader(data)

    pbar = partial(tqdm, data.list_samples(), desc="Processing samples")

    return xr.concat(
        [
            func(data.fetch_sample(sample_name))
            for sample_name in (data.list_samples() if not bar else pbar)
        ],
        dim="sample",
    )


def inplace(func):
    """
    Decorator function to modify a dataset in place - allows one to run mutations on the dataset
    without messing up the interface chains.
    """

    def wrapper(dataset, *args, **kwargs):
        original = CorpusInterface(dataset)
        result = func(original.corpus, *args, **kwargs)
        original.corpus = result
        return original._corpus

    return wrapper


def mutate(dataset, func):
    inner = CorpusInterface(dataset)
    inner.corpus = CorpusInterface(func(dataset)).corpus
    return inner._corpus


def load_dataset(dataset, with_samples=True, with_state=True):
    """
    Loads a dataset from disk. If the dataset is not present,
    it will be created.
    """
    return (LazySampleLoader if not with_samples else CorpusInterface)(
        disk.load_dataset(dataset, with_samples=with_samples, with_state=with_state)
    )


def train_test_split(dataset, *test_chroms: Union[str, List[str]], lazy=False):

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

        drop_vars = dataset.sections.groups["Features"] + dataset.sections.groups["Regions"]
        train._base_corpus.corpus = train._base_corpus.drop_vars(drop_vars)
        #test._base_corpus.corpus = test._base_corpus.drop_vars(drop_vars)

        return train, test
    
    else:
        train = dataset.isel(locus=~test_mask)
        test = dataset.isel(locus=test_mask)

    return train, test


def lazy_load(dataset):
    return load_dataset(dataset, with_samples=False, with_state=False)

def eager_load(dataset):
    return load_dataset(dataset, with_samples=True, with_state=False)

def lazy_train_test_load(dataset, *test_chroms):
    return train_test_split(lazy_load(dataset), *test_chroms, lazy=True)
    
def eager_train_test_load(dataset, *test_chroms):
    return train_test_split(eager_load(dataset), *test_chroms, lazy=False)



def get_explanation(dataset, component):

    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required to calculate SHAP values")

    if not component in dataset["SHAP_values"].shap_component.values:
        raise ValueError(
            f"The dataset does not have SHAP values for component {component}."
        )

    shap_values = dataset["SHAP_values"]

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
        dataset.state.locus_features.sel(locus=shap_df.index).sel(
            feature=[
                f"{s}:0" if f"{s}:0" in dataset.state.feature.values else s
                for s in shap_df.columns
            ]
        )
    ).values

    display_features = (
        dataset.sections["Features"]
        .assign_coords(locus=dataset.locus.data)
        .sel(locus=shap_df.index)
    )

    display_data = DataFrame([display_features[s].data for s in shap_df.columns]).T

    expl = shap.Explanation(
        shap_df.values,
        feature_names=shap_df.columns,
        data=data,
        display_data=display_data,
    )

    return expl


def equal_size_quantiles(dataset, var_name, n_bins=10):

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

    key = f'{var_name.rsplit("/", 1)[-1]}_qbins_{n_bins}'
    dataset[key] = xr.DataArray(
        sorted_vals["bin"].loc[bin_nums].values,
        dims="locus",
    )

    logger.info("Added key: " + key)

    return dataset


def slice_regions(dataset, chrom: str, start: int, end: int, lazy=False):

    check_structure(dataset)

    regions = dataset.sections["Regions"]
    regions_mask = (regions.chrom.values == chrom) & (
        IntervalIndex.from_arrays(regions.start.values, regions.end.values).overlaps(
            Interval(start, end)
        )
    )

    if not np.any(regions_mask):
        raise ValueError("No regions match query")

    logger.info(
        f"Found {np.sum(regions_mask)}/{len(regions_mask)} regions matching query."
    )

    if lazy:
        return LazySlicer(dataset, locus=regions_mask)

    return dataset.isel(locus=regions_mask)


def annot_empirical_marginal(dataset):

    check_structure(dataset)

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

    logger.info('Added key: "empirical_marginal"')
    dataset["empirical_marginal"] = (
        X_emp / dataset.sections["Regions"].context_frequencies
    ).fillna(0.0)

    logger.info('Added key: "empirical_locus_marginal"')
    dataset["empirical_locus_marginal"] = (
        (
            X_emp.sum(dim=dims_except_for(X_emp.dims, "locus"))
            / dataset.sections["Regions"].length
        )
        .fillna(0.0)
        .astype(np.float32)
    )

    return dataset


def make_mixture_dataset(**datasets):

    source_names = list(datasets.keys())
    merge_dsets = []
    for source_name, dataset in datasets.items():
        
        rename_map = {
            old_name : f'{level}/{source_name}/{os.path.basename(old_name)}'
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

    return CorpusInterface(merged)


def dims_except_for(dims, *keepdims):
    return tuple({*dims}.difference({*keepdims}))


def match_dims(X, **dim_sizes):
    return X.expand_dims(
        {d: dim_sizes[d] for d in dims_except_for(dim_sizes.keys(), *X.dims)}
    )


def check_dims(dataset, model_state):
    rm_dim = dims_except_for(
        dataset.X.dims,
        *model_state.requires_dims,
    )
    if not len(rm_dim) == 0:
        logger.warning(
            f'The dataset {dataset.attrs["name"]} has extra dimensions: {", ".join(rm_dim)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
            "Having extra data dimensions will increase training time and memory usage,\n"
            'remove them by summing over them: `dataset.sum(dim="extra_dim", keep_attrs=True)`.'
        )

    missing_dims = set(model_state.requires_dims).difference(
        dataset.X.dims + ("sample",)
    )
    if not len(missing_dims) == 0:
        raise ValueError(
            f'The dataset {dataset.attrs["name"]} is missing the following dimensions: {", ".join(missing_dims)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
        )


def check_structure(dataset):

    required_vars = {
        "chrom",
        "start",
        "end",
        "length",
        "context_frequencies",
        "exposures",
    }

    if not "name" in dataset.attrs:
        raise ValueError("The dataset is missing a name attribute.")

    sections = dataset.sections.names

    if not "Regions" in sections:
        raise ValueError('The dataset is missing the "Regions" section.')

    if not "Features" in sections or len(dataset.sections["Features"].data_vars) == 0:
        raise ValueError(
            'The dataset is missing the "Features" section, or it is empty.'
        )

    for key in required_vars:
        if not hasattr(dataset, "Regions/" + key):
            raise ValueError(f'The dataset is missing the "{key}" node.')


def check_sample_data(dataset, dtype):
    pass


def check_feature(feature):

    if not "normalization" in feature.attrs:
        raise ValueError("The feature is missing a normalization attribute.")

    normalization = feature.attrs["normalization"]
    try:
        FeatureType(normalization)
    except ValueError:
        raise ValueError(
            f"Normalization type {normalization} not recognized. "
            f'Please use one of {", ".join(FeatureType.__members__)}'
        )

    allowed_types = FeatureType(normalization).allowed_dtypes
    dtype = feature.data.dtype

    if not dtype in allowed_types and not any(
        np.issubdtype(dtype, t) for t in allowed_types
    ):
        raise ValueError(
            f'The feature {feature.name} has dtype {dtype} but must be one of {", ".join(map(repr, allowed_types))}.'
        )


def check_corpus(dataset):

    check_structure(dataset)

    for feature in dataset.sections["Features"].values():
        check_feature(feature)


def check_feature_consistency(*datasets):

    type_dict = defaultdict(set)
    for feature_name, dtype in list(
        (feature_name, FeatureType(feature.attrs["normalization"]))
        for dataset in datasets
        for feature_name, feature in dataset.sections["Features"].data_vars.items()
    ):
        type_dict[feature_name].add(dtype)

    for feature_name, types in type_dict.items():
        if not len(types) == 1:
            raise ValueError(
                f"The feature {feature_name} has inconsistent normalization types across datasets: {str_wrapped_list(types)}"
            )

    def _get_classes(dataset, feature):
        try:
            return feature.attrs["classes"]
        except KeyError as err:
            raise KeyError(
                f'The feature {feature.name} in dataset {dataset.attrs["name"]} is missing the `classes` attribute.'
            ) from err

    priority_dict = defaultdict(list)
    for feature_name, classes in list(
        (feature_name, tuple(_get_classes(dataset, feature)))
        for dataset in datasets
        for feature_name, feature in dataset.sections["Features"].data_vars.items()
        if FeatureType(feature.attrs["normalization"])
        in (FeatureType.CATEGORICAL, FeatureType.MESOSCALE)
    ):
        priority_dict[feature_name].append(classes)

    for feature_name, priorities in priority_dict.items():
        if not all(p == priorities[0] for p in priorities):
            raise ValueError(
                f"The feature {feature_name} has inconsistent class priorities across datasets:\n\t"
                + "\n\t".join(map(lambda p: ", ".join(map(str, p)), priorities))
            )

    corpus_membership = {
        dataset.attrs["name"]: {
            feature_name
            for feature_name in dataset.sections["Features"].data_vars.keys()
        }
        for dataset in datasets
    }
    shared_features = set.intersection(*corpus_membership.values())

    for corpus_name, features in corpus_membership.items():
        extra_features = features.difference(shared_features)
        if len(extra_features) > 0:
            logger.warning(
                f'The dataset {corpus_name} has extra features: {", ".join(extra_features)}.\n'
                "Extra features will be ignored during training."
            )


def prepare_data(dataset):

    dataset["Regions/context_frequencies"] = dataset[
        "Regions/context_frequencies"
    ].transpose(..., "locus")

    dataset["Regions/context_frequencies"].data = np.asfortranarray(
        dataset["Regions/context_frequencies"].data,
        dtype=np.float32,
    )

    dataset["Regions/exposures"].data = np.ascontiguousarray(
        dataset["Regions/exposures"],
        dtype=np.float32,
    )

    return dataset
