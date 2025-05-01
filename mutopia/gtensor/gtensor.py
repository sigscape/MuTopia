import xarray as xr
import numpy as np
from typing import Union, List
from numpy.typing import NDArray
from pandas import MultiIndex, IntervalIndex, Index, Interval, DataFrame
from ..utils import FeatureType, logger, str_wrapped_list
from collections import defaultdict
from functools import reduce
from tqdm import tqdm


def GTensor(
    modality,
    *,
    name: str,
    chrom: List[str],
    start: List[int],
    end: List[int],
    context_frequencies: xr.DataArray,
    exposures: Union[None, NDArray[np.number]] = None,
):

    locus_coords = (Index(name),)

    shared_coords = {
        **modality.coords,
        "locus": locus_coords,
        "sample": [],
    }

    region_lengths = np.sum(
        context_frequencies.data, axis=tuple(range(context_frequencies.data.ndim - 1))
    )

    if exposures is None:
        exposures = np.ones(len(locus_coords), dtype=np.float64)

    return xr.Dataset(
        {
            "Regions/context_frequencies": context_frequencies,
            "Regions/length": xr.DataArray(region_lengths, dims=("locus")),
            "Regions/exposures": xr.DataArray(np.squeeze(exposures), dims=("locus")),
            "Regions/chrom": xr.DataArray(chrom, dims=("locus")),
            "Regions/start": xr.DataArray(start, dims=("locus")),
            "Regions/end": xr.DataArray(end, dims=("locus")),
        },
        coords=shared_coords,
        attrs={
            "name": name,
            "dtype": modality.MODE_ID,
        },
    )


def _add_feature(
    corpus,
    feature,
    group: str = "default",
    *,
    name: str,
    normalization: FeatureType,
):

    if not isinstance(feature, xr.DataArray):
        raise ValueError("feature must be an xarray.DataArray")

    check_structure(corpus)

    try:
        FeatureType(normalization)
    except ValueError:
        raise ValueError(
            f"Normalization type {normalization} not recognized. "
            f'Please use one of {", ".join(FeatureType.__members__)}'
        )

    allowed_types = FeatureType(normalization).allowed_dtypes

    if not any(np.issubdtype(feature.data.dtype, t) for t in allowed_types):
        raise ValueError(
            f'The feature {name} has dtype {feature.data.dtype} but must be one of {", ".join(map(repr, allowed_types))}.'
        )

    corpus.features[name] = xr.DataArray(
        data=np.array(arr),
        dims=("locus"),
        attrs={
            "normalization": normalization,
            "group": group,
        },
    )
    logger.info(f'Added key to features: "{name}"')

    return corpus


def _add_sample(
    corpus,
    sample: xr.DataArray,
    *,
    name: str,
):
    check_structure(corpus)
    ## input validation
    if not isinstance(sample, xr.DataArray):
        raise ValueError("sample must be an xarray.DataArray")

    if "sample" in corpus.dims:
        sample = sample.squeeze()

    required_dims = set(corpus.modality().dims).union({"locus"})

    if not set(sample.dims) == required_dims:
        raise ValueError(f"sample dims must be {required_dims}")
    ##

    sample = sample.ascoo().expand_dims({"sample": [name]}).ascsr("sample", "locus")

    root = xr.concat(
        [
            corpus.to_dataset(),
            xr.Dataset({"X": sample}, coords={"sample": [name]}),
        ],
        dim="sample",
    )

    root.X.ascsr("sample", "locus")

    corpus = DataTree(data=root, children=corpus.children)

    logger.info(f'Added sample to .X: "{name}"')

    return corpus


def get_explanation(corpus, component):

    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required to calculate SHAP values")

    if not component in corpus["SHAP_values"].shap_component.values:
        raise ValueError(
            f"The corpus does not have SHAP values for component {component}."
        )

    shap_values = corpus["SHAP_values"]

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
        corpus.state.locus_features.sel(locus=shap_df.index).sel(
            feature=[
                f"{s}:0" if f"{s}:0" in corpus.state.feature.values else s
                for s in shap_df.columns
            ]
        )
    ).values

    display_features = corpus.sections.features.assign_coords(locus=corpus.locus.data).sel(
        locus=shap_df.index
    )

    display_data = DataFrame([display_features[s].data for s in shap_df.columns]).T

    expl = shap.Explanation(
        shap_df.values,
        feature_names=shap_df.columns,
        data=data,
        display_data=display_data,
    )

    return expl


def equal_size_quantiles(corpus, var_name, n_bins=10):

    bin_nums = np.arange(corpus.dims["locus"])
    sorted_vals = DataFrame(
        {
            "length": corpus.sections.regions.length.values,
            "value": corpus[var_name].values,
        },
        index=bin_nums,
    )
    sorted_vals = sorted_vals.sort_values(by="value", ascending=True)

    sorted_vals["cumm_fraction"] = sorted_vals["length"].cumsum()
    sorted_vals["cumm_fraction"] /= sorted_vals["cumm_fraction"].iloc[-1]

    sorted_vals["bin"] = (sorted_vals.cumm_fraction // (1 / (n_bins - 1))).astype(int)

    key = f'{var_name.rsplit("/", 1)[-1]}_qbins_{n_bins}'
    corpus[key] = xr.DataArray(
        sorted_vals["bin"].loc[bin_nums].values,
        dims="locus",
    )

    logger.info("Added key: " + key)

    return corpus


def slice_regions(dataset, chrom: str, start: int, end: int):

    check_structure(dataset)

    regions = dataset.sections.regions
    regions_mask = (regions.chrom == chrom) & (
        IntervalIndex.from_arrays(regions.start, regions.end).overlaps(
            Interval(start, end)
        )
    )

    if not np.any(regions_mask):
        raise ValueError("No regions match query")

    logger.info(
        f"Found {np.sum(regions_mask)}/{len(regions_mask)} regions matching query."
    )

    return dataset.isel(locus=regions_mask)


def annot_empirical_marginal(corpus):

    check_structure(corpus)

    X_emp = reduce(
        lambda x, y: x + y,
        (
            corpus.fetch_sample(sample_name).ascoo()
            for sample_name in tqdm(
                corpus.list_samples()[1:],
                desc="Reducing samples",
            )
        ),
        corpus.fetch_sample(corpus.list_samples()[0]).asdense(),
    )

    X_emp = X_emp.asdense() if X_emp.is_sparse() else X_emp

    logger.info('Added key: "empirical_marginal"')
    corpus["empirical_marginal"] = (
        X_emp / corpus.sections.regions.context_frequencies
    ).fillna(0.0)

    logger.info('Added key: "empirical_locus_marginal"')
    corpus["empirical_locus_marginal"] = (
        (
            X_emp.sum(dim=dims_except_for(X_emp.dims, "locus"))
            / corpus.sections.regions.length
        )
        .fillna(0.0)
        .astype(np.float32)
    )

    return corpus


def dims_except_for(dims, *keepdims):
    return tuple({*dims}.difference({*keepdims}))


def match_dims(X, **dim_sizes):
    return X.expand_dims(
        {d: dim_sizes[d] for d in dims_except_for(dim_sizes.keys(), *X.dims)}
    )


def check_dims(corpus, model_state):
    rm_dim = dims_except_for(
        corpus.X.dims,
        *model_state.requires_dims,
    )
    if not len(rm_dim) == 0:
        logger.warning(
            f'The corpus {corpus.attrs["name"]} has extra dimensions: {", ".join(rm_dim)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
            "Having extra data dimensions will increase training time and memory usage,\n"
            'remove them by summing over them: `corpus.sum(dim="extra_dim", keep_attrs=True)`.'
        )

    missing_dims = set(model_state.requires_dims).difference(
        corpus.X.dims + ("sample",)
    )
    if not len(missing_dims) == 0:
        raise ValueError(
            f'The corpus {corpus.attrs["name"]} is missing the following dimensions: {", ".join(missing_dims)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
        )


def check_structure(corpus):

    required_vars = {
        "chrom",
        "start",
        "end",
        "length",
        "context_frequencies",
        "exposures",
    }

    if not "name" in corpus.attrs:
        raise ValueError("The corpus is missing a name attribute.")

    sections = corpus.sections.names

    if not "Regions" in sections:
        raise ValueError('The corpus is missing the "Regions" section.')

    if not "Features" in sections or len(corpus.sections.features.data_vars) == 0:
        raise ValueError(
            'The corpus is missing the "Features" section, or it is empty.'
        )

    for key in required_vars:
        if not hasattr(corpus, "Regions/" + key):
            raise ValueError(f'The corpus is missing the "{key}" node.')


def check_sample_data(corpus, dtype):
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


def check_corpus(corpus):

    check_structure(corpus)

    for feature in corpus.sections.features.values():
        check_feature(feature)


def check_feature_consistency(*corpuses):

    type_dict = defaultdict(set)
    for feature_name, dtype in list(
        (feature_name, FeatureType(feature.attrs["normalization"]))
        for corpus in corpuses
        for feature_name, feature in corpus.sections.features.data_vars.items()
    ):
        type_dict[feature_name].add(dtype)

    for feature_name, types in type_dict.items():
        if not len(types) == 1:
            raise ValueError(
                f"The feature {feature_name} has inconsistent normalization types across corpuses: {str_wrapped_list(types)}"
            )

    def _get_classes(corpus, feature):
        try:
            return feature.attrs["classes"]
        except KeyError as err:
            raise KeyError(
                f'The feature {feature.name} in corpus {corpus.attrs["name"]} is missing the `classes` attribute.'
            ) from err

    priority_dict = defaultdict(list)
    for feature_name, classes in list(
        (feature_name, tuple(_get_classes(corpus, feature)))
        for corpus in corpuses
        for feature_name, feature in corpus.sections.features.data_vars.items()
        if FeatureType(feature.attrs["normalization"])
        in (FeatureType.CATEGORICAL, FeatureType.MESOSCALE)
    ):
        priority_dict[feature_name].append(classes)

    for feature_name, priorities in priority_dict.items():
        if not all(p == priorities[0] for p in priorities):
            raise ValueError(
                f"The feature {feature_name} has inconsistent class priorities across corpuses:\n\t"
                + "\n\t".join(map(lambda p: ", ".join(map(str, p)), priorities))
            )

    corpus_membership = {
        corpus.attrs["name"]: {
            feature_name for feature_name in corpus.sections.features.data_vars.keys()
        }
        for corpus in corpuses
    }
    shared_features = set.intersection(*corpus_membership.values())

    for corpus_name, features in corpus_membership.items():
        extra_features = features.difference(shared_features)
        if len(extra_features) > 0:
            logger.warning(
                f'The corpus {corpus_name} has extra features: {", ".join(extra_features)}.\n'
                "Extra features will be ignored during training."
            )


def prepare_data(corpus):
    
    corpus['Regions/context_frequencies'] = corpus['Regions/context_frequencies']\
        .transpose(..., 'context', 'locus')
    
    corpus['Regions/context_frequencies'].data = np.asfortranarray(
        corpus['Regions/context_frequencies'].data,
        dtype=np.float32,
    )

    corpus['Regions/exposures'].data = np.ascontiguousarray(
        corpus['Regions/exposures'],
        dtype=np.float32,
    )

    return corpus
