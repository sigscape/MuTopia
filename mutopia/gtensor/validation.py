from . import dims_except_for
from collections import defaultdict
from numpy import issubdtype
from ..utils import FeatureType, logger, str_wrapped_list


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
        issubdtype(dtype, t) for t in allowed_types
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
