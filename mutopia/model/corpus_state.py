"""
Why go through the trouble of creating these redundant functions?
This interface is meant to be used by the model, which should not depend on the underlying
structure of the datasets.
"""

import xarray as xr
from ..gtensor import *
from ..utils import parallel_gen
import numpy as np
from functools import partial


def expand_datasets(datasets):
    for dataset in datasets:
        yield get_name(dataset), dataset


@inplace
def init_state(
    dataset,
    factor_model,
    locals_model,
):

    if "State" in dataset.sections.names:
        dataset = dataset.drop_vars(dataset.sections.groups["State"])
        if "component" in dataset.dims:
            dataset = dataset.drop_dims("component")
        if "feature" in dataset.dims:
            dataset = dataset.drop_dims("feature")

    sample_names = dataset.list_samples()
    n_components = factor_model.n_components
    genome_size = dataset.sections.regions.context_frequencies.sum().data.item()

    state_elements = {
        "normalizers": xr.DataArray(
            np.zeros(n_components, dtype=float),
            dims=("component",),
            attrs={"genome_size": genome_size},
        ),
    }

    for model in factor_model.models.values():
        state_elements.update(model.prepare_corpusstate(dataset))

    state_elements.update(locals_model.prepare_corpusstate(dataset))
    state_elements = {"State/" + k: v for k, v in state_elements.items()}

    dataset = (
        dataset.drop_dims("component", errors="ignore")
        .assign_coords(sample=sample_names)
        .assign(**state_elements)
    )

    return dataset


def update_state(
    dataset,
    model_state,
    from_scratch=False,
    par_context=None,
):
    for _ in parallel_gen(
        (
            partial(model.update_corpusstate, dataset, from_scratch=from_scratch)
            for model in model_state.models.values()
        ),
        par_context,
        ordered=False,
    ):
        pass

    return dataset


def get_dims(dataset):
    return dataset.sizes


def get_regions(dataset):
    return dataset.sections.regions


def get_features(dataset):
    return dataset.sections.features


def list_samples(dataset):
    return list(dataset.list_samples())


def iter_samples(dataset):
    return zip(list_samples(dataset), dataset.iter_samples())


def has_corpusstate(dataset):
    return "State" in dataset.sections


def is_marginal_corpus(dataset):
    return len(list_samples(dataset)) <= 1


def fetch_val(dataset, key):
    return dataset["State/" + key]


def fetch_normalizers(dataset):
    return fetch_val(dataset, "normalizers").data


def update_normalizers(dataset, normalizers):
    fetch_normalizers(dataset)[:] = normalizers
    return dataset


def genome_size(dataset):
    return fetch_val(dataset, "normalizers").attrs["genome_size"]


def get_name(dataset):
    return dataset.attrs["name"]


def fetch_topic_compositions(dataset, sample_name):

    gamma = (
        fetch_val(dataset, "topic_compositions")
        .sel(sample=sample_name)
        .transpose("component", ...)
        .data
    )

    if len(gamma.shape) > 1:
        gamma = gamma[:, 0]

    return gamma


def update_topic_compositions(dataset, sample_name, gamma):
    fetch_topic_compositions(dataset, sample_name)[:] = gamma
    return dataset


def using_exposures_from(*corpuses):

    corpus_dict = {get_name(dataset): dataset for dataset in corpuses}

    return lambda dataset, sample_name: fetch_topic_compositions(
        corpus_dict[get_name(dataset)], sample_name
    )
