"""
Why go through the trouble of creating these redundant functions?
This interface is meant to be used by the model, which should not depend on the underlying
structure of the datasets.
"""

import xarray as xr
from mutopia.gtensor import *
from mutopia.utils import parallel_gen
import numpy as np
from functools import partial
from os import path


def to_datasets(*datasets):
    return [ds for _, ds in expand_datasets(*datasets)]


def expand_datasets(*datasets):
    for dataset in datasets:
        for source in list_sources(dataset):
            ds = fetch_source(dataset, source)
            yield get_name(ds), ds


def observation_dims(dataset):
    return tuple(d for d in dataset.X.dims if not d == "sample")


def get_name(dataset):
    return dataset.attrs["name"]


def is_mixture_corpus(dataset):
    return "sources" in dataset.attrs and len(dataset.attrs["sources"]) > 1


def list_sources(corpus):
    return corpus.attrs["sources"]


def fetch_source(corpus, source):

    if not source in list_sources(corpus):
        raise ValueError(f"Source {source} not found in corpus")

    groups = corpus.sections.groups
    state = groups.pop("State", [])
    features = groups.pop("Features", [])

    use_features = [path.basename(v) for v in features if v.startswith("Features/" + source)]
    use_state = [path.basename(v) for v in state if v.startswith("State/" + source)]

    rename_map = {
        os.path.join("Features", source, v): os.path.join("Features", v)
        for v in use_features
    }
    rename_map.update(
        {os.path.join("State", source, v): os.path.join("State", v) for v in use_state}
    )

    other_features = [v for g in groups.values() for v in g]

    source_corpus = corpus[list(rename_map.keys()) + other_features].rename(rename_map)
    source_corpus.attrs["sources"] = [source]
    source_corpus.attrs["name"] = get_name(corpus) + "/" + source

    return source_corpus


def sources(corpus):
    for source in list_sources(corpus):
        ds = fetch_source(corpus, source)
        yield source, ds

@inplace
def init_state(
    dataset,
    factor_model,
    locals_model,
):

    if has_corpusstate(dataset):
        dataset = dataset.drop_vars(dataset.sections.groups["State"])
        if "component" in dataset.dims:
            dataset = dataset.drop_dims("component")
        if "feature" in dataset.dims:
            dataset = dataset.drop_dims("feature")

    sample_names = dataset.list_samples()
    n_components = factor_model.n_components
    genome_size = dataset.sections["Regions"].context_frequencies.sum().data.item()

    dataset = (
        dataset.drop_dims("component", errors="ignore")
        .assign_coords(sample=sample_names)
    )

    for source, data in sources(dataset):

        state_elements = {
            "normalizers": xr.DataArray(
                np.zeros(n_components, dtype=float),
                dims=("component",),
                attrs={"genome_size": genome_size},
            ),
        }

        state_elements.update(locals_model.prepare_corpusstate(data))

        for model in factor_model.models.values():
            state_elements.update(model.prepare_corpusstate(data))

        state_elements = {f"State/{source}/{k}": v for k, v in state_elements.items()}

        dataset = dataset.assign(**state_elements)

    return dataset


def update_state(
    dataset,
    model_state,
    from_scratch=False,
    par_context=None,
):
    for _ in parallel_gen(
        (
            partial(model.update_corpusstate, ds, from_scratch=from_scratch)
            for model in model_state.models.values()
            for _, ds in sources(dataset)
        ),
        par_context,
        ordered=False,
    ):
        pass

    return dataset


def _fetch_topic_compositions(dataset, sample_name):
    gamma = (
        fetch_val(dataset, "topic_compositions")
        .sel(sample=sample_name)
        .transpose("component", ...)
        .data
    )

    if len(gamma.shape) > 1:
        gamma = gamma[:, 0]

    return gamma


def fetch_topic_compositions(dataset, sample_name):
    return np.array([
        _fetch_topic_compositions(ds, sample_name)
        for name, ds in sources(dataset)
    ])


def update_topic_compositions(dataset, sample_name, gamma):
    for (_, ds), gamma_d in zip(sources(dataset), gamma):
        _fetch_topic_compositions(ds, sample_name)[:] = gamma_d


def using_exposures_from(*corpuses):

    corpus_dict = {get_name(dataset): dataset for dataset in corpuses}

    return lambda dataset, sample_name: fetch_topic_compositions(
        corpus_dict[get_name(dataset)], sample_name
    )

##
# These are the same as the normal gtensor functions.
##
def get_dims(dataset):
    return dataset.sizes


def get_regions(dataset):
    return dataset.sections["Regions"]


def get_features(dataset):
    return dataset.sections["Features"]


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
