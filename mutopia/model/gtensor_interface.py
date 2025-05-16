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


class GtensorInterface:

    @classmethod
    def expand_datasets(cls, *datasets):
        for dataset in datasets:
            yield cls.get_name(dataset), dataset

    @classmethod
    def to_datasets(cls, *datasets):
        return [ds for _, ds in cls.expand_datasets(*datasets)]

    @classmethod
    def observation_dims(cls, dataset):
        return tuple(d for d in dataset.X.dims if not d == "sample")

    @classmethod
    def init_state(
        cls,
        dataset,
        factor_model,
        locals_model,
    ):
        dataset = CorpusInterface(dataset)

        if cls.has_corpusstate(dataset):
            dataset.corpus = dataset.corpus.drop_vars(dataset.sections.groups["State"])
            if "component" in dataset.dims:
                dataset.corpus = dataset.corpus.drop_dims("component")
            if "feature" in dataset.dims:
                dataset.corpus = dataset.corpus.drop_dims("feature")

        sample_names = dataset.list_samples()
        n_components = factor_model.n_components
        genome_size = dataset.sections["Regions"].context_frequencies.sum().data.item()

        state_elements = {
            "normalizers": xr.DataArray(
                np.zeros(n_components, dtype=float),
                dims=("component",),
                attrs={"genome_size": genome_size},
            ),
        }

        state_elements.update(locals_model.prepare_corpusstate(dataset))

        for model in factor_model.models.values():
            state_elements.update(model.prepare_corpusstate(dataset))

        state_elements = {"State/" + k: v for k, v in state_elements.items()}

        dataset.corpus = (
            dataset.corpus.drop_dims("component", errors="ignore")
            .assign_coords(sample=sample_names)
            .assign(**state_elements)
        )

        return dataset

    @classmethod
    def update_state(
        cls,
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

    @classmethod
    def get_dims(cls, dataset):
        return dataset.sizes

    @classmethod
    def get_regions(cls, dataset):
        return dataset.sections["Regions"]

    @classmethod
    def get_exposures(cls, dataset):
        return dataset["Regions/exposures"]

    @classmethod
    def get_freqs(cls, dataset):
        return dataset["Regions/context_frequencies"]

    @classmethod
    def get_features(cls, dataset):
        return dataset.sections["Features"]

    @classmethod
    def list_samples(cls, dataset):
        return list(dataset.list_samples())

    @classmethod
    def iter_samples(cls, dataset):
        return zip(cls.list_samples(dataset), dataset.iter_samples())

    @classmethod
    def has_corpusstate(cls, dataset):
        return "State" in dataset.sections

    @classmethod
    def is_marginal_corpus(cls, dataset):
        return len(cls.list_samples(dataset)) <= 1

    @classmethod
    def fetch_val(cls, dataset, key):
        return dataset["State/" + key]

    @classmethod
    def fetch_normalizers(cls, dataset):
        return cls.fetch_val(dataset, "normalizers").data

    @classmethod
    def update_normalizers(cls, dataset, normalizers):
        cls.fetch_normalizers(dataset)[:] = normalizers
        return dataset

    @classmethod
    def genome_size(cls, dataset):
        return cls.fetch_val(dataset, "normalizers").attrs["genome_size"]

    @classmethod
    def get_name(cls, dataset):
        return dataset.attrs["name"]

    @classmethod
    def fetch_topic_compositions(cls, dataset, sample_name):
        gamma = (
            cls.fetch_val(dataset, "topic_compositions")
            .sel(sample=sample_name)
            .transpose("component", ...)
            .data
        )

        if len(gamma.shape) > 1:
            gamma = gamma[:, 0]

        return gamma

    @classmethod
    def update_topic_compositions(cls, dataset, sample_name, gamma):
        cls.fetch_topic_compositions(dataset, sample_name)[:] = gamma
        return dataset

    @classmethod
    def using_exposures_from(cls, *corpuses):
        corpus_dict = {cls.get_name(dataset): dataset for dataset in corpuses}

        return lambda dataset, sample_name: cls.fetch_topic_compositions(
            corpus_dict[cls.get_name(dataset)], sample_name
        )
