"""
Why go through the trouble of creating these redundant functions?
This interface is meant to be used by the model, which should not depend on the underlying
structure of the datasets.
"""

import xarray as xr
from ..gtensor import *
from mutopia.utils import parallel_gen
from ..model.gtensor_interface import GtensorInterface
import numpy as np
from functools import partial
from os import path


class MixtureInterface(GtensorInterface):

    @classmethod
    def to_datasets(cls, *datasets):
        return [ds for _, ds in cls.expand_datasets(*datasets)]

    @classmethod
    def expand_datasets(cls, *datasets):
        for dataset in datasets:
            for source in cls.list_sources(dataset):
                ds = cls.fetch_source(dataset, source)
                yield cls.get_name(ds), ds

    @classmethod
    def observation_dims(cls, dataset):
        return tuple(d for d in dataset.X.dims if not d == "sample")

    @classmethod
    def get_name(cls, dataset):
        return dataset.attrs["name"]

    @classmethod
    def is_mixture_corpus(cls, dataset):
        return cls.n_sources(dataset) > 1

    @classmethod
    def list_sources(cls, dataset):
        if "source" in dataset.coords:
            return dataset.coords["source"].values.tolist()
        return []

    @classmethod
    def n_sources(cls, dataset):
        return len(cls.list_sources(dataset))

    @staticmethod
    @mutate_wrapper
    def fetch_source(dataset, source):
        sources = MixtureInterface.list_sources(dataset)
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
            {
                os.path.join("State", source, v): os.path.join("State", v)
                for v in use_state
            }
        )

        other_dvars = [v for g in groups.values() for v in g]
        shared_features = [v for v in features if len(v.split("/")) == 2]
        source_corpus = dataset[
            list(rename_map.keys()) + other_dvars + shared_features
        ].rename(rename_map)
        source_corpus.attrs["name"] = MixtureInterface.get_name(dataset) + "/" + source

        if "source" in source_corpus.dims:
            source_corpus = source_corpus.sel(source=source, drop=True)

        return source_corpus

    @classmethod
    def sources(cls, dataset):
        for source in cls.list_sources(dataset):
            ds = cls.fetch_source(dataset, source)
            yield source, ds

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

        dataset.corpus = dataset.corpus.drop_dims(
            "component", errors="ignore"
        ).assign_coords(sample=sample_names)

        for source, data in cls.sources(dataset):
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

            state_elements = {
                f"State/{source}/{k}": v for k, v in state_elements.items()
            }

            dataset.corpus = dataset.corpus.assign(**state_elements)

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
                partial(model.update_corpusstate, ds, from_scratch=from_scratch)
                for model in model_state.models.values()
                for _, ds in cls.sources(dataset)
            ),
            par_context,
            ordered=False,
        ):
            pass

        return dataset

    @classmethod
    def _fetch_topic_compositions(cls, dataset, sample_name):
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
    def fetch_topic_compositions(cls, dataset, sample_name):
        return np.array(
            [
                cls._fetch_topic_compositions(ds, sample_name)
                for name, ds in cls.sources(dataset)
            ]
        )

    @classmethod
    def update_topic_compositions(cls, dataset, sample_name, gamma):
        for (_, ds), gamma_d in zip(cls.sources(dataset), gamma):
            cls._fetch_topic_compositions(ds, sample_name)[:] = gamma_d

    @classmethod
    def using_exposures_from(cls, *corpuses):
        corpus_dict = {cls.get_name(dataset): dataset for dataset in corpuses}

        return lambda dataset, sample_name: cls.fetch_topic_compositions(
            corpus_dict[cls.get_name(dataset)], sample_name
        )

    @classmethod
    def fetch_locals(cls, dataset):
        return xr.concat(
            [
                cls.fetch_val(source, "topic_compositions")
                for _, source in cls.sources(dataset)
            ],
            dim="source",
        ).assign_coords(
            source=cls.list_sources(dataset),
        )
