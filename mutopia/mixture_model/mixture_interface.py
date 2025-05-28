"""
Why go through the trouble of creating these redundant functions?
This interface is meant to be used by the model, which should not depend on the underlying
structure of the datasets.
"""

import xarray as xr
import os
from ..utils import parallel_gen
from ..model.gtensor_interface import GtensorInterface
from ..gtensor import CorpusInterface, mutate_method
import numpy as np
from functools import partial
from os import path

class MixtureInterface(GtensorInterface):

    def to_datasets(self, *datasets):
        return [ds for _, ds in self.expand_datasets(*datasets)]

    def expand_datasets(self, *datasets):
        for dataset in datasets:
            for source in self.list_sources(dataset):
                ds = self.fetch_source(dataset, source)
                yield self.get_name(ds), ds

    @classmethod
    def observation_dims(cls, dataset):
        return tuple(d for d in dataset.X.dims if not d == "sample")

    @classmethod
    def is_mixture_corpus(cls, dataset):
        return "source" in dataset.coords and len(dataset.coords["source"]) > 1

    def list_sources(self, dataset):
        if "source" in dataset.coords:
            return dataset.coords["source"].values.tolist()
        return []

    def n_sources(self, dataset):
        return len(self.list_sources(dataset))

    @mutate_method
    def fetch_source(self, dataset, source):
        sources = self.list_sources(dataset)
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

        source_corpus = source_corpus.assign_coords(**dataset.coords).drop_dims("source")

        return source_corpus

    def sources(self, dataset):
        for source in self.list_sources(dataset):
            ds = self.fetch_source(dataset, source)
            yield source, ds

    def init_state(
        self,
        dataset,
        factor_model,
        locals_model,
    ):
        dataset = CorpusInterface(dataset)

        if self.has_corpusstate(dataset):
            dataset.corpus = dataset.corpus.drop_vars(dataset.sections.groups["State"])
            if "component" in dataset.dims:
                dataset.corpus = dataset.corpus.drop_dims("component")
            if "feature" in dataset.dims:
                dataset.corpus = dataset.corpus.drop_dims("feature")

        sample_names = dataset.list_samples()
        n_components = factor_model.n_components

        dataset.corpus = dataset.corpus.drop_dims(
            "component", errors="ignore"
        ).assign_coords(sample=sample_names)

        for source, data in self.sources(dataset):
            state_elements = {}

            state_elements.update(locals_model.prepare_corpusstate(data))

            for model in factor_model.models.values():
                state_elements.update(model.prepare_corpusstate(data))

            state_elements = {
                f"State/{source}/{k}": v for k, v in state_elements.items()
            }

            dataset.corpus = dataset.corpus.assign(**state_elements)

        return dataset

    def update_state(
        self,
        dataset,
        model_state,
        from_scratch=False,
        par_context=None,
    ):
        for _ in parallel_gen(
            (
                partial(model.update_corpusstate, ds, from_scratch=from_scratch)
                for model in model_state.models.values()
                for _, ds in self.sources(dataset)
            ),
            par_context,
            ordered=False,
        ):
            pass

        return dataset

    def _fetch_topic_compositions(self, dataset, sample_name):
        gamma = (
            self.fetch_val(dataset, "topic_compositions")
            .sel(sample=sample_name)
            .transpose("component", ...)
            .data
        )

        if len(gamma.shape) > 1:
            gamma = gamma[:, 0]

        return gamma

    def fetch_topic_compositions(self, dataset, sample_name):
        return np.array(
            [
                self._fetch_topic_compositions(ds, sample_name)
                for name, ds in self.sources(dataset)
            ]
        )

    def update_topic_compositions(self, dataset, sample_name, gamma):
        for (_, ds), gamma_d in zip(self.sources(dataset), gamma):
            self._fetch_topic_compositions(ds, sample_name)[:] = gamma_d

    def using_exposures_from(self, *corpuses):
        corpus_dict = {self.get_name(dataset): dataset for dataset in corpuses}

        return lambda dataset, sample_name: self.fetch_topic_compositions(
            corpus_dict[self.get_name(dataset)], sample_name
        )

    def fetch_locals(self, dataset):
        return xr.concat(
            [
                self.fetch_val(source, "topic_compositions")
                for _, source in self.sources(dataset)
            ],
            dim="source",
        ).assign_coords(
            source=self.list_sources(dataset),
        )
