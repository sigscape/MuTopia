"""
Why go through the trouble of creating these redundant functions?
This interface is meant to be used by the model, which should not depend on the underlying
structure of the datasets.
"""

from numpy import asfortranarray, ascontiguousarray, float32
from functools import partial
from mutopia.gtensor import CorpusInterface
from mutopia.utils import parallel_gen


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
    def is_mixture_corpus(cls, dataset):
        return False

    @classmethod
    def n_sources(cls, dataset):
        return 1

    @classmethod
    def list_sources(cls, dataset):
        return [cls.get_name(dataset)]

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

        state_elements = {}

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
    def get_region_lengths(cls, dataset):
        return dataset["Regions/length"]

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
        return "State" in dataset.sections.groups

    @classmethod
    def is_marginal_corpus(cls, dataset):
        return len(cls.list_samples(dataset)) <= 1

    @classmethod
    def fetch_val(cls, dataset, key):
        return dataset["State/" + key]

    @classmethod
    def get_name(cls, dataset):
        return dataset.attrs["name"]

    @classmethod
    def fetch_topic_compositions(cls, dataset, sample_name):
        gamma = (
            cls.fetch_val(dataset, "topic_compositions")
            .sel(sample=sample_name)
            .transpose(..., "component")
            .data.ravel()
        )

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

    @classmethod
    def get_genome_size(cls, dataset):
        return cls.get_region_lengths(dataset).sum().item()

    @classmethod
    def prepare_data(cls, dataset):

        dataset["Regions/context_frequencies"] = dataset[
            "Regions/context_frequencies"
        ].transpose(..., "locus")

        dataset["Regions/context_frequencies"].data = asfortranarray(
            dataset["Regions/context_frequencies"].data,
            dtype=float32,
        )

        dataset["Regions/exposures"].data = ascontiguousarray(
            dataset["Regions/exposures"],
            dtype=float32,
        )

        if hasattr(dataset, "ploidy"):
            dataset["ploidy"].data.data = dataset["ploidy"].data.data.astype(float32)

        return dataset

    @classmethod
    def fetch_locals(cls, dataset):
        """
        Fetch local topic compositions for a given sample from the dataset.
        This is a wrapper around fetch_topic_compositions to maintain compatibility.
        """
        return cls.fetch_val(dataset, "topic_compositions")
