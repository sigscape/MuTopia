


import os
import mutopia.gtensor.disk_interface as disk

__all__ = [
    "CorpusInterface",
    "LazySampleLoader",
    "LazySlicer",
    "SampleCorpusFusion",
    "DifferentSamples",
]


class CorpusInterface:

    __slots__ = ("_corpus",)

    def __init__(self, corpus):  # datatree
        # first, make a copy of the corpus
        self._corpus = corpus

    def __getattr__(self, attr):
        return getattr(self._corpus, attr)

    @property
    def corpus(self):
        if issubclass(type(self._corpus), CorpusInterface):
            return self._corpus.corpus
        return self._corpus

    @corpus.setter
    def corpus(self, value):
        if issubclass(type(self._corpus), CorpusInterface):
            self._corpus.corpus = value
        else:
            self._corpus = value

    def __setitem__(self, key, value):
        self._corpus[key] = value

    def __getitem__(self, key):
        return self._corpus[key]

    def _clone(self):

        attributes = {"__init__": lambda self: None}
        for slot in self.__slots__:
            attributes[slot] = getattr(self, slot)

        return type(
            self.__class__.__name__,
            (self.__class__,),
            attributes,
        )()

    def mutate(self, fn):
        shift = self._clone()
        shift._corpus = shift._corpus.mutate(fn)
        return shift


class NoVars(ValueError):
    pass


class LazySampleLoader(CorpusInterface):

    def __init__(
        self,
        corpus: CorpusInterface,
    ):
        self._corpus = corpus

    @property
    def X(self):
        return self.fetch_sample(self.list_samples()[0]).X

    def _get_sample_vars(self, sample_name):
        sample_vars = [
            k for k, v in self._corpus.data_vars.items() if "sample" in v.dims
        ]
        if len(sample_vars) == 0:
            raise NoVars(
                "No sample variables found in the corpus. "
                "Please check the corpus structure."
            )

        return self._corpus[sample_vars].sel(sample=sample_name)

    def _get_filename(self):
        filename = self._corpus.attrs["filename"]
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"No such file exists anymore: {filename}, maybe it was deleted?"
            )
        return filename

    def fetch_sample(self, sample_name):
        try:
            return self._get_sample_vars(sample_name).merge(
                disk.load_sample(self._get_filename(), sample_name)
            )
        except NoVars:
            return disk.load_sample(self._get_filename(), sample_name)

    def iter_samples(self):

        sample_vars = [
            k for k, v in self._corpus.data_vars.items() if "sample" in v.dims
        ]
        has_sample_vars = len(sample_vars) > 0

        for sample_name, data in zip(
            self.list_samples(),
            disk.yield_samples(self._get_filename(), *self.list_samples()),
        ):
            if has_sample_vars:
                yield self._corpus[sample_vars].sel(sample=sample_name).merge(data)
            else:
                yield data


class LazySlicer(CorpusInterface):
    """
    Making slices of the corpus is memory-intensive.
    This problem is exacerbated when we want to slice by locus
    *then* by sample, since we have to generate a locus subset
    of the entire dataset and keep that in memory.

    Instead, this interface class allows us to lazily slice the corpus.
    First, we supply the corpus and the slices we want to apply.

    Later, we can fetch a sample by name and the slices will be applied
    to the original data matrix, foregoing the copying step.
    """

    __slots__ = ("_corpus", "_apply_slices", "_base_corpus")

    def __init__(self, corpus, keep_features=True, **slices):
        # first, make a copy of the corpus

        sliced = corpus.copy()

        if hasattr(sliced, "X"):
            sliced = sliced.drop_vars("X", errors="ignore")

        if not keep_features:
            sliced = sliced.drop_vars(corpus.sections.groups["Features"])

        self._base_corpus = corpus
        self._apply_slices = {d: s for d, s in slices.items() if not s is None}
        self._corpus = sliced.isel(**self._apply_slices)

    @property
    def X(self):
        return self._base_corpus.X

    def list_samples(self):
        return self._base_corpus.list_samples()

    def fetch_sample(self, sample_name):
        if not sample_name is None:
            return self._base_corpus.fetch_sample(sample_name).isel(
                **self._apply_slices
            )
        else:
            return self.isel(**self._apply_slices)

    def iter_samples(self):
        for sample in self._base_corpus.iter_samples():
            yield sample.isel(**self._apply_slices)


class SampleCorpusFusion(CorpusInterface):

    __slots__ = ("_corpus", "_sample")

    def __init__(self, corpus, sample):
        self._corpus = corpus
        self._sample = sample

    def list_samples(self):
        return ["0"]

    @property
    def X(self):
        return self._sample

    def fetch_sample(self, sample_name):
        return self._sample

    def iter_samples(self):
        yield self._sample


class BootstrapCorpus(CorpusInterface):

    def __init__(self, corpus, random_state):
        self._corpus = corpus

        self._sample = random_state.choice(
            corpus.list_samples(),
            size=len(corpus.list_samples()),
            replace=True,
        )

    def list_samples(self):
        return self._sample


class DifferentSamples(CorpusInterface):

    __slots__ = ("_corpus", "_samples")

    def __init__(self, corpus, samples):
        self._corpus = corpus
        self._samples = samples

    def list_samples(self):
        return self._samples
    
    def fetch_sample(self, sample_name):
        if sample_name not in self._samples:
            raise KeyError(f"Sample {sample_name} not found in the corpus.")
        return self._corpus.fetch_sample(sample_name)
    
    def iter_samples(self):
        for sample_name in self.list_samples():
            yield self.fetch_sample(sample_name)


class TransformerInterface(CorpusInterface):
    """
    This interface is used to transform the corpus into a different
    representation. For example, we might want to transform the corpus
    into a different space, or apply a different transformation to the
    data.
    """

    __slots__ = ("_corpus", "_func")

    def __init__(self, corpus, func):
        self._corpus = corpus
        self._func = func

    def fetch_sample(self, sample_name):
        return self._func(self._corpus.fetch_sample(sample_name))

    def iter_samples(self):
        for sample_name, data in self._corpus.iter_samples():
            yield sample_name, self._func(data)
