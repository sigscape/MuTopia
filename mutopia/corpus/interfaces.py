import os
from functools import lru_cache
from .disk_interface import _backend_load_sample, load_dataset

def lazy_load(corpus):
    return LazySampleLoader(CorpusInterface(load_dataset(corpus, with_samples=False, with_state=False)))

def eager_load(corpus):
    return CorpusInterface(load_dataset(corpus, with_samples=True, with_state=False))


def lazy_train_test_load(corpus, test_chroms):

    corpus = lazy_load(corpus)

    test_mask = corpus.regions.chrom.isin(test_chroms)
    if test_mask.sum() == 0:
        raise ValueError(f'None of the chromosomes in {",".join(test_chroms)} are present in the corpus. ')

    train = LazySlicer(corpus, locus=~test_mask)
    test = LazySlicer(corpus, locus=test_mask)

    del corpus

    train._base_corpus = train._base_corpus.drop_nodes(['features','regions'])
    test._base_corpus = test._base_corpus.drop_nodes(['features','regions'])

    return train, test


def eager_train_test_load(corpus, test_chroms):

    corpus = eager_load(corpus)

    test_mask = corpus.regions.chrom.isin(test_chroms)
    if test_mask.sum() == 0:
        raise ValueError(f'None of the chromosomes in {",".join(test_chroms)} are present in the corpus. ')

    train = corpus.isel(locus=~test_mask)
    test = corpus.isel(locus=test_mask)

    del corpus

    return train, test


def _fetch_sample_from_disk(
    dataset,
    sample_name : str,
):
    filename = dataset.attrs['filename']
    if not os.path.exists(filename):
        raise FileNotFoundError(f'No such file exists anymore: {filename}, maybe it was deleted?')

    sample = _backend_load_sample(
        filename, 
        f'raw/X/{sample_name}'
    )

    return sample


class CorpusInterface:
    '''
    Sometimes, we'd like to drop something else into the EM step
    instead of a G-Tensor corpus. If the object implements the 
    following interface (and the outputs are the expected type), it will work.

    Note, we don't need to copy "features", "varm" etc.
    because those elements are not used in the EM step.
    '''
    def __init__(
        self, 
        corpus # datatree
    ):
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
        

class LazySampleLoader(CorpusInterface):

    def __init__(self,
        corpus : CorpusInterface,
    ):
        self._corpus = corpus
        self._X = self.fetch_sample(self.list_samples()[0])

    @property
    def X(self):
        return self._X

    def fetch_sample(self, sample_name):
        return _fetch_sample_from_disk(
            self._corpus, 
            sample_name,
        )

    
class LazySlicer(CorpusInterface):
    '''
    Making slices of the corpus is memory-intensive.
    This problem is exacerbated when we want to slice by locus
    *then* by sample, since we have to generate a locus subset
    of the entire dataset and keep that in memory.

    Instead, this interface class allows us to lazily slice the corpus.
    First, we supply the corpus and the slices we want to apply.
    
    Later, we can fetch a sample by name and the slices will be applied
    to the original data matrix, foregoing the copying step.
    '''

    def __init__(self, corpus, keep_features=True, **slices):
        # first, make a copy of the corpus

        self._base_corpus=corpus
        self._apply_slices={d : s for d,s in slices.items() if not s is None}
        
        sliced = corpus.copy()

        if hasattr(sliced, 'X'):
            sliced = sliced.drop_vars('X', errors='ignore')

        if not keep_features:
            sliced = sliced.drop_nodes(('features',))

        self._corpus = sliced\
            .isel(**self._apply_slices)
        
    @property
    def X(self):
        return self._base_corpus.X
    
    def list_samples(self):
        return self._base_corpus.list_samples()
    
    def fetch_sample(self, sample_name):
        if not sample_name is None:
            return self._base_corpus.fetch_sample(sample_name)\
                    .isel(**{d:s for d,s in self._apply_slices.items()})
        else:
            return self.X.isel(**self._apply_slices)
        

class SampleCorpusFusion(CorpusInterface):

    def __init__(self, corpus, sample):
        self._corpus = corpus
        self._sample = sample

    def list_samples(self):
        return ['0']
    
    @property
    def X(self):
        return self._sample
    
    def fetch_sample(self, sample_name):
        return self._sample
        

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

    def __init__(self, corpus, samples):
        self._corpus = corpus
        self._samples = samples

    def list_samples(self):
        return self._samples
    