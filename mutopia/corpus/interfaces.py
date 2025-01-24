import os
from .disk_interface import _backend_load_sample, load_dataset

def lazy_load(corpus):
    return LazySampleLoader(CorpusInterface(load_dataset(corpus, with_samples=False)))


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

    def modality(self):
        return self.corpus.modality()

    @property
    def corpus(self):
        return self._corpus
    
    @property
    def features(self):
        return self.corpus.features

    @property
    def sample(self):
        return self.corpus.sample.values
    
    @property
    def sizes(self):
        return self.corpus.sizes
        
    @property
    def dims(self):
        return self.corpus.dims
    
    @property
    def coords(self):
        return self.corpus.coords
    
    @property
    def state(self):
        return self.corpus.state
    
    @property
    def regions(self):
        return self.corpus.regions
    
    @property
    def attrs(self):
        return self.corpus.attrs
    
    @property
    def X(self):
        return self.corpus.X
    
    def fetch_sample(self, sample_name):
        return self.corpus.fetch_sample(sample_name)
    

class LazySampleLoader(CorpusInterface):

    def __init__(self,
        corpus : CorpusInterface,
    ):
        self._corpus = corpus.corpus

    @property
    def X(self):
        return self.fetch_sample(self.corpus.sample.values[0])

    def fetch_sample(self, sample_name):
        return _fetch_sample_from_disk(
            self.corpus, 
            sample_name,
        )

    
class LazySampleSlicer(CorpusInterface):
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

    def __init__(self, corpus,*, sample, **slices):
        # first, make a copy of the corpus
        sliced = corpus.corpus.copy()
        # get a copy of the X layer
        self._apply_slices={d : s for d,s in slices.items() if not s is None}
        self._samples = sample

        if hasattr(sliced, 'X'):
            sliced = sliced.drop_vars('X', errors='ignore')
        sliced = sliced.drop_nodes(('features',))

        self._base_corpus=corpus
        
        self._corpus = sliced\
            .isel(**self._apply_slices)
        
    @property
    def X(self):
        return self._base_corpus.X
    
    @property
    def sample(self):
        return self._samples
    
    def fetch_sample(self, sample_name):
        if not sample_name is None:
            return self._base_corpus.fetch_sample(sample_name)\
                    .isel(**{d:s for d,s in self._apply_slices.items() if not d == 'sample'})
        else:
            return self.X.isel(**self._apply_slices)