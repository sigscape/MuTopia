
import xarray as xr
from datatree import DataTree
import numpy as np
from joblib import delayed
from ..corpus.interfaces import CorpusInterface

class CorpusState:

    @classmethod
    def update_corpusstate(
        cls,
        corpus,
        model_state,
        from_scratch=False,
        *,
        parallel_context,
    ):
        for _ in parallel_context(
            delayed(model.update_corpusstate)(
                corpus,
                from_scratch=from_scratch
            ) for model in model_state.models.values()
        ):
            pass
        
        return corpus
    
    @classmethod
    def has_corpusstate(cls, corpus):
        try:
            corpus.state
            return True
        except AttributeError:
            return False
    
    @classmethod
    def is_marginal_corpus(cls, corpus):
        return len(cls.list_samples(corpus)) <= 1
    
    @classmethod
    def list_samples(cls, corpus):
        return list(corpus.list_samples())
    
    @classmethod
    def iter_samples(cls, corpus):
        return zip(corpus.list_samples(), corpus.iter_samples())

    @classmethod
    def fetch_sample(cls, corpus, sample_name):
        if sample_name is None:
            return corpus.X
        return corpus.fetch_sample(sample_name)

    @classmethod
    def sample_dims(cls, corpus):
        return corpus.X.dims
    
    @classmethod
    def observation_dims(cls, corpus):
        return [d for d in cls.sample_dims(corpus) if not d in ('locus', 'sample')]

    @classmethod
    def update_normalizers(cls, corpus, normalizers):
        
        '''train_genome_size = model_state.get_genome_size(corpus)
        this_genome_size = cls.genome_size(corpus)

        cls.fetch_normalizers(corpus)[:] = \
            model_state.get_normalizers(corpus) - np.log(this_genome_size/train_genome_size)
        
        return corpus'''
        cls.fetch_normalizers(corpus)[:] = normalizers
        return corpus
    

    @classmethod
    def init_corpusstate(
                cls,
                corpus,
                model_state,
            ):

        sample_names = cls.list_samples(corpus)
        n_components = model_state.n_components
            
        state_elements = {
            'normalizers' : xr.DataArray(
                np.zeros(n_components, dtype=float),
                dims=('component',),
            ),
        }

        for model in model_state.models.values():
            state_elements.update(
                model.prepare_corpusstate(corpus)
            )

        # a smidge of interface chicanery here
        # The "corpus" passed to this function may be a nested series of interfaces, or it could be a plain G-Tensor.
        # Since we want to be able to modify the wrapped corpus, we add one more layer of indirection,
        # then modify the corpus in place.
        corpus = CorpusInterface(corpus)
        corpus.corpus = corpus.drop_dims('component', errors='ignore')

        DataTree(
            data=xr.Dataset(
                state_elements,
                coords={
                    **corpus.coords,
                    'sample' : sample_names,
                },
                attrs={
                    'genome_size' : corpus.regions.context_frequencies\
                                        .sum().data.item()
                }
            ),
            name='state',
            parent=corpus.corpus,
        )

        return corpus
    

    @staticmethod
    def genome_size(corpus):
        return corpus.state.attrs['genome_size']
    
    @staticmethod
    def fetch_normalizers(corpus):
        return corpus.state['normalizers'].data
    
    @staticmethod
    def fetch_val(corpus, key):
        return corpus.state[key]
    
    @staticmethod
    def get_name(corpus):
        return corpus.attrs['name']
    
    @staticmethod
    def update(corpus, **items):
        corpus.state = corpus.state.assign(**items)
        return corpus

    @classmethod
    def update_topic_compositions(cls, corpus, sample_name, gamma):
        cls.fetch_topic_compositions(corpus, sample_name)[:] = gamma
        return corpus
    

    @classmethod
    def fetch_topic_compositions(cls, corpus, sample_name):
        gamma = cls.fetch_val(corpus, 'topic_compositions')\
                    .sel(sample=sample_name)\
                    .transpose('component',...)\
                    .data
        
        if len(gamma.shape) > 1:
            gamma = gamma[:,0]

        return gamma
    
    
    @classmethod
    def using_exposures_from(cls, *corpuses):
        
        corpus_dict = {cls.get_name(corpus) : corpus for corpus in corpuses}
    
        return lambda corpus, sample_name : \
            cls.fetch_topic_compositions(
                corpus_dict[cls.get_name(corpus)],
                sample_name
            )
        