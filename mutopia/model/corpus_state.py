
import xarray as xr
from datatree import DataTree
import numpy as np

class CorpusState:

    @classmethod
    def update_corpusstate(
        cls,
        corpus,
        model_state,
        from_scratch=False,
    ):
        
        for model in model_state.models.values():
            model.update_corpusstate(
                corpus, 
                from_scratch=from_scratch
            )
        
        return corpus
    
    @classmethod
    def has_corpusstate(cls, corpus):
        return hasattr(corpus, 'state')
    
    @classmethod
    def is_marginal_corpus(cls, corpus):
        return not 'sample' in corpus.coords
    
    @classmethod
    def list_samples(cls, corpus):
        if cls.is_marginal_corpus(corpus):
            return [None]
        
        return list(corpus.X.coords['sample'].values)

    @classmethod
    def fetch_sample(cls, corpus, sample_name):
        if cls.is_marginal_corpus(corpus) and sample_name is None:
            return corpus.X

        return corpus.X.sel(sample=sample_name)
    
    @classmethod
    def sample_dims(cls, corpus):
        return corpus.X.dims

    @classmethod
    def init_corpusstate(
                cls,
                corpus,
                model_state,
            ):

        sample_names = cls.list_samples(corpus)
        n_components = model_state.n_components
            
        state_elements = {}
        for model in model_state.models.values():
            state_elements.update(
                model.prepare_corpusstate(corpus)
            )

        DataTree(
            data=xr.Dataset(
                state_elements,
                coords={
                    **corpus.coords,
                    'sample' : sample_names,
                }
            ),
            name='state',
            parent=corpus
        )

        return corpus
    

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
        return cls.fetch_val(corpus, 'topic_compositions')\
                    .sel(sample=sample_name)\
                    .data
        