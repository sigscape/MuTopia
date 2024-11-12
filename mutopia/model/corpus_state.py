
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
    def list_samples(cls, corpus):
        return list(corpus.coords['sample'].values)

    @classmethod
    def fetch_sample(cls, corpus, sample_name):
        return corpus.layers.sel(sample=sample_name).X
    
    @classmethod
    def sample_dims(cls, corpus):
        return corpus.layers.X.dims

    @classmethod
    def init_corpusstate(
                cls,
                corpus,
                model_state,
            ):

        sample_names = cls.list_samples(corpus)
        n_components = model_state.n_components
            
        state_elements = {
                    'log_mutrate_normalizer' : xr.DataArray(
                        np.zeros(n_components),
                        dims=('component',),
                    ),
                }
        
        for model in model_state.models.values():
            state_elements.update(
                model.prepare_corpusstate(corpus)
            )

        DataTree(
            data=xr.Dataset(
                state_elements,
                coords={
                    **corpus.coords,
                    'components' : list(range(n_components)),
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
        