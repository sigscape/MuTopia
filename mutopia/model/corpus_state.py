
import xarray as xr
from datatree import DataTree
from scipy.special import logsumexp
import numpy as np
import warnings

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
    def init_corpusstate(
                cls,
                corpus,
                model_state,
            ):

        sample_names = list(corpus.samples.keys())
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
        