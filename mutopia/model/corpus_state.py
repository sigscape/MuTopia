
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
            parent=corpus
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
        return cls.fetch_val(corpus, 'topic_compositions')\
                    .sel(sample=sample_name)\
                    .data
    
    
    @classmethod
    def using_exposures_from(cls, *corpuses):
        
        corpus_dict = {cls.get_name(corpus) : corpus for corpus in corpuses}
    
        return lambda corpus, sample_name : \
            cls.fetch_topic_compositions(
                corpus_dict[cls.get_name(corpus)],
                sample_name
            )
        