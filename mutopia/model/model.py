from .corpus_state import CorpusState as CS
from .optim import *
from .model_components import *
from .latent_var_models import *
from .optim import fit_model
from joblib import dump, load
   
'''
The Model class is a wrapper around a trained model state object, 
and provides the high-level interface for interacting with the model.
This is the entry point for the user to interact with and annotate data.
'''
class Model:

    @classmethod
    def load_model(cls, path):
        return load(path)

    def __init__(
        self,
        model_state,
        modality,
    ):
        self.model_state_ = model_state
        self.modality_ = modality


    def _setup_corpus(self, corpus):

        args = dict(
            corpus,
            self.model_state_,
        )
        corpus = CS.init_corpusstate(**args)
        corpus = CS.update_corpusstate(**args, from_scratch=True)

        return corpus
        

    def save(self, path):
        for model in self.model_state_.models.values():
            model.prepare_to_save()

        dump(self, path)
    

    def predict_exposures(self, corpus):
        pass