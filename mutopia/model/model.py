from .corpus_state import CorpusState as CS
from .optim import *
from .model_components import *
from .latent_var_models import *
from ..plot.coef_matrix_plot import _plot_interaction_matrix
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
    

    def plot_signature(self, component, **kwargs):
        self.modality_.plot(
            self.model_state_.format_signature(component), 
            **kwargs
        )


    def plot_interaction_matrix(self, 
            component, 
            model='context_model', 
            palette='vlag',
            **kw,
        ):
        return _plot_interaction_matrix(
            self.model_state_.models[model]\
                .component_coef_summary(component),
            palette=palette,
            **kw
        )

    def predict_exposures(self, corpus):
        pass