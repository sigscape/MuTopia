from .corpus_state import CorpusState as CS
from .optim import *
from .model_components import *
from .latent_var_models import *
from ..plot.coef_matrix_plot import _plot_interaction_matrix
from joblib import dump, load

import logging
logger = logging.getLogger(' Mutopia-Model ')
logger.setLevel('INFO')
   
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

    @property
    def n_components(self):
        return self.model_state_.n_components
    
    @property
    def component_names(self):
        return self.model_state_.component_names


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
    

    def annot_exposures(
        self,
        corpus,
        threads=1,
        verbose=0,
    ):
        
        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        with ParContext(threads, verbose=verbose) as par:
            exposures = self.model_state_.locals_model\
                            .predict(
                                corpus,
                                self.model_state_,
                                parallel_context=par
                            )

        corpus = corpus.assign_coords({'component' : self.component_names})
        corpus.obsm.update({'exposures' : exposures})
        logger.info('Added key to obsm: "exposures"')
        return corpus
    
    
    def annot_component_distributions(
        self, 
        corpus,
        threads=1,
    ):

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        with ParContext(threads) as par:
            topics = self.model_state_._get_log_mutation_rate_tensor(
                corpus,
                parallel_context=par,
                with_context=False,
            )

        corpus = corpus.assign_coords({'component' : self.component_names})
        corpus.varm.update({'component_distributions' : topics})
        logger.info('Added key to varm: "component_distributions"')
        return corpus


    def annot_imputed(
        self,
        corpus,
        threads=1,
    ):
        logger.info('Added key to layers: "imputed"')
        pass


    def annot_SHAP_values(
        self,
        corpus,
        threads=1,
    ):
        try:
            import shap
        except ImportError:
            raise ImportError('SHAP is required to calculate SHAP values')
        
        locus_model = self.model_state_.models['theta_model']
        X = locus_model._fetch_feature_matrix(corpus)

        background_idx = np.random.RandomState(0).choice(
                            len(X), size = 1000, replace = False
                        )
        
        def _component_shap(k):
            return shap.TreeExplainer(
                    locus_model.rate_models[k],
                    X[background_idx],
                ).shap_values(
                    X,
                    check_additivity=False,
                    approximate=False,
                )
        
        shap_matrix = np.array([
            np.squeeze(_component_shap(k))
            for k in range(self.n_components)
        ])
        
        features = corpus.state.coords['feature'].data

        corpus = corpus.assign_coords({'locus_features' : features})
        corpus.varm.update({
            'SHAP_values' : xr.DataArray(
                shap_matrix,
                dims=('component','locus','locus_features'),
            )
        })

        logger.info('Added key to varm: "SHAP_values"')
        return corpus

    
    def annot_residuals(
        self,
        corpus,
        threads=1,
        keepdims=('locus',),
    ):
        logger.info('Added key to varm: "residuals"')
        pass
