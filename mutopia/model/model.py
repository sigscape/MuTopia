from .corpus_state import CorpusState as CS
import numpy as np
import xarray as xr
from ..utils import ParContext, using_exposures_from, \
    check_corpus, check_dims
from .model_components import *
from .eval import deviance_locus
from .latent_var_models import *
from ..plot.coef_matrix_plot import _plot_interaction_matrix
from ..corpus import *
from functools import partial  
from joblib import dump
import typing

import logging
logger = logging.getLogger(' Mutopia-Model ')
logger.setLevel('INFO')
   
'''
The Model class is a wrapper around a trained model state object, 
and provides the high-level interface for interacting with the model.
This is the entry point for the user to interact with and annotate data.
'''
class Model:

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
        #return self.model_state_.component_names
        try:
            self.component_names_
        except AttributeError:
            return ['component_{}'.format(i) for i in range(1, self.n_components+1)]
        
    def _check_corpus(self, corpus):
        check_corpus(corpus)
        check_dims(corpus, self.model_state_)
        
    def set_component_names(self, names : typing.List[str]):
        if not len(names) == self.n_components:
            raise ValueError('The number of names must match the number of components')
        
        self.component_names_ = names


    def _setup_corpus(self, corpus):
        args = (corpus, self.model_state_)
        
        CS.init_corpusstate(*args)
        
        with ParContext(1) as par:
            CS.update_corpusstate(
                *args, 
                from_scratch=True,
                parallel_context=par,
            )
        
        CS.update_normalizers(
            corpus, 
            self.model_state_.get_normalizers(corpus)
        )
        return corpus
        

    def save(self, path):

        for model in self.model_state_.models.values():
            model.prepare_to_save()

        dump(self, path)
    
    
    def plot_signature(self, 
        component, 
        *select, 
        normalization='global',
        **kwargs
    ):
        if len(select) == 0:
            select = ['Baseline']

        return self.modality_.plot(
            self.model_state_.format_signature(component, normalization=normalization), 
            *select,
            **kwargs
        )


    def plot_interaction_matrix(self, 
            component,
            palette='vlag',
            **kw,
        ):

        flatten = partial(self.modality_._flatten_observations)

        interactions = self.model_state_.format_interactions(component)
        
        shared_effects = interactions.sel(context='Shared effect').to_pandas()
        interactions = flatten(interactions.drop_sel(context='Shared effect')).to_pandas()
        
        baseline = flatten(
            self.model_state_.format_signature(component).sel(
                mesoscale_state='Baseline',
            )
        ).to_pandas()

        return _plot_interaction_matrix(
            interactions,
            baseline,
            shared_effects.iloc[:,0],
            palette=palette,
            **kw
        )
    

    def annot_exposures(
        self,
        corpus,
        threads=1,
        verbose=0,
    ):
        self._check_corpus(corpus)

        self.model_state_.locals_model.estep_iterations=10000
        self.model_state_.locals_model.difference_tol=1e-5

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        with ParContext(threads, verbose=verbose) as par:
            exposures = self.model_state_.locals_model\
                .predict(
                    corpus,
                    self.model_state_,
                    parallel_context=par
                )

        corpus = corpus.assign_coords({
            'component' : self.component_names,
        })
        corpus.obsm['exposures'] = exposures
        logger.info('Added key to obsm: "exposures"')
        return corpus
    
    
    def annot_component_distributions(
        self, 
        corpus,
        threads=1,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        with ParContext(threads) as par:
            lmrt = self.model_state_._get_log_mutation_rate_tensor(
                corpus,
                parallel_context=par,
                with_context=False,
            )

        corpus = corpus.assign_coords({'component' : self.component_names})
        corpus.varm['component_distributions'] =\
             np.exp(lmrt - lmrt.max(skipna=True)).fillna(0.)
        
        logger.info('Added key to varm: "component_distributions"')
        return corpus
        

    def annot_marginal_prediction(
        self,
        corpus,
        exposures=None,
        threads=1,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        try:
            corpus.varm['component_distributions']
        except KeyError:
            corpus = self.annot_component_distributions(corpus, threads)

        marginal_exposures = corpus.obsm['exposures'].sum('sample').data

        corpus.varm['predicted_marginal'] = \
            self.model_state_._log_marginalize_mutrate(
                np.log(corpus.varm['component_distributions']),
                marginal_exposures
            )
        
        logger.info('Added key to varm: "predicted_marginal"')
        return corpus


    def annot_imputed(
        self,
        corpus,
        exposures=None,
        threads=1,
    ):  
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        try:
            corpus.varm['component_distributions']
        except KeyError:
            corpus = self.annot_component_distributions(corpus, threads)

        if exposures is None:
            exposures = using_exposures_from(corpus)

        log_component_mutrate = np.log(corpus.varm['component_distributions'])

        imputed = xr.concat(
            [
                self.model_state_._marginalize_mutrate(
                    log_component_mutrate,
                    exposures(sample_name)
                )
                for sample_name in CS.list_samples(corpus)
            ],
            dim='sample',
        ).transpose(
            'sample', *self.modality_.dims, 'locus',
        )

        corpus['imputed'] = imputed
        logger.info('Added key to layers: "imputed"')
        return corpus



    def annot_SHAP_values(
        self,
        corpus,
        threads=1,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

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
            
            logger.info(f'Calculating SHAP values for {self.component_names[k]} ...')

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

        corpus = corpus.assign_coords({'transformed_features' : features})
        corpus.varm['SHAP_values'] = xr.DataArray(
                shap_matrix,
                dims=('component','locus','transformed_features'),
            )

        logger.info('Added key to varm: "SHAP_values"')
        return corpus


    def score(
        self,
        corpus,
        exposures=None,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self._setup_corpus(corpus)

        with ParContext(1) as par:
            return deviance_locus(
                self.model_state_,
                (corpus,),
                exposures_fn=using_exposures_from(corpus) if exposures is None else exposures,
                parallel_context=par,
            )

    
    def annot_residuals(
        self,
        corpus,
        threads=1,
        keepdims=('locus',),
    ):
        self._check_corpus(corpus)
        
        logger.info('Added key to varm: "residuals"')
        pass
