from .corpus_state import CorpusState as CS
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from ..utils import *
from .model_components import *
from .eval import deviance_locus, pearson_residuals
from .latent_var_models import *
from ..plot.coef_matrix_plot import _plot_interaction_matrix
from ..corpus import *
from ..corpus.interfaces import CorpusInterface
from functools import partial  
from joblib import dump, delayed
from collections import defaultdict
import typing
   
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
        try:
            self.component_names_
        except AttributeError:
            return ['M{}'.format(i) for i in range(0, self.n_components)]
        
    def _get_k(self, component_name):
        if isinstance(component_name, int):
            return component_name

        try:
            return self.component_names.index(component_name)
        except ValueError:
            raise ValueError(f'Component {component_name} not found in model.')


    def _check_corpus(self, corpus, enforce_sample=True):
        check_corpus(corpus, enforce_sample=enforce_sample)
        if enforce_sample:
            check_dims(corpus, self.model_state_)
        

    def set_component_names(self, names : typing.List[str]):
        if not len(names) == self.n_components:
            raise ValueError('The number of names must match the number of components')
        
        self.component_names_ = names


    def setup_corpus(self, corpus):
        
        logger.info('Setting up dataset state ...')

        corpus = CS.init_corpusstate(corpus, self.model_state_)
        
        with ParContext(1) as par:
            CS.update_corpusstate(
                corpus, self.model_state_, 
                from_scratch=True,
                parallel_context=par,
            )
        
        CS.update_normalizers(
            corpus, 
            self.model_state_.get_normalizers(corpus)
        )

        logger.info('Done ...')
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

        component = self._get_k(component)

        return self.modality_.plot(
            self.model_state_.format_signature(component, normalization=normalization), 
            *select,
            **kwargs
        )
    

    def signature_report(
        self, 
        component, 
        normalization='global',
        width=5.25,
        height=2.,
        show=True,
    ):

        component = self._get_k(component)

        signatures = self.model_state_.format_signature(component, normalization=normalization)
        n_rows = len(signatures.mesoscale_state)

        state_groups = defaultdict(list)
        for state in signatures.mesoscale_state.values:
            state_groups[state.split(':')[0]].append(state)

        for k, v in state_groups.items():
            if not k=='Baseline' and len(v)==1:
                state_groups[k].append('Baseline')

        max_n_states = max(map(len, state_groups.values()))
        n_sigs = len(state_groups)
        fig = plt.figure(figsize=(
            max(width*max_n_states, 10),
            height*n_sigs + 3
        ))

        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[height*n_sigs, 1+0.35*n_rows],
            hspace=0.1,
        )

        gs0 = gs[0].subgridspec(
            n_sigs + 1, max_n_states, 
            hspace=0.75, 
            wspace=0.5,
            width_ratios=[3] + [1]*(max_n_states-1),
        )

        for i, states in enumerate(state_groups.values()):
            ax = fig.add_subplot(gs0[i, :len(states)])
            self.plot_signature(
                component,
                *states,
                ax=ax,
            )

        self.plot_interaction_matrix(
            component,
            gridspec=gs[1],
            normalization=normalization,
        )

        fig.suptitle(
            f'Signature {component} report', 
            fontsize=12,
            y=0.95
        )

        if show:
            plt.show()

        return fig


    def plot_interaction_matrix(self, 
            component,
            palette='coolwarm',
            gridspec=None,
            normalization='global',
            **kw,
        ):

        component = self._get_k(component)

        flatten = partial(self.modality_._flatten_observations)

        interactions = self.model_state_.format_interactions(component)
        
        shared_effects = interactions.sel(context='Shared effect').to_pandas()
        interactions = flatten(interactions.drop_sel(context='Shared effect')).to_pandas()
        
        baseline = flatten(
            self.model_state_.format_signature(component, normalization=normalization).sel(
                mesoscale_state='Baseline',
            )
        ).to_pandas()

        return _plot_interaction_matrix(
            interactions,
            baseline,
            shared_effects, #.iloc[:,0],
            palette=palette,
            gridspec=gridspec,
            **kw
        )
    
    def signature_panel(
        self,
        ncols=3,
        normalization='global',
        width=3.5,
        height=1.25,
        show=True,
        **kwargs,
    ):
        
        K = self.n_components
        nrows = int(np.ceil(K/ncols))

        fig, ax = plt.subplots(
            nrows, ncols, 
            figsize=(width*ncols,height*nrows), 
            gridspec_kw={'hspace': 0.5, 'wspace': 0.25}
        )

        for k in range(self.n_components):
            _ax=np.ravel(ax)[k]
            self.plot_signature(k, ax=_ax, normalization=normalization, **kwargs)
            _ax.set_ylabel(self.component_names[k], fontsize=7)
            _ax.set_xticks([])

        for _ax in np.ravel(ax)[self.n_components:]:
            _ax.axis('off')

        if show:
            plt.show()
        else:
            return fig
    

    def annot_contributions(
        self,
        corpus,
        threads=1,
        verbose=0,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self.setup_corpus(corpus)

        with ParContext(threads, verbose=verbose) as par:
            contributions = self.model_state_.locals_model\
                .predict(
                    corpus,
                    self.model_state_,
                    parallel_context=par
                )
        
        corpus = CorpusInterface(corpus)
        corpus.corpus = (
            corpus
            .assign_coords({
                'component' : self.component_names,
            })
            .assign({
                'contributions' : contributions,
            })
        )

        logger.info('Added key to dataset: "contributions"')
        return corpus
    
    
    
    def annot_component_distributions(
        self, 
        corpus,
        threads=1,
    ):
        self._check_corpus(corpus, enforce_sample=False)

        if not CS.has_corpusstate(corpus):
            corpus = self.setup_corpus(corpus)

        with ParContext(threads) as par:
            lmrt = self.model_state_._get_log_mutation_rate_tensor(
                corpus,
                parallel_context=par,
                with_context=False,
            )

        corpus.varm['component_distributions'] =\
             np.exp(lmrt - lmrt.max(skipna=True)).fillna(0.)
        
        corpus.varm['component_locus_distributions'] =\
            (corpus.varm.component_distributions * corpus.regions.context_frequencies)\
            .sum(dim=dims_except_for(corpus.varm.component_distributions.dims, 'locus', 'component'))/corpus.regions.length
        
        logger.info('Added key to varm: "component_distributions"')
        logger.info('Added key to varm: "component_locus_distributions"')
        return corpus
        

    def annot_marginal_prediction(
        self,
        corpus,
        exposures=None,
        threads=1,
    ):
        self._check_corpus(corpus, enforce_sample=False)

        if not CS.has_corpusstate(corpus):
            corpus = self.setup_corpus(corpus)

        try:
            corpus.varm['component_distributions']
        except KeyError:
            corpus = self.annot_component_distributions(corpus, threads)

        if exposures is None:
            try:
                corpus['contributions']
            except KeyError:
                corpus = self.annot_contributions(corpus, threads)
            marginal_exposures = corpus['contributions'].sum(dim='sample').data
        else:
            marginal_exposures = np.sum(
                [exposures(corpus, sample_name) for sample_name in CS.list_samples(corpus)],
                axis=0,
            )
            
        marginal = \
            self.model_state_._log_marginalize_mutrate(
                np.log(corpus.varm['component_distributions']),
                marginal_exposures
            )
        
        corpus.varm['predicted_marginal'] = np.exp(marginal - marginal.max(skipna=True)).fillna(0.)
        corpus.varm['predicted_locus_marginal'] = (np.exp(marginal) * corpus.regions.context_frequencies)\
            .sum(dim=dims_except_for(marginal.dims, 'locus'))/corpus.regions.length
        
        logger.info('Added key to varm: "predicted_marginal"')
        logger.info('Added key to varm: "predicted_locus_marginal"')
        
        return corpus


    def annot_SHAP_values(
        self,
        corpus,
        *components,
        threads=1,
    ):
        
        try:
            import shap
        except ImportError:
            raise ImportError('SHAP is required to calculate SHAP values')
        
        self._check_corpus(corpus, enforce_sample=False)

        if not CS.has_corpusstate(corpus):
            corpus = self.setup_corpus(corpus)

        locus_model = self.model_state_.models['theta_model']
        X = locus_model._fetch_feature_matrix(corpus)

        background_idx = np.random.RandomState(0).choice(
            len(X), size = min(1000, len(X)), replace = False
        )
        
        def _component_shap(k):
            logger.info(f'Calculating SHAP values for {self.component_names[k]} ...')

            shaps = shap.TreeExplainer(
                locus_model.rate_models[k],
                X[background_idx],
            ).shap_values(
                X,
                check_additivity=False,
                approximate=False,
            )

            return np.squeeze(shaps)
        

        use_components = list(map(
            self._get_k,
            components if not len(components)==0 else list(range(self.n_components))
        ))

        '''with ParContext(threads) as par:
            shap_matrix = np.array(list(par(
                delayed(_component_shap)(k) for k in use_components
            )))

        print(shap_matrix)'''

        shap_matrix = np.array([
            _component_shap(k)
            for k in use_components
        ])
        
        corpus.varm['SHAP_values'] = xr.DataArray(
            shap_matrix,
            dims=('shap_component','locus','transformed_features'),
        )

        features = corpus.state.coords['feature'].data

        corpus = update_view(
            corpus,
            varm = corpus.varm.assign_coords({
                'transformed_features' : features,
                'shap_component' : [self.component_names[k] for k in use_components],
            }).to_dataset(),
        )

        logger.info('Added key to varm: "SHAP_values"')
        return corpus


    def score(
        self,
        corpus,
        exposures=None,
        threads=1,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self.setup_corpus(corpus)

        with ParContext(threads) as par:
            return deviance_locus(
                self.model_state_,
                (corpus,),
                exposures_fn=using_exposures_from(corpus) if exposures is None else exposures,
                parallel_context=par,
            )

    
    def annot_residuals(
        self,
        corpus,
        exposures=None,
        threads=1,
    ):
        self._check_corpus(corpus)

        if not CS.has_corpusstate(corpus):
            corpus = self.setup_corpus(corpus)
        
        with ParContext(threads) as par:
            residuals = pearson_residuals(
                self.model_state_,
                (corpus,),
                exposures_fn=using_exposures_from(corpus) if exposures is None else exposures,
                parallel_context=par,
            )
        
        corpus.varm['pearson_residuals'] = residuals
        logger.info('Added key to varm: "residuals"')

        return corpus