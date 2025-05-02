
from .model_components import *
from .model_components.base import _svi_update_fn
from .latent_var_models import *
from ..utils import parallel_map, parallel_gen
from . import corpus_state as CS
import numpy as np
import warnings
from functools import partial
from joblib import delayed
from itertools import chain
from scipy.special import logsumexp
from functools import reduce
from collections import defaultdict
import xarray as xr


class FactorModel:

    def __init__(self,
        corpuses,
        offsets_fn = None,
        **models,
    ):

        self.offsets_fn = offsets_fn
        self._models = {}

        for model_name, model in models.items():
            self._models[model_name] = model
        
        self._normalizers = {
            CS.get_name(corpus) : np.zeros(self.n_components)
            for corpus in corpuses
        }

        self._genome_size = {
            CS.get_name(corpus) : CS.get_regions(corpus).context_frequencies.sum().item()
            for corpus in corpuses
        }

    @property
    def n_components(self):
        return next(iter(self._models.values())).n_components

    @property
    def models(self):
        return self._models
    
    def __getitem__(self, model_name):
        return self._models[model_name]
    
    def __getattr__(self, model_name):
        if model_name in self._models:
            return self._models[model_name]
        else:
            raise AttributeError(f"Model {model_name} not found.")

    
    @property
    def requires_dims(self):
        return reduce(
            lambda x,y : x.union(y.requires_dims),
            self.models.values(),
            set(['sample'])
        )

    def get_normalizers(self, corpus):
        return self._normalizers[CS.get_name(corpus)]
    
    def get_genome_size(self, corpus):
        return self._genome_size[CS.get_name(corpus)]

    def _get_propto_log_mutation_rate(self, k, corpus):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            un_normalized = reduce(
                lambda x,y: x+y, 
                (
                    model.predict(k, corpus)
                    for model in self.models.values()
                    if model.requires_normalization
                ),  # sum over models
                np.log(CS.get_regions(corpus).exposures) \
                    + np.log(CS.get_regions(corpus).context_frequencies)  # start with the background rates
            )

        return un_normalized
    
    
    def predict(self, k, corpus, with_context=True):
        '''
        The difference between this method and the _get_propto_log_mutation_rate
        is that this method returns the log mutation rate for all
        models, not just those that require normalization.
        '''
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            y_hat = reduce(
                    lambda x,y: x+y, 
                    (
                        model.predict(k, corpus)
                        for model in self.models.values()
                    ),  # sum over models
                    np.log(CS.get_regions(corpus).exposures) \
                    + CS.fetch_normalizers(corpus)[k]
                )
            if with_context:
                y_hat += np.log(CS.get_regions(corpus).context_frequencies)

        return y_hat
    

    def _get_log_mutation_rate_tensor(
        self, 
        corpus,
        par_context=None,
        *,
        with_context=True,
    ):
        return xr.concat(
            parallel_map(
                (
                    partial(self.predict, k, corpus, with_context=with_context, par_context=par_context) 
                    for k in range(self.n_components)
                )
            ),
            dim='component'
        )
    
    
    @staticmethod
    def _log_marginalize_mutrate(log_mutrate_tensor, exposures):
        
        def logsafe_matmul(y, log_x):
            alpha = log_x.max()
            return alpha + np.log(
                np.dot(np.nan_to_num(np.exp(log_x - alpha), nan=0., copy=False), y)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return xr.apply_ufunc(
                logsafe_matmul,
                exposures/np.sum(exposures),
                log_mutrate_tensor,
                input_core_dims=[[], ['component']],
            )
        
    
    def _log_component_posterior(self, log_mutrate_tensor, exposures):
        log_X_tild = self._log_marginalize_mutrate(log_mutrate_tensor, exposures)
        return log_mutrate_tensor - log_X_tild
    

    def get_sstats_dict(self, corpuses):
        # model -> corpus -> component -> sstats
        return {
            model_name + '_sstats' : {
                CS.get_name(corpus) : model.spawn_sstats(corpus)
                for corpus in corpuses
            }
            for model_name, model in self.models.items()
        }
    

    def get_exp_offsets_dict(self, 
        corpuses,
        *,
        par_context=None,
    ):
        '''
        We want `all_offsets` to be a dictionary of <model_name> -> <corpus> -> <k> -> <offset>
        '''
        all_offsets = defaultdict(lambda : defaultdict(dict))
        normalizers = {
            CS.get_name(corpus) : np.zeros_like(self.get_normalizers(corpus))
            for corpus in corpuses
        }
        
        args = [
            (k, corpus)
            for k in range(self.n_components)
            for corpus in corpuses
        ]

        offset_fns = (
            partial(self.offsets_fn or self._get_exp_offsets_k_c, k, corpus)
            for k, corpus in args
        )
            
        for (k, corpus), (norm, exp_offsets) in zip(args, parallel_gen(offset_fns, par_context)):
            for model_name, _offsets in exp_offsets.items():
                
                all_offsets\
                    [model_name+ '_offsets']\
                    [CS.get_name(corpus)]\
                    [k] = _offsets
                
                normalizers[CS.get_name(corpus)][k] = norm

        return all_offsets, normalizers
    
    
    def _get_exp_offsets_k_c(self, k, corpus):

        model_predictions = {
            model_name : model.predict(k, corpus)
            for model_name, model in self.models.items()
            if model.requires_normalization
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            log_mutation_rate = reduce(
                lambda x,y: x+y, 
                model_predictions.values(),  # sum over models
                np.log(CS.get_regions(corpus).exposures) \
                    + np.log(CS.get_regions(corpus).context_frequencies)  # start with the background rates
            )

            '''
            To get the exp offsets for a model, you need to know the contribution of
            every other model to the log mutation rate - this way the other models are treated
            as fixed effects.

            IF the model "requires normalization" - or is part of the mutation rate estimation,
            then we need to subtract the model's prediction from the total prediction. This
            yields the sum of predictions from every other model.
            '''
            exp_offsets = defaultdict(
                lambda : None, 
                {
                    model_name : model.get_exp_offset(
                        log_mutation_rate - model_predictions[model_name],
                        corpus
                    )
                    for model_name, model in self.models.items()
                    if model.requires_normalization
                }
            )
            
            return (
                -logsumexp(log_mutation_rate.data),
                exp_offsets
            )
        

    def _step_normalizers(self, 
        corpus,
        normalizers, 
        learning_rate=1., 
        subsample_rate=1.
    ):
        curr = self._normalizers[CS.get_name(corpus)]
        return  _svi_update_fn(
            curr,
            np.log(subsample_rate or 1.) + normalizers,
            learning_rate
        )
        
    
    def _calc_normalizers(self, 
        corpus,
        par_context=None,
    ):
        
        if self.offsets_fn is None:
            norm_fn = lambda k : \
                -logsumexp(self._get_propto_log_mutation_rate(k, corpus).data)
        else:
            norm_fn = lambda k : self.offsets_fn(k, corpus)[0]
            

        return np.array(parallel_map(
            (partial(norm_fn, k) for k in range(self.n_components)),
            par_context
        ))
    

    def update_normalizers(self, corpuses, par_context=None):
        for corpus in corpuses:
            CS.update_normalizers(
                corpus, 
                self._calc_normalizers(corpus, par_context)
            )
        

    def format_signature(self, k, normalization='global'):
        return np.exp(reduce(
            lambda x,y : x+y,
            (
                model.format_signature(k, normalization=normalization)
                for model in self.models.values()
            )
        ))
    

    def format_interactions(self, k):
        return reduce(
            lambda x,y : x+y,
            (
                model.get_interaction_summary(k)
                for model in self.models.values()
                if hasattr(model, 'get_interaction_summary')
            )
        )
    

    def Mstep(self,
        corpuses,
        sstats,
        offsets,
        par_context=None,
        learning_rate=1.,
        update_prior=True,
        use_parallel=True,
    ):

        if update_prior:
            self.locals_model.partial_fit(sstats['locals_sstats'], learning_rate=learning_rate)

        update_fns = chain.from_iterable((
            model.partial_fit(
                    k,
                    sstats[model_name + '_sstats'],
                    offsets[model_name + '_offsets'],
                    corpuses,
                    learning_rate=learning_rate,
                )
            for k in range(self.n_components)
            for model_name, model in self.models.items()
        ))

        it = parallel_gen(update_fns, par_context, ordered=False) \
            if use_parallel else (fn() for fn in update_fns)
        
        for _ in it: pass
