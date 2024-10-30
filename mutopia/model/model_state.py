
from .model_components import *
from .model_components.base import _svi_update_fn
from .latent_var_models import *
import numpy as np
import warnings
from functools import partial
from joblib import delayed
from itertools import chain
from scipy.special import logsumexp
from functools import cache, wraps, reduce
from collections import defaultdict


class ModelState:
    '''
    A simple container for the model components that allows for easy updating of the model state.
    '''
    def __init__(self,
                corpuses,
                *,
                context_model,
                mutation_model,
                theta_model,
                locals_model,
                ):
        self.context_model = context_model
        self.theta_model = theta_model
        self.mutation_model = mutation_model
        self.locals_model = locals_model
        
        self._normalizers = {
            corpus.attrs['name'] : np.zeros(self.n_components)
            for corpus in corpuses
        }

    @property
    def n_components(self):
        return self.context_model.n_components
    

    @property
    def models(self):
        return {
            'context' : self.context_model,
            'mutation' : self.mutation_model,
            'theta' : self.theta_model,
            'locals' : self.locals_model,
        }
    
    @property
    def nonlocals(self):
        return {
            model_name : model
            for model_name, model in self.models.items()
            if model_name != 'locals'
        }


    def get_normalizers(self, corpus):
        return self._normalizers[corpus.attrs['name']]
        
        
    def format_counterfactual(self, k):
        return dict(zip(
                    self.context_model.transformer.feature_names_out,
                    np.exp(
                        self.context_model.format_counterfactual(k)[:,:,None] \
                            + np.log(self.context_model.context_distribution_[None,:,None]) \
                            + self.mutation_model.format_counterfactual(k)
                    )
                ))
    

    def get_propto_log_mutation_rate(self, k, corpus):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            un_normalized = (
                            np.log(corpus.regions.exposures) \
                            + np.log(corpus.regions.context_frequencies) \
                            + self.theta_model.predict(k, corpus) \
                            + self.context_model.predict(k, corpus)
                          )

        return un_normalized
    

    def get_sstats_dict(self, corpuses):
        # model -> corpus -> component -> sstats
        return {
            model_name + '_sstats' : {
                corpus.attrs['name'] : model.spawn_sstats(corpus)
                for corpus in corpuses
            }
            for model_name, model in self.models.items()
        }
    

    def get_exp_offsets_dict(self, 
        corpuses,*,
        parallel_context,
        norm_update_fn,
    ):
        '''
        We want `all_offsets` to be a dictionary of <model_name> -> <corpus> -> <k> -> <offset>
        '''
        all_offsets = defaultdict(lambda : defaultdict(dict))
        
        args = [
            (k, corpus)
            for k in range(self.n_components)
            for corpus in corpuses
        ]

        offset_fns = (
            partial(self._get_exp_offsets_k_c, k, corpus)
            for k, corpus in args
        )
            
        for (k, corpus), (norm, exp_offsets) in zip(
            args,
            parallel_context(delayed(fn)() for fn in offset_fns)
        ):
            for model_name, _offsets in exp_offsets.items():
                
                all_offsets\
                    [model_name+ '_offsets']\
                    [corpus.attrs['name']]\
                    [k] = _offsets
                
                norm_update_fn(
                    k, corpus, norm,
                )

        return all_offsets
    
    
    def _get_exp_offsets_k_c(self, k, corpus):

        model_predictions = {
            model_name : model.predict(k, corpus)
            for model_name, model in self.nonlocals.items()
            if model.requires_normalization
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            log_mutation_rate = reduce(
                lambda x,y: x+y, 
                model_predictions.values(),  # sum over models
                np.log(corpus.regions.exposures) \
                    + np.log(corpus.regions.context_frequencies)  # start with the background rates
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
                    for model_name, model in self.nonlocals.items()
                    if model.requires_normalization
                }
            )
            
            return (
                -logsumexp(log_mutation_rate.data),
                exp_offsets
            )
    

    def init_normalizers(self, corpuses):
        for corpus in corpuses:
            for k in range(self.n_components):
                self._update_normalizer(
                    k, corpus, 
                    -logsumexp(self.get_propto_log_mutation_rate(k, corpus).data)
                )

    def _update_normalizer(self, 
            k, corpus, logsum_mutation_rate, 
            learning_rate=1., 
            subsample_rate=1.
        ):
        
        norm = self._normalizers[corpus.attrs['name']]
        
        norm[k] = _svi_update_fn(
                norm[k],
                np.log(subsample_rate) + logsum_mutation_rate,
                learning_rate
            )
        

    def Mstep(self,
            sstats,
            offsets,
            corpuses,
            *,
            parallel_context,
            learning_rate=1.,
            subsample_rate=1.,
            update_prior=True,
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
            for model_name, model in self.nonlocals.items()
        ))

        for _ in parallel_context(delayed(fn)() for fn in update_fns):
            pass
