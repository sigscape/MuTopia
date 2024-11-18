
from .model_components import *
from .model_components.base import _svi_update_fn
from .latent_var_models import *
from .corpus_state import CorpusState as CS
import numpy as np
import warnings
from functools import partial
from joblib import delayed
from itertools import chain
from scipy.special import logsumexp
from functools import cache, wraps, reduce
from collections import defaultdict
import xarray as xr


class ModelState:

    def __init__(self,
                corpuses,
                *,
                locals_model,
                **models,
                ):

        self.locals_model = locals_model
        self._models = {}

        for model_name, model in models.items():
            self._models[model_name] = model
        
        self._normalizers = {
            CS.get_name(corpus) : np.zeros(self.n_components)
            for corpus in corpuses
        }

    @property
    def n_components(self):
        return next(iter(self._models.values())).n_components

    @property
    def models(self):
        return {
            'locals' : self.locals_model,
            **self._models
        }
    
    @property
    def nonlocals(self):
        return self._models
    
    @property
    def requires_dims(self):
        return reduce(
            lambda x,y : x.union(y.requires_dims),
            self.nonlocals.values(),
            set(['sample'])
        )


    def get_normalizers(self, corpus):
        return self._normalizers[CS.get_name(corpus)]
    

    def _get_propto_log_mutation_rate(self, k, corpus):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            un_normalized = reduce(
                lambda x,y: x+y, 
                (
                    model.predict(k, corpus)
                    for model in self.nonlocals.values()
                    if model.requires_normalization
                ),  # sum over models
                np.log(corpus.regions.exposures) \
                    + np.log(corpus.regions.context_frequencies)  # start with the background rates
            )

        return un_normalized
    
    
    def predict(self, k, corpus, with_context=True):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            y_hat = reduce(
                lambda x,y: x+y, 
                (
                    model.predict(k, corpus)
                    for model in self.nonlocals.values()
                ),  # sum over models
                np.log(corpus.regions.exposures) \
                    + self.get_normalizers(corpus)[k]
            )
            if with_context:
                y_hat += np.log(corpus.regions.context_frequencies)

        return y_hat
    

    def _get_log_mutation_rate_tensor(
        self, 
        corpus,
        *,
        parallel_context,
        with_context=True,
    ):
        return xr.concat(
            list(parallel_context(
                delayed(self.predict)(k, corpus, with_context=with_context)
                for k in range(self.n_components)
            )),
            dim='component'
        )
    
    
    @staticmethod
    def _marginalize_mutrate(log_mutrate_tensor, exposures):
        return xr.apply_ufunc(
            lambda gamma, mu : np.nan_to_num(
                np.dot(mu, gamma/gamma.sum()),
                copy=False,
                nan=0.
            ),
            exposures,
            np.exp(log_mutrate_tensor - log_mutrate_tensor.max()),
            input_core_dims=[[], ('component',)],
        )
    

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
                    [CS.get_name(corpus)]\
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
    

    def init_normalizers(self, 
            corpuses,
            *,
            parallel_context
        ):

        _update_norm = lambda k, corpus : \
            self._update_normalizer(
                k, corpus, 
                -logsumexp(self._get_propto_log_mutation_rate(k, corpus).data)
            )

        update_fns = (
            partial(_update_norm, k, corpus)
            for k in range(self.n_components)
            for corpus in corpuses
        )

        for _ in parallel_context(delayed(fn)() for fn in update_fns):
            pass


    def _update_normalizer(self, 
            k, corpus, logsum_mutation_rate, 
            learning_rate=1., 
            subsample_rate=1.
        ):
        norm = self._normalizers[CS.get_name(corpus)]
        norm[k] = _svi_update_fn(
                norm[k],
                np.log(subsample_rate) + logsum_mutation_rate,
                learning_rate
            )
        
    def format_signature(self, k):
        return np.exp(reduce(
            lambda x,y : x+y,
            (
                model.format_signature(k)
                for model in self.nonlocals.values()
            )
        ))
        

    def Mstep(self,
            corpuses,
            sstats,
            offsets,
            *,
            parallel_context,
            learning_rate=1.,
            subsample_rate=1.,
            update_prior=True,
            use_parallel=False,
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

        it = parallel_context(delayed(fn)() for fn in update_fns) if use_parallel \
                else (fn() for fn in update_fns)
        
        for _ in it: pass


    def Estep(
        self,
        corpuses,
        learning_rate = 1.,
        subsample_rate = 1.,    
        *,
        parallel_context,
    ):

        latent_vars_model = self.locals_model

        '''
        A suffstats dictionary with the following structure:
        sstats[parameter_name][corpus_name] <- suffstats
        '''
        sstats = self.get_sstats_dict(corpuses)
        elbo = 0.

        '''
        Construct the update function for each sample from the corpus
        in a generator fashion. Curry the initial gamma value for the 
        update, but don't call the update function yet - we want to
        pass this expression to the multiprocessing pool.
        '''
        samples, updates = self.locals_model.get_update_fns(
            corpuses,
            self,
            learning_rate=learning_rate,
            subsample_rate=subsample_rate,
            parallel_context=parallel_context,
        )
        
        for (corpus, sample_name), (sample_suffstats, gamma_new, bound) in zip(
            samples, 
            parallel_context(delayed(update)() for update in updates)
        ):
                elbo += bound
                # Update the topic compositions for the sample
                CS.update_topic_compositions(
                    corpus, 
                    sample_name, 
                    gamma_new
                )
                
                for model_name, model in self.models.items():
                    '''
                    The latent variables model handle the observation data format
                    and the gamma updates. Here, the model state just delegates the
                    suffstat reduction back to the latent variables model, which
                    calls the model's reduce_sstats method depending on the data type.
                    '''
                    latent_vars_model.reduce_model_sstats(
                        model,
                        sstats[model_name + '_sstats'][CS.get_name(corpus)],
                        corpus,
                        **sample_suffstats
                    )


        return sstats, elbo