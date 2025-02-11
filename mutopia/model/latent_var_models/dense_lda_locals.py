from xarray import DataArray
import numpy as np
from functools import partial
import warnings
from ..model_components.base import _svi_update_fn, PrimitiveModel
from ..corpus_state import CorpusState as CS
from ...utils import match_dims
from ._dirichlet_update import update_alpha
from .base import *


class LDAUpdateDense(LocalUpdate):
    
    def _convert_sample(self, sample):
        return np.ascontiguousarray(sample.load().data, dtype=self.dtype)


    @staticmethod
    def _calc_sstats(
        alpha, 
        conditional_likelihood, 
        weights,
        gamma,
        batch_subsample=1.,
        *,
        dims,
        ):

        args = (
            alpha,
            conditional_likelihood,
            weights,
            gamma,
        )

        weighted_posterior = calc_local_variables(*args)
        elbo = bound(*args, weighted_posterior)        

        suffstats = {
            'weighted_posterior' : DataArray(
                weighted_posterior/batch_subsample, 
                dims=('component', *dims),
            ),
            'gamma' : gamma, 
        }

        return (suffstats, gamma, elbo)
    
    
    @staticmethod
    def _apply_update(
        gamma, 
        *,
        suffstat_fn,
        update_fn, 
        learning_rate
    ):
        return suffstat_fn(
            _svi_update_fn(
                gamma, 
                update_fn(gamma), 
                learning_rate
            )
        )
    

    def _get_update_fn(
        self,
        gamma,
        learning_rate=1.,
        locus_subsample=1.,
        batch_subsample=1.,
        *,
        corpus,
        sample,
        conditional_likelihood,
        dims,
    ):
        '''
        Why am I doing it this way? - one could pass 
        the update function pointfree to some multiprocessing
        generator.

        I took pains to prevent the large corpus and model_state 
        objects from being pulled into the scope of the update function.
        '''
        
        # 1. get the information we need from the sample
        weights = self._convert_sample(sample)/locus_subsample
        alpha = np.ascontiguousarray(self.alpha[CS.get_name(corpus)])
        
        args = (
            alpha, 
            conditional_likelihood, 
            weights,
        )
        
        map_update = partial(
            iterative_update,
            *args,
            self.estep_iterations,
            self.difference_tol,
        )
        
        suffstat_fn = partial(
            self._calc_sstats,
            *args,
            dims=dims,
            batch_subsample=batch_subsample,
        )

        svi_update = partial(
            self._apply_update,
            np.ascontiguousarray(gamma),
            update_fn=map_update,
            learning_rate=learning_rate,
            suffstat_fn=suffstat_fn
        )

        return svi_update
    

    def _conditional_observation_likelihood(
        self,
        corpus,
        model_state,
        logsafe=True,
        *,
        parallel_context,
    ):
        
        obs_dims = CS.observation_dims(corpus)
        sample_dims = (*obs_dims, 'locus')

        logsafe_transform = lambda x : np.nan_to_num(np.exp(x - np.nanmax(x)), nan=0., neginf=0.)
        exp_transform = lambda x : np.nan_to_num(np.exp(x), nan=0., neginf=0.)

        lcol = model_state\
            ._get_log_mutation_rate_tensor(
                corpus, 
                parallel_context=parallel_context,
                with_context=False,
            )\
            .transpose('component', *sample_dims)\
            .data
        
        return np.ascontiguousarray(
            (logsafe_transform if logsafe else exp_transform)(lcol) 
        )
    

    def get_update_fns(
        self,
        corpuses,
        model_state,
        learning_rate=1.,
        locus_subsample=1.,
        batch_subsample=1.,
        exposures_fn=CS.fetch_topic_compositions,
        *,
        parallel_context,
    ):
        
        samples = [
            (corpus, sample_name)
            for corpus in corpuses
            for sample_name in CS.list_samples(corpus)
        ]

        likelihood_tensors = {
            CS.get_name(corpus) : (
                self._conditional_observation_likelihood(
                    corpus,
                    model_state,
                    parallel_context=parallel_context,
                    logsafe=True,
                )
            )
            for corpus in corpuses
        }

        updates = (
            self._get_update_fn(
                exposures_fn(corpus, sample_name),
                sample=CS.fetch_sample(corpus, sample_name),
                corpus=corpus,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
                conditional_likelihood=likelihood_tensors[CS.get_name(corpus)],
                dims=(*CS.observation_dims(corpus), 'locus'),
            )
            for (corpus, sample_name) in samples
        )

        return (
            samples,
            updates
        )
    
    
    def _sample_deviance(
        self,
        corpus,
        sample,
        model_state,
        *,
        gamma,
        log_mutrate_tensor,
        context_sum,
        context_effects,
    ):
        
        weights = self._convert_sample(sample)
        gamma = np.ascontiguousarray(gamma)

        log_marginal_mutrate = model_state\
            ._log_marginalize_mutrate(
                log_mutrate_tensor,
                gamma
            )
        
        # saturated
        y_sum = np.sum(weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            d_sat = np.nansum(weights * np.log(weights)) - y_sum * np.log(y_sum)
            # model
            d_fit = np.nansum(weights * log_marginal_mutrate.data)
            # null
            d_null = np.nansum(weights * context_effects) - y_sum * np.log(context_sum)

        return (
            d_sat - d_fit,
            d_sat - d_null,
        )
    

    def get_deviance_fns(
        self,
        corpuses,
        model_state,
        exposures_fn=CS.fetch_topic_compositions,
        *,
        parallel_context,
    ):
        '''
        -> List[F() -> Tuple[float, float]]
        '''
        samples = [
            (corpus, sample_name)
            for corpus in corpuses
            for sample_name in CS.list_samples(corpus)
        ]

        obs_dims = CS.observation_dims(corpuses[0])
        sample_dims = (*obs_dims, 'locus')
        
        log_mutrate_tensors = {
            CS.get_name(corpus) : model_state\
                ._get_log_mutation_rate_tensor(
                    corpus, 
                    parallel_context=parallel_context,
                    with_context=True
                )\
                .transpose('component', *sample_dims)
            for corpus in corpuses
        }      

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            context_effects = {
                CS.get_name(corpus) : match_dims(
                    np.log(corpus.regions.context_frequencies) + np.log(corpus.regions.exposures),
                    **{d : corpus.sizes[d] for d in sample_dims}
                )\
                .transpose(*sample_dims)\
                .data
                for corpus in corpuses
            }

            context_sums = {
                CS.get_name(corpus) : np.nansum(np.exp(context_effects[CS.get_name(corpus)].data))
                for corpus in corpuses
            }

        deviance_fns = (
            partial(
                self._sample_deviance,
                sample=CS.fetch_sample(corpus, sample_name),
                corpus=corpus,
                model_state=model_state,
                gamma=exposures_fn(corpus, sample_name),
                context_sum=context_sums[CS.get_name(corpus)],
                log_mutrate_tensor=log_mutrate_tensors[CS.get_name(corpus)],
                context_effects=context_effects[CS.get_name(corpus)],
            )
            for (corpus, sample_name) in samples
        )

        return deviance_fns
    

    @staticmethod
    def reduce_model_sstats(
        model, 
        carry, 
        corpus, 
        **sample_sstats
    ):
        return model.reduce_dense_sstats(
            carry, 
            corpus,
            **sample_sstats
        )
