from xarray import DataArray
import numpy as np
from functools import partial, reduce
from ..model_components.base import _svi_update_fn, PrimitiveModel
from ..corpus_state import CorpusState as CS
from ._dirichlet_update import update_alpha
from .base import *


class LocalUpdateDense(PrimitiveModel, LocalUpdate):

    def __init__(self,
            corpuses,
            n_components,
            prior_alpha=1.0,
            estep_iterations=300,
            difference_tol=1e-4,
            dtype=float,
            *,
            random_state,
        ):
        self.estep_iterations = estep_iterations
        self.difference_tol = difference_tol
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype

        self.alpha = {
            corpus.attrs['name'] : np.ones(n_components, dtype=dtype)*prior_alpha
            for corpus in corpuses
        }
    
    @staticmethod
    def _convert_sample(sample):
        return np.ascontiguousarray(sample.data)


    @staticmethod
    def _calc_sstats(
        alpha, 
        conditional_likelihood, 
        weights,
        gamma,
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
                weighted_posterior, 
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
        learning_rate=1.,
        subsample_rate=1.,
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
        weights = self._convert_sample(sample)/subsample_rate
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
        )

        svi_update = partial(
            self._apply_update,
            update_fn=map_update,
            learning_rate=learning_rate,
            suffstat_fn=suffstat_fn
        )

        return svi_update
    

    def get_update_fns(
        self,
        corpuses,
        model_state,
        learning_rate=1.,
        subsample_rate=1.,
        *,
        parallel_context,
    ):
        
        samples = [
            (corpus, sample_name)
            for corpus in corpuses
            for sample_name in CS.list_samples(corpus)
        ]

        _corpus, _sname = samples[0]
        sample_dims = _corpus.samples[_sname].dims

        transform = lambda x : np.ascontiguousarray(np.nan_to_num(np.exp(x), nan=0.))
        
        likelihood_tensors = {
            CS.get_name(corpus) : transform(
                model_state\
                ._get_log_mutation_rate_tensor(
                    corpus, 
                    parallel_context=parallel_context,
                    with_context=False
                )\
                .transpose('component', *sample_dims)\
                .data
            )
            for corpus in corpuses
        }

        updates = (
            partial(
                self._get_update_fn(
                    sample=corpus.samples[sample_name],
                    corpus=corpus,
                    learning_rate=learning_rate,
                    subsample_rate=subsample_rate,
                    conditional_likelihood=likelihood_tensors[CS.get_name(corpus)],
                    dims=sample_dims,
                ),
                np.ascontiguousarray(CS.fetch_topic_compositions(corpus, sample_name)),
            )
            for (corpus, sample_name) in samples
        )

        return (
            samples,
            updates
        )

    
    def bound(self,
        gamma,
        subsample_rate=1.,
        *,
        corpus,
        sample,
        model_state,
    ):
        
        sample_dict = self._convert_sample(sample)
        alpha = np.ascontiguousarray(self.alpha[CS.get_name(corpus)])
        gamma = np.ascontiguousarray(gamma)
        weights = sample_dict['weights']/subsample_rate
        
        conditional_likelihood = \
            self._conditional_observation_likelihood(
                corpus, 
                model_state,
                **sample_dict
            )

        return self._calc_sstats(
            alpha,
            conditional_likelihood,
            weights,
            gamma,
            sample_dict=sample_dict
        )[-1]
    

    ## 
    # M-step functionality to satisfy the PrimModel interface
    ##
    def init_locals(self, n_samples):
        return self.random_state.gamma(
                            100., 
                            1./100., 
                            size=(self.n_components, n_samples)
                        )
    

    def prepare_corpusstate(self, corpus):
        return dict(
            topic_compositions = DataArray(
                self.init_locals( len(corpus.samples.keys()) ),
                dims=('component','sample')
            )
        )

    def spawn_sstats(self, corpus):
        return []
    

    @staticmethod
    def reduce_dense_sstats(
        sstats, 
        corpus,
        *,
        gamma,
        **kw,
    ):
        sstats.append(gamma)
        return sstats
    

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


    def partial_fit(
        self, sstats, learning_rate
    ):
        for corpus_name, gammas in sstats.items():
            alpha0 = self.alpha[corpus_name]
            self.alpha[corpus_name] = _svi_update_fn(
                alpha0,
                update_alpha(alpha0, np.array(gammas)),
                learning_rate
            )
