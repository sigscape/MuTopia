from xarray import DataArray
import numpy as np
from functools import partial, reduce
from ..model_components.base import _svi_update_fn, PrimitiveModel
from ._dirichlet_update import update_alpha
from ..corpus_state import CorpusState as CS
from ...utils import dims_except_for
from math import prod
from .base import *
import warnings


class LDAUpdateSparse(PrimitiveModel, LocalUpdate):

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
            CS.get_name(corpus) : np.ones(n_components, dtype=dtype)*prior_alpha
            for corpus in corpuses
        }
    
    ##
    # E-step functionality to satisfy the LocalUpdate interface
    ##
    def _convert_sample(self, sample):

        sample = sample.sparse_to_coo()
        weights = np.ascontiguousarray(
            sample.data.data.astype(self.dtype, copy=False)
        )

        idx_dict = dict(zip(
            tuple(sample.coords['obs_indices'].data),
            sample.indices.data,
        ))

        return dict(
            **idx_dict,
            weights=weights,
        )
    

    def _get_log_context_effect(
        self,
        corpus,
        *,
        locus,
        **idx_dict,
    ):
        
        freqs = corpus.regions.context_frequencies\
                    .transpose(...,'locus')
        
        idx_arrs = [idx_dict[dim] for dim in freqs.dims[:-1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return np.log(freqs.data[*idx_arrs, locus]) \
                + np.log(corpus.regions.exposures.data[locus])


    def _conditional_observation_likelihood(
        self,
        corpus,
        model_state,
        logsafe=True,
        *,
        weights,
        **idx_dict,
    ):

        ##
        # What's going on here: we have the normalized log mutation rate for each signature, configuration, context, locus.
        # For the mutations in this sample, we select over these axes.
        ##
        logp_normalizer = CS.fetch_normalizers(corpus)[:,None]

        offset = self._get_log_context_effect(
            corpus,
            **idx_dict
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            logp_X = reduce(
                lambda x,y: x+y,
                (
                    model.predict_sparse(corpus, **idx_dict)
                    for model in model_state.nonlocals.values()
                ),
                logp_normalizer \
                    + offset
            )

        if logsafe:
            logp_X - logp_X.max()

        return np.ascontiguousarray(
            np.nan_to_num(np.exp(logp_X), nan=0)\
                .astype(self.dtype, copy=False)
        )
    

    @staticmethod
    def _calc_sstats(
        alpha, 
        conditional_likelihood, 
        weights,
        gamma,
        *,
        sample_dict,
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
                **sample_dict,
                'weighted_posterior' : weighted_posterior, 
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
        subsample_rate=1.,
        *,
        corpus,
        sample,
        model_state,
    ):
        '''
        Why am I doing it this way? - one could pass 
        the update function pointfree to some multiprocessing
        generator.

        I took pains to prevent the large corpus and model_state 
        objects from being pulled into the scope of the update function.
        '''
        
        # 1. get the information we need from the sample
        sample_dict = self._convert_sample(sample)
        weights = sample_dict['weights']/subsample_rate
        alpha = np.ascontiguousarray(self.alpha[CS.get_name(corpus)])

        conditional_likelihood = \
            self._conditional_observation_likelihood(
                corpus, 
                model_state,
                **sample_dict
            )
        
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
            sample_dict=sample_dict,
        )

        svi_update = partial(
            self._apply_update,
            np.ascontiguousarray(gamma),
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
        '''
        In this case, each update function can be computed independently,
        and does not depend on any contextual evaluations.
        I added "get_update_fns" to the API for the locals model because 
        some update functions may depend on something computed ahead of time.
        '''

        samples = [
            (corpus, sample_name)
            for corpus in corpuses
            for sample_name in CS.list_samples(corpus)
        ]

        updates = (
            self._get_update_fn(
                CS.fetch_topic_compositions(corpus, sample_name),
                sample=CS.fetch_sample(corpus, sample_name),
                corpus=corpus,
                model_state=model_state,
                learning_rate=learning_rate,
                subsample_rate=subsample_rate,
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
        context_sum
    ):
        sample_dict = self._convert_sample(sample)
        conditional_likelihood = \
            self._conditional_observation_likelihood(
                corpus, 
                model_state,
                logsafe=False, # we want the actual likelihood, don't remove the max for numerical stability
                **sample_dict
            )
        
        weights = sample_dict['weights']
        contributions = np.ascontiguousarray(gamma/np.sum(gamma))

        # the context frequencies are missing dimensions that the observations have ...
        log_context_effect = self._get_log_context_effect(
            corpus,
            **sample_dict
        )

        # 1. figure out the missing dimensions
        missing_dims = dims_except_for(sample.dims, *corpus.regions.context_frequencies.dims)
        # 2. figure out the number of possible types of observations missing
        n_types = prod(corpus.sizes[dim] for dim in missing_dims)
        # 3. penalize the log context effect for the missing dimensions
        #log_context_effect -= np.log(n_types)
        
        # saturated
        y_sum = np.sum(weights)
        d_sat = weights @ np.log(weights) - y_sum * np.log(y_sum)
        # model
        d_fit = weights @ ( np.log(conditional_likelihood.T.dot(contributions)) - log_context_effect )
        # null
        d_null = -y_sum * np.log(context_sum) - y_sum*np.log(n_types)

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

        context_sums = {
            CS.get_name(corpus) : np.sum(
                corpus.regions.context_frequencies.data,
            )
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
            )
            for (corpus, sample_name) in samples
        )

        return deviance_fns

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
                self.init_locals( len(CS.list_samples(corpus)) ),
                dims=('component','sample')
            )
        )

    def spawn_sstats(self, corpus):
        return []
    

    @staticmethod
    def reduce_sparse_sstats(
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
        return model.reduce_sparse_sstats(
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
    