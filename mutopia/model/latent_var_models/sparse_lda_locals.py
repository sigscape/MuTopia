from xarray import DataArray
import numpy as np
from functools import partial, reduce
import sparse
from ..model_components.base import _svi_update_fn
from ..corpus_state import CorpusState as CS
from ...utils import dims_except_for
from ...corpus.interfaces import *
from math import prod
from .base import *
import warnings


class LDAUpdateSparse(LocalUpdate):


    ##
    # E-step functionality to satisfy the LocalUpdate interface
    ##
    @classmethod
    def _convert_sample(cls, sample, dtype=float):

        sample = sample.sparse_to_coo()
        weights = np.ascontiguousarray(
            sample.data.data.astype(dtype, copy=False)
        )

        idx_dict = dict(zip(
            tuple(sample.coords['obs_indices'].data),
            sample.indices.data,
        ))

        return dict(
            **idx_dict,
            weights=weights,
        )
    

    @classmethod
    def _unconvert_sample(cls, sample, sample_dict, data):

        dims = tuple(sample.dims)
        indices = np.array([sample_dict[k] for k in dims])
        shape = tuple(sample.sizes[k] for k in dims)

        sp_matrix = sparse.GCXS(
            sparse.COO(
                indices,
                data,
                shape=shape,
            ),
            compressed_axes=(dims.index('locus'),),
        )

        return DataArray(
            sp_matrix,
            dims=dims,
            attrs={
                k : v
                for k,v in sample.attrs.items() 
                if not k in ('shape', 'format')
            },
        )

    @classmethod
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


    @classmethod
    def _conditional_observation_likelihood(
        cls,
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

        offset = cls._get_log_context_effect(
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
            np.nan_to_num(np.exp(logp_X), nan=0.)
        )
    

    @staticmethod
    def _calc_sstats(
        alpha, 
        conditional_likelihood, 
        weights,
        gamma,
        batch_subsample=1.,
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
            'weighted_posterior' : weighted_posterior/batch_subsample, 
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
        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        weights = sample_dict['weights']/locus_subsample
        alpha = np.ascontiguousarray(self.alpha[CS.get_name(corpus)])

        conditional_likelihood = \
            self._conditional_observation_likelihood(
                corpus, 
                model_state,
                **sample_dict
            ).astype(self.dtype, copy=False)
        
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
            batch_subsample=batch_subsample,
        )

        svi_update = partial(
            self._apply_update,
            np.ascontiguousarray(gamma),
            update_fn=map_update,
            suffstat_fn=suffstat_fn,
            learning_rate=learning_rate,
        )

        return svi_update
    

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
                exposures_fn(corpus, sample_name),
                sample=sample,
                corpus=corpus,
                model_state=model_state,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
            )
            for corpus in corpuses
            for (sample_name, sample) in zip(
                CS.list_samples(corpus),
                corpus
            )
        )

        return (samples, updates)
    

    def posterior_assign_sample(
        self,
        sample,
        corpus,
        model_state,
        alpha=None,
    ):

        subsample_rate = (
                corpus
                .regions
                .context_frequencies
                .sum().item()/
                model_state.get_genome_size(corpus)
            )
        
        self.estep_iterations=10000
        self.difference_tol=5e-5

        if not alpha is None:
            raise ValueError('Cannot refit the model with a fixed alpha.')

        with ParContext(1) as par:
            _, update_fns = self.get_update_fns(
                (SampleCorpusFusion(CorpusInterface(corpus), sample),),
                model_state,
                parallel_context=par,
                exposures_fn=random_locals(np.random.RandomState(1776), self.n_components),
                locus_subsample=subsample_rate,
            )

        suffstats, _, ll = next(update_fns)()
        posterior = suffstats['weighted_posterior']/suffstats['weights']*subsample_rate

        return (ll, posterior)
    

    def marginal_ll_sample(
        self,
        sample,
        corpus,
        model_state,
        alpha=None,
        threads=1,
        sample_steps=1000,
        quiet=False,
        reps=100,
    ):
        # marginalize out gamma, and just return the mutation annotations.
        # For panel and exome data, there may be too few mutations to 
        # infer anything useful from gamma. We'll lower the variance of estimation
        # instead and just marginalize it out.

        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        alpha = np.ascontiguousarray(self.alpha[CS.get_name(corpus)] if alpha is None else alpha)

        # KxI - I=number of mutations, K=number of signatures
        conditional_likelihood = self._conditional_observation_likelihood(
            corpus,
            model_state,
            **sample_dict,
            logsafe=False,
        ).astype(self.dtype, copy=False)
        
        return AIS_marginal_ll(
            alpha,
            conditional_likelihood,
            sample_dict['weights'],
            threads=threads,
            inner_iters=sample_steps,
            outer_iters=reps,
            quiet=quiet,
        )


    def _apply_per_sample(
        self,
        fn,
        *,
        corpus,
        sample,
        model_state,
        gamma,
        **kw,
    ):

        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        
        conditional_likelihood = \
            self._conditional_observation_likelihood(
                corpus, 
                model_state,
                logsafe=False, # we want the actual likelihood, don't remove the max for numerical stability
                **sample_dict
            ).astype(self.dtype, copy=False)
        
        weights = sample_dict['weights']

        # the context frequencies are missing dimensions that the observations have ...
        log_context_effect = self._get_log_context_effect(
            corpus,
            **sample_dict
        )

        return fn(
            corpus,
            sample,
            model_state,
            gamma=gamma,
            conditional_likelihood=conditional_likelihood,
            weights=weights,
            log_context_effect=log_context_effect,
            sample_dict=sample_dict,
            **kw,
        )


    def _sample_deviance(
        self,
        corpus,
        sample,
        model_state,
        *,
        gamma,
        conditional_likelihood,
        weights,
        log_context_effect,
        context_sum,
        **kw,
    ):
        
        contributions = np.ascontiguousarray(gamma/np.sum(gamma))        

        # 1. figure out the missing dimensions
        missing_dims = dims_except_for(sample.dims, *corpus.regions.context_frequencies.dims)
        # 2. figure out the number of possible types of observations missing
        n_types = prod(corpus.sizes[dim] for dim in missing_dims)
        # 3. penalize the log context effect for the missing dimensions

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
    
        context_sums = {
            CS.get_name(corpus) : np.sum(
                corpus.regions.context_frequencies.data,
            )
            for corpus in corpuses
        }

        deviance_fns = (
            partial(
                self._apply_per_sample,
                self._sample_deviance,
                corpus=corpus,
                model_state=model_state,
                sample=sample,
                gamma=exposures_fn(corpus, sample_name),
                context_sum=context_sums[CS.get_name(corpus)],
            )
            for corpus in corpuses
            for sample_name, sample in zip(
                CS.list_samples(corpus),
                corpus
            )
        )

        return deviance_fns
    

    def _deviance_residuals(
        self,
        corpus,
        sample,
        model_state,
        *,
        gamma,
        conditional_likelihood,
        weights,
        log_context_effect,
        sample_dict,
    ):
        
        contributions = np.ascontiguousarray(gamma/np.sum(gamma))        
        y_sum = np.sum(weights)
        
        log_y = np.log(weights)
        log_pi_hat = ( np.log(conditional_likelihood.T.dot(contributions)) + np.log(y_sum) )
        pi_hat = np.exp(log_pi_hat)

        resid = np.sqrt( 2*(weights*(log_y - log_pi_hat) - weights + pi_hat) ) * np.sign(weights - pi_hat)

        return self._unconvert_sample(sample, sample_dict, resid)


    def get_residual_fns(
        self,
        corpuses,
        model_state,
        exposures_fn=CS.fetch_topic_compositions,
        *,
        parallel_context,
    ):
        residuals_fns = (
            partial(
                self._apply_per_sample,
                self._deviance_residuals,
                corpus=corpus,
                model_state=model_state,
                sample=sample,
                gamma=exposures_fn(corpus, sample_name),
            )
            for corpus in corpuses
            for sample_name, sample in zip(
                CS.list_samples(corpus),
                corpus
            )
        )

        return residuals_fns    


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
