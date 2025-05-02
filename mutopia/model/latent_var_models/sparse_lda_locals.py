from xarray import DataArray
import numpy as np
from functools import partial, reduce
import sparse
from ..model_components.base import _svi_update_fn
from .. import corpus_state as CS
from ...gtensor import dims_except_for
from math import prod
from .base import *
from ...utils import parallel_map
import warnings


class LDAUpdateSparse(LocalsModel):


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
        dataset,
        *,
        locus,
        **idx_dict,
    ):
        
        freqs = CS.get_regions(dataset).context_frequencies\
                    .transpose(...,'locus')
        
        idx_arrs = [idx_dict[dim] for dim in freqs.dims[:-1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return np.log(freqs.data[*idx_arrs, locus]) \
                + np.log(CS.get_regions(dataset).exposures.data[locus])


    @classmethod
    def _conditional_observation_likelihood(
        cls,
        dataset,
        factor_model,
        logsafe=True,
        *,
        weights,
        **idx_dict,
    ):

        ##
        # What's going on here: we have the normalized log mutation rate for each signature, configuration, context, locus.
        # For the mutations in this sample, we select over these axes.
        ##
        logp_normalizer = CS.fetch_normalizers(dataset)[:,None]

        offset = cls._get_log_context_effect(
            dataset,
            **idx_dict
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            logp_X = reduce(
                lambda x,y: x+y,
                (
                    model.predict_sparse(dataset, **idx_dict)
                    for model in factor_model.models.values()
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
        dataset,
        sample,
        factor_model,
    ):
        '''
        Why am I doing it this way? - one could pass 
        the update function pointfree to some multiprocessing
        generator.

        I took pains to prevent the large dataset and factor_model 
        objects from being pulled into the scope of the update function.
        '''
        
        # 1. get the information we need from the sample
        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        weights = sample_dict['weights']/locus_subsample
        alpha = np.ascontiguousarray(self.alpha[CS.get_name(dataset)])

        conditional_likelihood = \
            self._conditional_observation_likelihood(
                dataset, 
                factor_model,
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
    

    def _get_update_fns(
        self,
        corpuses,
        factor_model,
        learning_rate=1.,
        locus_subsample=1.,
        batch_subsample=1.,
        exposures_fn=CS.fetch_topic_compositions,
        par_context=None,
    ):
        '''
        In this case, each update function can be computed independently,
        and does not depend on any contextual evaluations.
        I added "get_update_fns" to the API for the locals model because 
        some update functions may depend on something computed ahead of time.
        '''

        samples = [
            (dataset, sample_name)
            for dataset in corpuses
            for sample_name in CS.list_samples(dataset)
        ]

        updates = (
            self._get_update_fn(
                exposures_fn(dataset, sample_name),
                sample=sample,
                dataset=dataset,
                factor_model=factor_model,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
            )
            for dataset in corpuses
            for (sample_name, sample) in CS.iter_samples(dataset)
        )

        return (samples, updates)
    

    def posterior_assign_sample(
        self,
        sample,
        dataset,
        factor_model,
        alpha=None,
        steps=5000,
        warmup=1000,
        seed=42,
    ):
        
        if alpha is None:
            alpha = self.alpha[CS.get_name(dataset)]

        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        
        conditional_likelihood = \
            self._conditional_observation_likelihood(
                dataset, 
                factor_model,
                logsafe=False, # we want the actual likelihood, don't remove the max for numerical stability
                **sample_dict
            ).astype(self.dtype, copy=False)
        
        weights = sample_dict['weights']

        p_z, gammas = gibbs_sample_posterior(
            alpha,
            conditional_likelihood,
            weights,
            steps=steps,
            warmup=warmup,
            seed=seed,
            quiet=True,
        )

        return p_z, gammas
    
    
    def marginal_ll_sample(
        self,
        dataset,
        sample,
        factor_model,
        alpha,
        steps=100000,
        seed=42,
    ):
        # marginalize out gamma, and just return the mutation annotations.
        # For panel and exome data, there may be too few mutations to 
        # infer anything useful from gamma. We'll lower the variance of estimation
        # instead and just marginalize it out.
        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        # KxI - I=number of mutations, K=number of signatures
        conditional_likelihood = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            **sample_dict,
            logsafe=False,
        )
        
        return AIS_marginal_ll(
            alpha,
            conditional_likelihood,
            sample_dict['weights'],
            steps=steps,
            seed=seed,
        )
    

    def _get_marginal_ll_fns(
        self,
        corpuses,
        factor_model,
        alpha=None,
        steps=100000,
        seed=42,
    ):

        marginal_ll_fns = (
            partial(
                self.marginal_ll_sample,
                dataset,
                sample,
                factor_model,
                alpha=self.alpha[CS.get_name(dataset)] if alpha is None else alpha,
                steps=steps,
                seed=seed,
            )
            for dataset in corpuses
            for (_, sample) in CS.iter_samples(dataset)
        )

        return marginal_ll_fns


    def _apply_per_sample(
        self,
        fn,
        *,
        dataset,
        sample,
        factor_model,
        gamma,
        **kw,
    ):

        sample_dict = self._convert_sample(sample, dtype=self.dtype)
        
        conditional_likelihood = \
            self._conditional_observation_likelihood(
                dataset, 
                factor_model,
                logsafe=False, # we want the actual likelihood, don't remove the max for numerical stability
                **sample_dict
            ).astype(self.dtype, copy=False)
        
        weights = sample_dict['weights']

        # the context frequencies are missing dimensions that the observations have ...
        log_context_effect = self._get_log_context_effect(
            dataset,
            **sample_dict
        )

        return fn(
            dataset,
            sample,
            factor_model,
            gamma=gamma,
            conditional_likelihood=conditional_likelihood,
            weights=weights,
            log_context_effect=log_context_effect,
            sample_dict=sample_dict,
            **kw,
        )


    def _deviance_sample(
        self,
        dataset,
        sample,
        factor_model,
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
        missing_dims = dims_except_for(sample.dims, *CS.get_regions(dataset).context_frequencies.dims)
        # 2. figure out the number of possible types of observations missing
        n_types = prod(dataset.sizes[dim] for dim in missing_dims)
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
    

    def _get_deviance_fns(
        self,
        corpuses,
        factor_model,
        exposures_fn=CS.fetch_topic_compositions,
        par_context=None,
    ):
    
        context_sums = {
            CS.get_name(dataset) : np.sum(
                CS.get_regions(dataset).context_frequencies.data,
            )
            for dataset in corpuses
        }

        deviance_fns = (
            partial(
                self._apply_per_sample,
                self._deviance_sample,
                dataset=dataset,
                factor_model=factor_model,
                sample=sample,
                gamma=exposures_fn(dataset, sample_name),
                context_sum=context_sums[CS.get_name(dataset)],
            )
            for dataset in corpuses
            for (sample_name, sample) in CS.iter_samples(dataset)
        )

        return deviance_fns
    

    def _deviance_residuals(
        self,
        dataset,
        sample,
        factor_model,
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


    def _get_residual_fns(
        self,
        corpuses,
        factor_model,
        exposures_fn=CS.fetch_topic_compositions,
        par_context=None,
    ):
        residuals_fns = (
            partial(
                self._apply_per_sample,
                self._deviance_residuals,
                dataset=dataset,
                factor_model=factor_model,
                sample=sample,
                gamma=exposures_fn(dataset, sample_name),
            )
            for dataset in corpuses
            for (sample_name, sample) in CS.iter_samples(dataset)
        )

        return residuals_fns    


    @staticmethod
    def reduce_model_sstats(
        model, 
        carry, 
        dataset, 
        **sample_sstats
    ):
        return model.reduce_sparse_sstats(
                carry, 
                dataset,
                **sample_sstats
            )
