from xarray import DataArray
import numpy as np
from functools import partial
import warnings
from ..model_components.base import _svi_update_fn
from .. import gtensor_interface as CS
from ...gtensor import match_dims
from .base import *


class LDAUpdateDense(LocalsModel):

    def _convert_sample(self, sample):
        return np.ascontiguousarray(sample.X.load().data, dtype=self.dtype)

    def _conditional_observation_likelihood(
        self,
        dataset,
        factor_model,
        logsafe=True,
        *,
        par_context,
    ):

        logsafe_exp_transform = lambda x: np.nan_to_num(
            np.exp(x - np.nanmax(x)), nan=0.0, neginf=0.0
        )
        exp_transform = lambda x: np.nan_to_num(np.exp(x), nan=0.0, neginf=0.0)

        lcol = (
            factor_model._get_log_mutation_rate_tensor(
                dataset,
                par_context=par_context,
                with_context=False,
            )
            .transpose("component", *self.GT.observation_dims(dataset))
            .data
        )

        return np.ascontiguousarray(
            (logsafe_exp_transform if logsafe else exp_transform)(lcol)
        )

    @staticmethod
    def _calc_sstats(
        alpha,
        conditional_likelihood,
        weights,
        gamma,
        batch_subsample=1.0,
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
        # elbo = bound(*args, weighted_posterior)

        suffstats = {
            "weighted_posterior": DataArray(
                weighted_posterior / batch_subsample,
                dims=("component", *dims),
            ),
            "gamma": gamma,
        }

        return suffstats

    def _update_fn(
        self,
        gamma,
        learning_rate=1.0,
        locus_subsample=1.0,
        batch_subsample=1.0,
        *,
        dataset,
        sample,
        conditional_likelihood,
        dims,
    ):
        # 1. get the information we need from the sample
        weights = self._convert_sample(sample) / locus_subsample
        alpha = np.ascontiguousarray(self.alpha[self.GT.get_name(dataset)])
        gamma = np.ascontiguousarray(gamma, dtype=self.dtype)

        args = (
            alpha,
            conditional_likelihood,
            weights,
        )

        map_estimate = iterative_update(
            *args,
            self.estep_iterations,
            self.difference_tol,
            gamma,
        )

        new_gamma = _svi_update_fn(gamma, map_estimate, learning_rate)

        suffstats = self._calc_sstats(
            *args,
            new_gamma,
            batch_subsample=batch_subsample,
            dims=dims,
        )

        return suffstats

    def _get_update_fns(
        self,
        dataset,
        factor_model,
        learning_rate=1.0,
        locus_subsample=1.0,
        batch_subsample=1.0,
        exposures_fn=None,
        *,
        par_context,
    ):

        likelihoods = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            par_context=par_context,
            logsafe=True,
        )

        updates = (
            partial(
                self._update_fn,
                (exposures_fn or self.GT.fetch_topic_compositions)(dataset, sample_name),
                sample=sample,
                dataset=dataset,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
                conditional_likelihood=likelihoods,
                dims=self.GT.observation_dims(dataset),
            )
            for (sample_name, sample) in self.GT.iter_samples(dataset)
        )

        return updates

    def _deviance_sample(
        self,
        sample,
        factor_model,
        *,
        gamma,
        log_mutrate_tensor,
        context_sum,
        context_effects,
    ):

        weights = self._convert_sample(sample)
        gamma = np.ascontiguousarray(gamma)

        log_marginal_mutrate = factor_model._log_marginalize_mutrate(
            log_mutrate_tensor, gamma
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

    def _get_deviance_fns(
        self,
        dataset,
        factor_model,
        exposures_fn=None,
        *,
        par_context,
    ):
        """
        -> List[F() -> Tuple[float, float]]
        """

        sample_dims = self.GT.observation_dims(dataset)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            log_conditional_likelihood = factor_model._get_log_mutation_rate_tensor(
                dataset,
                with_context=True,
            ).transpose("component", *sample_dims)

            context_effects = (
                match_dims(
                    np.log(self.GT.get_regions(dataset).context_frequencies)
                    + np.log(self.GT.get_regions(dataset).exposures),
                    **{d: dataset.sizes[d] for d in sample_dims},
                )
                .transpose(*sample_dims)
                .data
            )

            context_sum = np.nansum(np.exp(context_effects.data))

        for sample_name, sample in self.GT.iter_samples(dataset):

            yield partial(
                self._deviance_sample,
                sample=sample,
                factor_model=factor_model,
                gamma=(exposures_fn or self.GT.fetch_topic_compositions)(dataset, sample_name),
                context_sum=context_sum,
                log_mutrate_tensor=log_conditional_likelihood,
                context_effects=context_effects,
            )

    def reduce_model_sstats(self, model, carry, dataset, **sample_sstats):
        return model.reduce_dense_sstats(
            carry[self.GT.get_name(dataset)], 
            dataset, 
            **sample_sstats
        )
