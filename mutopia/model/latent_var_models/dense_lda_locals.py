from xarray import DataArray
import numpy as np
from functools import partial
import warnings
from ..model_components.base import _svi_update_fn, logsafe_vector_matmul, logsumexp
from .. import gtensor_interface as CS
from ...gtensor import match_dims
from .base import *

logsafe_exp_transform = lambda x: np.nan_to_num(
    np.exp(x - np.nanmax(x)), nan=0.0, neginf=0.0
)

exp_transform = lambda x: np.nan_to_num(np.exp(x), nan=0.0, neginf=0.0)


@njit(
    "Tuple((float32, float32))(float32[::1,:], float32[::1], float32[::1], float32, float32[::1])",
    nogil=True
)
def _deviance_sample(
    conditional_likelihood, # K:: x I
    weights,
    Nk,
    context_sum,
    LOG_context_effects,
):
    # K @ K x I => I
    mutation_rate = Nk @ conditional_likelihood
    mutrate_norm = np.sum(mutation_rate)
    y_sum = np.sum(weights)

    # saturated
    d_sat = np.nansum(weights * np.log(weights)) - y_sum * np.log(y_sum)
    # model
    d_fit = np.nansum(weights * np.log(mutation_rate)) - y_sum * np.log(mutrate_norm)
    # null
    d_null = np.nansum(weights * LOG_context_effects) - y_sum * np.log(context_sum)

    return (
        d_sat - d_fit,
        d_sat - d_null,
    )


class LDAUpdateDense(LocalsModel):

    def _get_weights(self, sample):
        return np.ascontiguousarray(sample.X.load().data, dtype=self.dtype)

    def _conditional_observation_likelihood(
        self,
        dataset,
        factor_model,
        logsafe=True,
        with_context=True,
        par_context=None,
    ):

        ll = (
            factor_model._get_log_mutation_rate_tensor(
                dataset,
                par_context=par_context,
                with_context=with_context,
            )
            .transpose("component", *self.GT.observation_dims(dataset))
            .data
        )

        return np.ascontiguousarray(
            (logsafe_exp_transform if logsafe else exp_transform)(ll),
            dtype=self.dtype,
        )

    @staticmethod
    def _calc_sstats(
        alpha,
        conditional_likelihood,
        weights,
        Nk,
        batch_subsample=1.0,
        *,
        dims,
    ):

        args = (
            alpha,
            conditional_likelihood,
            weights,
            Nk,
        )

        weighted_posterior = calc_local_variables(*args)

        suffstats = {
            "weighted_posterior": DataArray(
                weighted_posterior / batch_subsample,
                dims=("component", *dims),
            ),
            "Nk": Nk,
        }

        return suffstats

    def _update_fn(
        self,
        Nk,
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
        weights = self._get_weights(sample) / locus_subsample
        alpha = self.to_contig(self.alpha[self.GT.get_name(dataset)])
        Nk = self.to_contig(Nk)

        args = (
            alpha,
            conditional_likelihood,
            weights,
        )

        map_estimate = iterative_update(
            *args,
            self.estep_iterations,
            self.difference_tol,
            Nk,
        )

        new_Nk = _svi_update_fn(Nk, map_estimate, learning_rate)

        suffstats = self._calc_sstats(
            *args,
            new_Nk,
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
        par_context=None,
    ):

        likelihoods = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            par_context=None,
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

    
    def _get_deviance_fns(
        self,
        dataset,
        factor_model,
        exposures_fn=None,
        par_context=None,
    ):
        """
        -> List[F() -> Tuple[float, float]]
        """

        sample_dims = self.GT.observation_dims(dataset)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            conditional_likelihood = (
                self._conditional_observation_likelihood(
                    dataset,
                    factor_model,
                    logsafe=False,
                    with_context=True,
                    par_context=None,
                )
            )

            LOG_context_effects = (
                match_dims(
                    np.log(self.GT.get_freqs(dataset))
                    + np.log(self.GT.get_exposures(dataset)),
                    **{d: dataset.sizes[d] for d in sample_dims},
                )
                .transpose(*sample_dims)
                .data
            )

            context_sum = np.nansum(np.exp(LOG_context_effects.data)).astype(self.dtype)

        
        K = conditional_likelihood.shape[0]
        conditional_likelihood = np.asfortranarray(
            conditional_likelihood.reshape(K, -1),
            dtype=self.dtype,
        )

        LOG_context_effects = self.to_contig(LOG_context_effects.reshape(-1))

        for sample_name, sample in self.GT.iter_samples(dataset):

            yield partial(
                _deviance_sample,
                conditional_likelihood,
                self.to_contig(self._get_weights(sample).reshape(-1)),
                self.to_contig((exposures_fn or self.GT.fetch_topic_compositions)(dataset, sample_name)),
                context_sum,
                LOG_context_effects,
            )


    def reduce_model_sstats(self, model, carry, dataset, **sample_sstats):
        return model.reduce_dense_sstats(
            carry[self.GT.get_name(dataset)], 
            dataset, 
            **sample_sstats
        )
