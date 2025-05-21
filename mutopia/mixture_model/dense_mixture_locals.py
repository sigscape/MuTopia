import numpy as np
from functools import reduce, partial
from .base import *
from ..model.model_components.base import _svi_update_fn
from ..model.latent_var_models import LDAUpdateDense

logsafe_exp_transform = lambda x: np.nan_to_num(
    np.exp(x - np.nanmax(x)), nan=0.0, neginf=0.0
)

exp_transform = lambda x: np.nan_to_num(np.exp(x), nan=0.0, neginf=0.0)


class DenseMixtureModel(LDAUpdateDense):

    def _source_log_likelihood(
        self,
        dataset,
        factor_model,
        par_context=None,
        with_context=True,
    ):
        return (
            factor_model._get_log_mutation_rate_tensor(
                dataset,
                par_context=par_context,
                with_context=with_context,
            )
            .transpose("component", *self.GT.observation_dims(dataset))
            .data
        )

    def _conditional_observation_likelihood(
        self,
        dataset,
        factor_model,
        logsafe=False,
        par_context=None,
        with_context=True,
    ):

        X = np.array(
            [
                self._source_log_likelihood(
                    ds,
                    factor_model,
                    par_context=par_context,
                    with_context=with_context,
                )
                for _, ds in self.GT.sources(dataset)
            ]
        )

        X = (logsafe_exp_transform if logsafe else exp_transform)(X)
        n_sources = len(self.GT.list_sources(dataset))
        trailing_shap = X.shape[2:]

        return np.ascontiguousarray(
            X.reshape(n_sources * self.n_components, *trailing_shap),
            dtype=self.dtype,
        )

    def _update_fn(
        self,
        Nk,
        learning_rate=1.0,
        locus_subsample=1.0,
        batch_subsample=1.0,
        *,
        dataset,
        sample,
        alpha,
        tau,
        component_map,  # Dx(D*K)
        fraction_map,
        #
        dims,
        conditional_likelihood,
    ):
        # 1. get the information we need from the sample
        weights = self._get_weights(sample) / locus_subsample

        args = (
            self.same_exposures,
            alpha,  # D*K
            tau,  # D
            component_map,  # Dx(D*K)
            fraction_map,  # Dx(D*K)
            conditional_likelihood,
            weights,
        )

        Nk = np.ascontiguousarray(Nk.ravel(), dtype=self.dtype)

        map_estimate = iterative_update(
            self.estep_iterations,
            self.difference_tol,
            *args,
            Nk,
        )

        Nk = _svi_update_fn(Nk, map_estimate, learning_rate)

        weighted_posterior = calc_local_variables(*args, Nk)  # (D*K, I)
        trailing_shape = weighted_posterior.shape[1:]

        n_sources = len(tau)

        weighted_posterior = weighted_posterior.reshape(
            -1, self.n_components, *trailing_shape
        )
        Nk = Nk.reshape(n_sources, self.n_components)

        suffstats = {
            "weighted_posterior": DataArray(
                weighted_posterior / batch_subsample,
                dims=("source", "component", *dims),
            ),
            "Nk": Nk,
        }

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

        conditional_likelihood = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            logsafe=True,
            with_context=False,
        )

        dims = self.GT.observation_dims(dataset)

        mixture_kw = self._get_mixture_kw(dataset)

        learning_rates = dict(
            learning_rate=learning_rate,
            locus_subsample=locus_subsample,
            batch_subsample=batch_subsample,
        )

        updates = (
            partial(
                self._update_fn,
                (exposures_fn or self.GT.fetch_topic_compositions)(
                    dataset, sample_name
                ),
                sample=sample,
                dataset=dataset,
                conditional_likelihood=conditional_likelihood,
                dims=dims,
                **mixture_kw,
                **learning_rates,
            )
            for (sample_name, sample) in self.GT.iter_samples(dataset)
        )

        return updates

    def reduce_model_sstats(
        self, model, carry, dataset, *, weighted_posterior, Nk, **sample_sstats
    ):
        for (name, ds), Nk_d, w_d in zip(
            self.GT.expand_datasets(dataset),
            Nk,
            weighted_posterior,
        ):

            model.reduce_dense_sstats(
                carry[name], ds, weighted_posterior=w_d, Nk=Nk_d, **sample_sstats
            )
