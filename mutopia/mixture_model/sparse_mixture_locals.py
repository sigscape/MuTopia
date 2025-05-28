import numpy as np
import warnings
from functools import reduce, partial
from .base import *
from ..model.model_components.base import _svi_update_fn
from ..model.latent_var_models import LDAUpdateSparse


class SparseMixtureModel(LDAUpdateSparse):

    def _source_log_likelihood(
        self,
        dataset,
        factor_model,
        *,
        sample_dict,
    ):
        ##
        # What's going on here: we have the normalized log mutation rate for each signature, configuration, context, locus.
        # For the mutations in this sample, we select over these axes.
        ##
        logp_normalizer = factor_model.get_normalizers(dataset)[:, None]

        offset = self._get_log_context_effect(dataset, **sample_dict)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logp_X = reduce(
                lambda x, y: x + y,
                (
                    model.predict_sparse(dataset, **sample_dict)
                    for model in factor_model.models.values()
                ),
                logp_normalizer + offset,
            )

        return logp_X

    def _conditional_observation_likelihood(
        self,
        dataset,
        factor_model,
        logsafe=False,
        *,
        sample_dict,
    ):
        X = np.array(
            [
                self._source_log_likelihood(
                    ds,
                    factor_model,
                    sample_dict=sample_dict,
                )
                for _, ds in self.GT.sources(dataset)
            ]
        )

        if logsafe:
            X = X - np.max(X)

        X = np.nan_to_num(np.exp(X), nan=0.0)

        n_sources = len(self.GT.list_sources(dataset))
        return np.ascontiguousarray(
            X.reshape(n_sources * self.n_components, -1),
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
        factor_model,
        sample,
        alpha,
        tau,
        component_map,  # Dx(D*K)
        fraction_map,
    ):

        weights = self._get_weights(sample) / locus_subsample
        sample_dict = self._convert_sample(sample)

        conditional_likelihood = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            sample_dict=sample_dict,
            logsafe=True,
        )

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

        n_sources = len(tau)
        weighted_posterior = weighted_posterior.reshape(
            n_sources, self.n_components, -1
        )
        Nk = Nk.reshape(n_sources, self.n_components)

        suffstats = {
            **sample_dict,
            "weighted_posterior": weighted_posterior / batch_subsample,
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
        """
        In this case, each update function can be computed independently,
        and does not depend on any contextual evaluations.
        I added "get_update_fns" to the API for the locals model because
        some update functions may depend on something computed ahead of time.
        """

        mixture_kw = self._get_mixture_kw(dataset)

        updates = (
            partial(
                self._update_fn,
                (exposures_fn or self.GT.fetch_topic_compositions)(
                    dataset, sample_name
                ),
                sample=sample,
                dataset=dataset,
                factor_model=factor_model,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
                **mixture_kw,
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

            model.reduce_sparse_sstats(
                carry[name], ds, weighted_posterior=w_d, Nk=Nk_d, **sample_sstats
            )
