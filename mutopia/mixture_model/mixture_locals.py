import numpy as np
import warnings
from functools import reduce, partial
from .base import *
from ..model.model_components.base import _svi_update_fn, idx_array_to_design
from ..model.latent_var_models import LDAUpdateSparse


class MixtureModel(LocalsModel):

    def __init__(
        self,
        GT, # gtensor_interface
        datasets,
        prior_alpha=1.0,
        prior_tau=1.0,
        estep_iterations=1000,
        difference_tol=5e-5,
        dtype=np.float32,
        *,
        random_state,
        n_components,
        **kw,
    ):
        self.estep_iterations = estep_iterations
        self.difference_tol = difference_tol
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype
        self.GT = GT

        self.alpha = {
            name: np.ones(n_components, dtype=dtype) * prior_alpha
            for name, _ in self.GT.expand_datasets(*datasets)
        }

        self.tau = np.ones(len(self.GT.to_datasets(*datasets)), dtype=dtype) * prior_tau


class SparseMixtureModel(MixtureModel, LDAUpdateSparse):

    def _get_mixture_kw(self, dataset):

        alpha = np.concatenate([self.alpha[name] for name, _ in self.GT.expand_datasets(dataset)])
        alpha = np.ascontiguousarray(alpha, dtype=self.dtype)
        tau = np.ascontiguousarray(self.tau, dtype=self.dtype)

        n_sources = len(self.GT.list_sources(dataset))

        fraction_map = (
            idx_array_to_design(np.repeat(np.arange(n_sources), self.n_components), n_sources)
            .todense()
            .T
        )
        fraction_map = np.ascontiguousarray(fraction_map, dtype=self.dtype)

        return {
            "alpha": alpha,
            "tau": tau,
            "fraction_map": fraction_map,
        }
    

    def _source_log_conditional_observation_likelihood(
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
        logp_normalizer = self.GT.fetch_normalizers(dataset)[:, None]

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
        logsafe=False, # for API consistency, not used
        *,
        sample_dict,
    ):
        X = np.array([
            self._source_conditional_observation_likelihood(
                ds,
                factor_model,
                sample_dict=sample_dict,
            )
            for _, ds in self.GT.sources(dataset)
        ])

        if logsafe:
            X = X - np.max(X)

        X = np.nan_to_num(np.exp(X), nan=0.0), 

        n_sources = len(self.GT.list_sources(dataset))
        return np.ascontiguousarray(
            X.reshape(n_sources * self.n_components, -1),
            dtype=self.dtype,
        )
    
    def _calc_sstats(
        self,
        alpha,  # D*K
        tau,  # D
        fraction_map,  # Dx(D*K)
        conditional_likelihood,
        weights,
        Nk,
        batch_subsample=1.0,
        *,
        sample_dict,
    ):

        args = (
            alpha,  # D*K
            tau,  # D
            fraction_map,  # Dx(D*K)
            conditional_likelihood,
            weights,
        )

        weighted_posterior = calc_local_variables(*args, Nk) # (D*K, I)

        n_sources = len(tau)
        weighted_posterior = weighted_posterior.reshape(n_sources, self.n_components, -1) 
        Nk = Nk.reshape(n_sources, self.n_components)

        suffstats = {
            **sample_dict,
            "weighted_posterior": weighted_posterior / batch_subsample,
            "gamma": Nk,
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
        factor_model,
        sample,
        alpha,
        tau,
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
            alpha,  # D*K
            tau,  # D
            fraction_map,  # Dx(D*K)
            conditional_likelihood,
            weights,
        )

        Nk = np.ascontiguousarray(Nk.ravel(), dtype=self.dtype)

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
            sample_dict=sample_dict,
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
        """
        In this case, each update function can be computed independently,
        and does not depend on any contextual evaluations.
        I added "get_update_fns" to the API for the locals model because
        some update functions may depend on something computed ahead of time.
        """

        updates = (
            partial(
                self._update_fn,
                (exposures_fn or self.GT.fetch_topic_compositions)(dataset, sample_name),
                sample=sample,
                dataset=dataset,
                factor_model=factor_model,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
                **self._get_mixture_kw(dataset),
            )
            for (sample_name, sample) in self.GT.iter_samples(dataset)
        )

        return updates

    
    def reduce_model_sstats(
        self,
        model, carry, dataset, 
        *,
        weighted_posterior, 
        gamma, 
        **sample_sstats
    ):
        for (name, ds), gamma_d, w_d in zip(
            self.GT.expand_datasets(dataset),
            gamma,
            weighted_posterior,
        ):
            
            model.reduce_sparse_sstats(
                carry[name],
                ds,
                weighted_posterior=w_d,
                gamma=gamma_d,
                **sample_sstats
            )