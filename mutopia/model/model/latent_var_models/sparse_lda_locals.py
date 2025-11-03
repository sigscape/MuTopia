import numpy as np
from functools import partial, reduce
from mutopia.gtensor import dims_except_for
from ..model_components.base import _svi_update_fn
from math import prod
from .base import *
import warnings
# ignore PerformanceWarnings from numba
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


class WrapsFactorModel:
    """
    A bit of a clunky hack - we need to temporarily spoof the get_normalizers
    method on the factor model to return the normalizers
    for a specific dataset. For SNVs, this is necessary to make the loss work
    for some reason?
    """

    def __init__(self, factor_model, normalizers):
        self.factor_model = factor_model
        self.normalizers = normalizers

    def __getattr__(self, name):
        return getattr(self.factor_model, name)

    def get_normalizers(self, dataset):
        return self.normalizers[self.factor_model.GT.get_name(dataset)]


@njit(
    "Tuple((float32, float32))(float32[::1], float32[:,:], float32[::1], float32, float32[::1], int64)",
    nogil=True,
)
def _sample_deviance(
    weights,  # I
    conditional_likelihood,  # K x I
    Nk,  # K
    context_sum,
    log_context_effect,
    n_types,
):
    # saturated
    y_sum = np.sum(weights)
    d_sat = weights @ np.log(weights) - y_sum * np.log(y_sum)

    # model
    contributions = Nk / np.sum(Nk)
    d_fit = weights @ (
        np.log(contributions @ conditional_likelihood) - log_context_effect
    )

    # null
    d_null = -y_sum * np.log(context_sum) - y_sum * np.log(n_types)

    return (
        d_sat - d_fit,
        d_sat - d_null,
    )


class LDAUpdateSparse(LocalsModel):

    def _get_weights(self, sample):
        return np.ascontiguousarray(sample.X.data.data, dtype=self.dtype)

    @classmethod
    def _convert_sample(cls, sample):
        sample = sample.X.sparse_to_coo()

        idx_dict = dict(
            zip(
                tuple(sample.coords["obs_indices"].data),
                sample.indices.data,
            )
        )

        return idx_dict

    """@classmethod
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
            compressed_axes=(dims.index("locus"),),
        )

        return DataArray(
            sp_matrix,
            dims=dims,
            attrs={
                k: v for k, v in sample.attrs.items() if not k in ("shape", "format")
            },
        )"""

    def _get_log_context_effect(
        self,
        dataset,
        *,
        locus,
        **idx_dict,
    ):

        freqs = self.GT.get_freqs(dataset).transpose(..., "locus")

        idx_arrs = [idx_dict[dim] for dim in freqs.dims[:-1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return np.log(freqs.data[*idx_arrs, locus]) + np.log(
                self.GT.get_exposures(dataset).data[locus]
            )

    def _conditional_observation_likelihood(
        self,
        dataset,
        factor_model,
        logsafe=True,
        renormalize=False,
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

        if logsafe:
            logp_X = logp_X - logp_X.max()

        return np.ascontiguousarray(
            np.nan_to_num(np.exp(logp_X), nan=0.0), dtype=self.dtype
        )

    @staticmethod
    def _calc_sstats(
        alpha,
        conditional_likelihood,
        weights,
        Nk,
        batch_subsample=1.0,
        *,
        sample_dict,
    ):

        args = (
            alpha,
            conditional_likelihood,
            weights,
            Nk,
        )

        weighted_posterior = calc_local_variables(*args)

        suffstats = {
            **sample_dict,
            "weighted_posterior": weighted_posterior / batch_subsample,
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
        factor_model,
    ):
        # 1. get the information we need from the sample
        weights = self._get_weights(sample) / locus_subsample
        sample_dict = self._convert_sample(sample)

        conditional_likelihood = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            sample_dict=sample_dict,
        )

        alpha = np.ascontiguousarray(self.alpha[self.GT.get_name(dataset)])
        Nk = np.ascontiguousarray(Nk.ravel(), dtype=self.dtype)

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
            sample_dict=sample_dict,
            batch_subsample=batch_subsample,
        )

        return suffstats
    
    '''def _predict(self, dataset, factor_model, threads=1, estep_iterations=100000, difference_tol=0.00001):
        _, norms = factor_model.get_exp_offsets_dict((dataset,))

        factor_model = WrapsFactorModel(
            factor_model,
            norms,
        )
        return super()._predict(dataset, factor_model, threads, estep_iterations, difference_tol)'''

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
                (exposures_fn or self.GT.fetch_topic_compositions)(
                    dataset, sample_name
                ),
                sample=sample,
                dataset=dataset,
                factor_model=factor_model,
                learning_rate=learning_rate,
                locus_subsample=locus_subsample,
                batch_subsample=batch_subsample,
            )
            for (sample_name, sample) in self.GT.iter_samples(dataset)
        )

        return updates

    def _deviance_sample(
        self,
        dataset,
        sample,
        factor_model,
        *,
        Nk,
        context_sum,
        n_types,
        **kw,
    ):
        weights = self._get_weights(sample)
        sample_dict = self._convert_sample(sample)

        # K x I::
        conditional_likelihood = self._conditional_observation_likelihood(
            dataset,
            factor_model,
            logsafe=False,  # we want the actual likelihood, don't remove the max for numerical stability
            renormalize=True,  # we want a proper likelihood distribution
            sample_dict=sample_dict,
        )

        # the context frequencies are missing dimensions that the observations have ...
        log_context_effect = self.to_contig(
            self._get_log_context_effect(dataset, **sample_dict)
        )
        Nk = self.to_contig(Nk)

        # 3. penalize the log context effect for the missing dimensions
        return _sample_deviance(
            weights,
            np.asfortranarray(conditional_likelihood),
            Nk,
            context_sum,
            log_context_effect,
            n_types,
        )

    def _get_deviance_fns(
        self,
        dataset,
        factor_model,
        exposures_fn=None,
        par_context=None,
    ):

        _, norms = factor_model.get_exp_offsets_dict((dataset,))

        factor_model = WrapsFactorModel(
            factor_model,
            norms,
        )

        context_sum = np.sum(self.GT.get_freqs(dataset).data)
        alpha = np.ascontiguousarray(self.get_alpha(dataset))

        # 1. figure out the missing dimensions
        missing_dims = dims_except_for(
            self.GT.observation_dims(dataset), *self.GT.get_freqs(dataset).dims
        )
        # 2. figure out the number of possible types of observations missing
        n_types = prod(dataset.sizes[dim] for dim in missing_dims)

        deviance_fns = (
            partial(
                self._deviance_sample,
                dataset=dataset,
                factor_model=factor_model,
                sample=sample,
                Nk=(exposures_fn or self.GT.fetch_topic_compositions)(
                    dataset, sample_name
                ).ravel() + alpha,
                context_sum=context_sum,
                n_types=n_types,
            )
            for (sample_name, sample) in self.GT.iter_samples(dataset)
        )

        return deviance_fns

    def reduce_model_sstats(self, model, carry, dataset, **sample_sstats):
        model.reduce_sparse_sstats(
            carry[self.GT.get_name(dataset)], dataset, **sample_sstats
        )
