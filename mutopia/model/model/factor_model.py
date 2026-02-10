import numpy as np
import warnings
from functools import partial
from itertools import chain
from scipy.special import logsumexp
from functools import reduce
from collections import defaultdict
import xarray as xr
from mutopia.utils import parallel_map, parallel_gen
from .model_components import *
from .model_components.base import _svi_update_fn
from .latent_var_models import *
from .gtensor_interface import GtensorInterface


def overrided_by(default_fn):
    def inner(fn):
        def wrapped_fn(self, *args, **kwargs):
            if getattr(self, default_fn) is None:
                return fn(*args, **kwargs)
            else:
                return getattr(self, default_fn)(self, *args, **kwargs)

        return wrapped_fn

    return inner


class FactorModel:

    def __init__(
        self,
        GT: GtensorInterface,
        datasets,
        offsets_fn=None,
        predict_fn=None,
        **models,
    ):

        self.offsets_fn = offsets_fn
        self.predict_fn = predict_fn
        self.GT = GT

        self._models = {}

        for model_name, model in models.items():
            self._models[model_name] = model

        self._normalizers = {
            name: np.zeros(self.n_components)
            for name, _ in self.GT.expand_datasets(*datasets)
        }

        self._genome_size = {
            name: self.GT.get_genome_size(dataset)
            for name, dataset in self.GT.expand_datasets(*datasets)
        }

    def Mstep(
        self,
        datasets,
        sstats,
        offsets,
        par_context=None,
        learning_rate=1.0,
        use_parallel=True,
    ):
        datasets = self.GT.to_datasets(*datasets)

        update_fns = chain.from_iterable(
            (
                model.partial_fit(
                    k,
                    sstats[model_name + "_sstats"],
                    offsets[model_name + "_offsets"],
                    datasets,
                    learning_rate=learning_rate,
                )
                for k in range(self.n_components)
                for model_name, model in self.models.items()
            )
        )

        it = (
            parallel_gen(update_fns, par_context, ordered=False)
            if use_parallel
            else (fn() for fn in update_fns)
        )

        for _ in it:
            pass

    @property
    def n_components(self):
        return next(iter(self._models.values())).n_components

    @property
    def models(self):
        return self._models

    def __getitem__(self, model_name):
        return self._models[model_name]

    @property
    def requires_dims(self):
        return reduce(
            lambda x, y: x.union(y.requires_dims), self.models.values(), set(["sample"])
        )

    def get_normalizers(self, dataset, genome_size=None):

        if genome_size is None:
            genome_size = self.GT.get_genome_size(dataset)

        return self._normalizers[self.GT.get_name(dataset)] - np.log(
            genome_size / self.get_genome_size(dataset)
        )

    def get_genome_size(self, dataset):
        return self._genome_size[self.GT.get_name(dataset)]

    @overrided_by("predict_fn")
    def predict(self, k, dataset, with_context=True):
        """
        The difference between this method and the _get_propto_log_mutation_rate
        is that this method returns the log mutation rate for all
        models, not just those that require normalization.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            y_hat = reduce(
                lambda x, y: x + y,
                (
                    model.predict(k, dataset) for model in self.models.values()
                ),  # sum over models
                np.log(self.GT.get_exposures(dataset))
                + self.get_normalizers(dataset)[k],
            )
            if with_context:
                y_hat += np.log(self.GT.get_freqs(dataset))

        return y_hat

    def _get_log_mutation_rate_tensor(
        self,
        dataset,
        par_context=None,
        *,
        with_context=True,
    ):
        return xr.concat(
            parallel_map(
                (
                    partial(
                        self.predict,
                        k,
                        dataset,
                        with_context=with_context,
                    )
                    for k in range(self.n_components)
                ),
                par_context,
                ordered=True,
            ),
            dim="component",
        )

    @staticmethod
    def _log_marginalize_mutrate(log_mutrate_tensor, exposures):

        def logsafe_matmul(log_x, y):

            y = y.ravel()
            log_x = log_x.reshape(*log_x.shape[:-2], -1)

            alpha = log_x.max()
            return alpha + np.log(
                np.dot(np.nan_to_num(np.exp(log_x - alpha), nan=0.0, copy=False), y)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return xr.apply_ufunc(
                logsafe_matmul,
                log_mutrate_tensor,
                exposures,
                input_core_dims=[["source", "component"], ["source", "component"]],
            )

    def _log_component_posterior(self, log_mutrate_tensor, exposures):
        log_X_tild = self._log_marginalize_mutrate(log_mutrate_tensor, exposures)
        return log_mutrate_tensor - log_X_tild

    def get_sstats_dict(self, datasets):
        # model -> dataset -> component -> sstats
        return {
            model_name
            + "_sstats": {
                name: model.spawn_sstats(dataset)
                for name, dataset in self.GT.expand_datasets(*datasets)
            }
            for model_name, model in self.models.items()
        }

    def get_exp_offsets_dict(
        self,
        datasets,
        *,
        par_context=None,
    ):
        """
        We want `all_offsets` to be a dictionary of <model_name> -> <dataset> -> <k> -> <offset>
        """
        all_offsets = defaultdict(lambda: defaultdict(dict))
        normalizers = {
            name: np.zeros_like(self.get_normalizers(dataset))
            for name, dataset in self.GT.expand_datasets(*datasets)
        }

        args = [
            (k, dataset)
            for k in range(self.n_components)
            for _, dataset in self.GT.expand_datasets(*datasets)
        ]

        offset_fns = (
            partial(self.offsets_fn or self._get_exp_offsets_k_c, self, k, dataset)
            for k, dataset in args
        )

        for (k, dataset), (norm, exp_offsets) in zip(
            args, parallel_map(offset_fns, par_context)
        ):
            for model_name, _offsets in exp_offsets.items():

                all_offsets[model_name + "_offsets"][self.GT.get_name(dataset)][
                    k
                ] = _offsets

                normalizers[self.GT.get_name(dataset)][k] = norm

        return all_offsets, normalizers

    @overrided_by("offsets_fn")
    def _get_exp_offsets_k_c(self, k, dataset):
        model_predictions = {
            model_name: model.predict(k, dataset)
            for model_name, model in self.models.items()
            if model.requires_normalization
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            log_mutation_rate = reduce(
                lambda x, y: x + y,
                model_predictions.values(),  # sum over models
                np.log(self.GT.get_exposures(dataset))
                + np.log(self.GT.get_freqs(dataset)),  # start with the background rates
            )

            """
            To get the exp offsets for a model, you need to know the contribution of
            every other model to the log mutation rate - this way the other models are treated
            as fixed effects.

            IF the model "requires normalization" - or is part of the mutation rate estimation,
            then we need to subtract the model's prediction from the total prediction. This
            yields the sum of predictions from every other model.
            """
            exp_offsets = defaultdict(
                lambda: None,
                {
                    model_name: model.get_exp_offset(
                        log_mutation_rate - model_predictions[model_name], dataset
                    )
                    for model_name, model in self.models.items()
                    if model.requires_normalization
                },
            )

            return (-logsumexp(log_mutation_rate.data), exp_offsets)

    def update_normalizers(
        self, datasets, normalizers, learning_rate=1.0, subsample_rate=1.0
    ):
        genome_size = self.GT.get_genome_size(datasets[0])

        for name, ds in self.GT.expand_datasets(*datasets):

            curr = self._normalizers[name]

            self._normalizers[name][:] = _svi_update_fn(
                curr,
                np.log(genome_size / self.get_genome_size(ds)) + normalizers[name],
                learning_rate,
                parameter_name=f"normalizers_{name}",
            )

    def init_normalizers(self, datasets, par_context=None):
        _, self._normalizers = self.get_exp_offsets_dict(
            datasets, par_context=par_context
        )

    def format_component(self, k, normalization="global"):
        return np.exp(
            reduce(
                lambda x, y: x + y,
                (
                    model.format_component(k, normalization=normalization)
                    for model in self.models.values()
                ),
            )
        )
    
    @property
    def has_interactions(self) -> bool:
        return self.models["context_model"].has_interactions

    def format_interactions(self, k):
        return self.models["context_model"].get_interaction_summary(k)
