from numba import njit
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import xarray as xr
import numpy as np
from ..model.model_components.base import idx_array_to_design, _svi_update_fn
from ..model.latent_var_models.base import *
from .mixture_interface import MixtureInterface
from os.path import basename


def reshape_output(fn):
    @wraps(fn)
    def wrapper(*args):
        likelihood = args[-3]
        tensor_dim = likelihood.shape
        out = fn(*args)
        return reshape_same_mem(out, tensor_dim)

    return wrapper


def flatten_tensor_for_update(fn):
    @wraps(fn)
    def wrapper(*args):
        likelihood, weights = args[-3], args[-2]
        K = likelihood.shape[0]
        remainder_shape = likelihood.shape[1:]
        assert (
            remainder_shape == weights.shape
        ), "conditional_likelihood and weights must have the same trailing shape"

        return fn(
            *args[:-3],
            reshape_same_mem(likelihood, (K, -1)),
            reshape_same_mem(weights, (-1,)),
            *args[-1:],
        )

    return wrapper


"""
Mixture model inference functions
"""


@njit(
    "float32[::1](float32[::1], float32[::1], float32[:,:], float32[:,:], float32[::1])",
    nogil=True,
)
def shared_exposures_prior(alpha, tau, component_map, fraction_map, Nk):
    exp_Elog_prior = np.exp(
        psivec32(alpha + (Nk @ component_map.T)) @ component_map
        + psivec32(tau + (Nk @ fraction_map.T)) @ fraction_map
    )
    return exp_Elog_prior


@njit(
    "float32[::1](float32[::1], float32[::1], float32[:,:], float32[:,:], float32[::1])",
    nogil=True,
)
def different_exposures_prior(alpha, tau, component_map, fraction_map, Nk):
    exp_Elog_prior = np.exp(
        psivec32(alpha + Nk)
        - psivec32((alpha + Nk) @ fraction_map.T) @ fraction_map
        + psivec32(tau + (Nk @ fraction_map.T)) @ fraction_map
    )
    return exp_Elog_prior


@njit(
    "float32[::1](bool, float32[::1], float32[::1], float32[:,:], float32[:,:], float32[:,::1], float32[::1], float32[::1])",
    nogil=True,
)
def _update_step(
    same_exposure,
    alpha,  # D*K
    tau,  # D
    component_map,  # Kx(D*K)
    fraction_map,  # Dx(D*K)
    conditional_likelihood,  # D*K x I
    weights,  # I,
    Nk,
):

    # D*K
    prior_args = (alpha, tau, component_map, fraction_map, Nk)

    if same_exposure:
        exp_Elog_prior = shared_exposures_prior(*prior_args)
    else:
        exp_Elog_prior = different_exposures_prior(*prior_args)

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    Nk = exp_Elog_prior * (conditional_likelihood @ X_div_X_tild)

    return Nk


@flatten_tensor_for_update
@njit(
    "float32[::1](int64, float32, bool, float32[::1], float32[::1], float32[:,:], float32[:,:], float32[:,::1], float32[::1], float32[::1])",
    nogil=True,
)
def iterative_update(
    iters,
    tol,
    same_exposure,
    alpha,  # D*K
    tau,  # D
    component_map,  # Kx(D*K)
    fraction_map,  # Dx(D*K)
    conditional_likelihood,
    weights,
    Nk,  # move gamma to the end so we can curry the function
):
    for _ in range(iters):  # inner e-step loop

        old_Nk = Nk.copy()
        Nk = _update_step(
            same_exposure,
            alpha,  # D*K
            tau,  # D
            component_map,  # Kx(D*K)
            fraction_map,  # Dx(D*K)
            conditional_likelihood,
            weights,
            Nk,
        )

        if (np.abs(Nk - old_Nk) / np.sum(Nk)).sum() < tol:
            break

    return Nk


@reshape_output
@flatten_tensor_for_update
@njit(
    "float32[:,::1](bool, float32[::1], float32[::1], float32[:,:], float32[:,:], float32[:,::1], float32[::1], float32[::1])",
    nogil=True,
)
def calc_local_variables(
    same_exposure,
    alpha,  # D*K
    tau,  # D
    component_map,  # Kx(D*K)
    fraction_map,  # Dx(D*K)
    conditional_likelihood,
    weights,
    Nk,
):
    prior_args = (alpha, tau, component_map, fraction_map, Nk)

    if same_exposure:
        exp_Elog_prior = shared_exposures_prior(*prior_args)
    else:
        exp_Elog_prior = different_exposures_prior(*prior_args)

    X_div_X_tild = np.where(
        weights > 0.0,
        weights / (exp_Elog_prior @ conditional_likelihood),
        np.float32(0.0),
    )

    phi_matrix = np.outer(exp_Elog_prior, X_div_X_tild) * conditional_likelihood

    return phi_matrix


class MixtureModelBase(LocalsModel):

    same_exposures = True

    def __init__(
        self,
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
        self.GT = MixtureInterface()


    def _init_locals_sample(self, sample,*, alpha, tau, **_):

        N = sample.X.sum().data.item()
        
        prior = tau[:,None] * alpha[None,:]  # D x K
        p_hat = prior/prior.sum(axis=1, keepdims=True)  # normalize

        output_shape = p_hat.shape
        concentration = p_hat.ravel() * 1000
        # Sample from Dirichlet distribution and scale by N
        Nk = np.random.dirichlet(concentration) * N

        return Nk.reshape(output_shape) # G*D*K


    def init_locals(self, dataset):

        mixture_kw = self._get_mixture_kw(dataset)

        return self.to_contig(
            np.array(
                [
                    self._init_locals_sample(sample, **mixture_kw)
                    for _, sample in self.GT.iter_samples(dataset)
                ]
            )
        )

    def prepare_corpusstate(self, dataset):
        return dict(
            topic_compositions=DataArray(
                self.init_locals(dataset),
                dims=("sample", "source", "component"),
            ),
        )

    def _get_mixture_kw(self, dataset):
        raise NotImplementedError()

    def Mstep(self, datasets, learning_rate=1):
        raise NotImplementedError()


class SharedExposuresMixtureModel(MixtureModelBase):
    """
    This model assumes that all samples share the same exposure to the mixture components.
    """

    same_exposures = True

    def __init__(
        self,
        datasets,
        prior_alpha=1.0,
        prior_tau=1.0,
        **kw,
    ):
        super().__init__(datasets, prior_alpha, prior_tau, **kw)

        self.alpha = {
            self.GT.get_name(dataset): np.ones(self.n_components, dtype=self.dtype)
            * prior_alpha
            for dataset in datasets
        }

        self.tau = {
            self.GT.get_name(dataset): np.ones(
                len(self.GT.list_sources(dataset)), dtype=self.dtype
            )
            * prior_tau
            for dataset in datasets
        }

    def _get_mixture_kw(self, dataset):

        alpha = self.to_contig(self.alpha[self.GT.get_name(dataset)])
        tau = self.to_contig(self.tau[self.GT.get_name(dataset)])

        n_sources = len(self.GT.list_sources(dataset))
        n_comps = self.n_components

        component_map = self.to_contig(
            idx_array_to_design(np.tile(np.arange(n_comps), n_sources), n_comps)
            .todense()
            .T
        )

        fraction_map = self.to_contig(
            idx_array_to_design(np.repeat(np.arange(n_sources), n_comps), n_sources)
            .todense()
            .T
        )

        return {
            "tau": tau,
            "alpha": alpha,
            "component_map": component_map,
            "fraction_map": fraction_map,
        }

    def Mstep(self, datasets, learning_rate=1):

        for dataset in datasets:

            name = self.GT.get_name(dataset)

            locals = self.GT.fetch_locals(dataset)
            Nks = locals.sum(dim="source").transpose("sample", ...).data

            alpha0 = self.alpha[name]
            self.alpha[name] = _svi_update_fn(
                alpha0, update_alpha(alpha0, np.array(Nks)), learning_rate
            )

            Nds = locals.sum(dim="component").transpose("sample", ...).data

            tau = self.tau[name]
            self.tau[name] = _svi_update_fn(tau, update_alpha(tau, Nds), learning_rate)

        return self


class DifferentExposuresMixtureModel(MixtureModelBase):

    def __init__(
        self,
        datasets,
        prior_alpha=1.0,
        prior_tau=1.0,
        **kw,
    ):
        super().__init__(datasets, prior_alpha, prior_tau, **kw)

        self.alpha = {
            self.GT.get_name(dataset): np.ones(self.n_components, dtype=self.dtype)
            * prior_alpha
            for dataset in datasets
        }

        self.tau = {
            self.GT.get_name(dataset): np.ones(
                len(self.GT.list_sources(dataset)), dtype=self.dtype
            )
            * prior_tau
            for dataset in datasets
        }

    def _get_mixture_kw(self, dataset):
        raise NotImplementedError()

    def Mstep(self, datasets, learning_rate=1):

        Nks_by_source = {}
        for name, dataset in self.GT.expand_datasets(*datasets):

            Nks = (
                self.GT.fetch_val(dataset, "topic_compositions")
                .transpose("sample", ...)
                .data
            )

            source_name = basename(name)

            if not source_name in Nks_by_source:
                Nks_by_source[source_name] = Nks
            else:
                Nks_by_source[source_name] = np.concatenate(
                    (Nks_by_source[source_name], Nks), axis=0
                )

        for source_name, alpha0 in self.alpha.items():

            self.alpha[source_name] = _svi_update_fn(
                alpha0, update_alpha(alpha0, Nks_by_source[source_name]), learning_rate
            )

        GT = self.GT

        for dataset in datasets:

            name = GT.get_name(dataset)

            Nds = (
                xr.concat(
                    [
                        GT.fetch_val(
                            GT.fetch_source(dataset, source), "topic_compositions"
                        )
                        for source in GT.list_sources(dataset)
                    ],
                    dim="source",
                )
                .sum(dim="component")
                .assign_coords(source=GT.list_sources(dataset))
                .transpose("sample", "source")
                .values
            )

            tau = self.tau[name]
            self.tau[name] = _svi_update_fn(tau, update_alpha(tau, Nds), learning_rate)

        return self
