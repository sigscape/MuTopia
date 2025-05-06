from .model import gtensor_interface as CS
from .model.model_components.base import idx_array_to_design
from .utils import logger, ParContext
from .gtensor import check_structure, dims_except_for
from tqdm import tqdm
from sparse import COO
import xarray as xr
import numpy as np
from functools import reduce


COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def _fetch_contributions(corpus, exposures=None):
    if exposures is None:
        try:
            contributions = corpus["contributions"].data
        except KeyError:
            raise ValueError(
                "No exposures function provided and no contributions found in G-Tensor. "
                f"Please run `mutopia predict <model> {corpus.attrs['filename']}` first."
            )
    else:
        contributions = np.array(
            [exposures(corpus, sample_name) for sample_name in CS.list_samples(corpus)]
        )
    return contributions


def _get_state(model):
    return model.model_state_


def setup_mixture_model(
    sample,
    model,
    *corpuses,
    alpha=None,
    tau=None,
    steps=64000,
    seed=None,
):

    # extract sample dictionary using the first model
    sample_dict = _get_state(model).locals_model._convert_sample(sample)
    weights = sample_dict["weights"]

    likelihoods = []
    for corpus in corpuses:

        if not CS.has_corpusstate(corpus):
            state = _get_state(model)
            locals_model = state.locals_model
            corpus = model.setup_corpus(corpus)
            model.renormalize_model(corpus)

        likelihoods.append(
            locals_model._conditional_observation_likelihood(
                corpus,
                state,
                **sample_dict,
                logsafe=False,
            )
        )

    conditional_likelihood = np.vstack(likelihoods)
    del likelihoods

    fraction_map = (
        idx_array_to_design(
            np.array(
                [j for j, _ in enumerate(corpuses) for _ in range(model.n_components)]
            ),
            len(corpuses),
        )
        .todense()
        .T
    )

    component_map = np.vstack([np.eye(model.n_components) for _ in corpuses]).T

    log_conditional_likelihood = np.ascontiguousarray(np.log(conditional_likelihood).T)

    weights = np.ascontiguousarray(weights)

    args = (
        alpha,
        tau,
        fraction_map,
        component_map,
        log_conditional_likelihood,
        weights,
        steps,
        seed or 42,
    )

    pass


def simulate_from_model(
    model, corpus, seed=None, exposures=None, scale_num_mutations=1.0
):

    model_state = _get_state(model)
    random_state = np.random.RandomState(seed or 42)

    with ParContext(1) as par:
        lmrt = model_state._get_log_mutation_rate_tensor(
            corpus,
            parallel_context=par,
            with_context=True,
        )

    def _resample_sample(contributions):

        log_marginal_mutrate = model_state._log_marginalize_mutrate(
            lmrt,
            contributions,
        )

        n_mutations = int(
            scale_num_mutations * contributions.sum()
            - model_state.locals_model.alpha[CS.get_name(corpus)].sum()
        )

        p_vec = np.exp(log_marginal_mutrate).data.ravel()
        p_vec /= p_vec.sum()

        dense_obs = (
            random_state.multinomial(
                n_mutations,
                p_vec,
            )
            .reshape(log_marginal_mutrate.shape)
            .astype(np.float32)
        )

        return xr.DataArray(
            COO.from_numpy(dense_obs),
            dims=log_marginal_mutrate.dims,
        )

    contributions = _fetch_contributions(corpus, exposures=exposures)

    if not CS.has_corpusstate(corpus):
        raise ValueError(
            "The provided G-Tensor is not annotated. "
            f'Please run `mutopia predict <model> {corpus.attrs["filename"]}` with a trained model to annotate the corpus.'
        )

    X_sampled = xr.concat(
        [
            _resample_sample(sample_contributions)
            for sample_contributions in tqdm(
                contributions,
                desc="Resampling dataset",
                ncols=100,
            )
        ],
        dim="sample",
    )

    corpus_sampled = corpus.copy()
    corpus_sampled["X"] = X_sampled

    return corpus_sampled
