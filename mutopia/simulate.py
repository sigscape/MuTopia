from mutopia.model import GtensorInterface as CS
from .utils import ParContext
from tqdm import tqdm
from sparse import COO
import xarray as xr
import numpy as np


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
