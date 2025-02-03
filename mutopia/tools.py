from .modalities._sbs_nucdata import CONTEXTS
from .genome_utils.fancy_iterators import RegionOverlapComparitor
from .genome_utils.bed12_utils import unstack_regions
from .model.corpus_state import CorpusState as CS
from .model.model_components.base import idx_array_to_design
from .model.latent_var_models.base import mixture_update_step
from .utils import dims_except_for, check_structure, logger
import os
from pyfaidx import Fasta
import numpy as np
from functools import partial

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def _revcomp(seq):
    return "".join(reversed([COMPLEMENT[nuc] for nuc in seq]))


def deepscan_neutral_matagenesis(
    model,
    corpus,
    exposures=None,
    *,
    chrom,
    start,
    end,
    fasta,
):

    try:
        corpus.varm["component_distributions"]
    except KeyError:
        corpus = model.annot_component_distributions(corpus)

    if exposures is None:
        try:
            contributions = corpus["contributions"].data
        except KeyError:
            raise ValueError(
                "No exposures function provided and no contributions found in corpus"
            )
    else:
        contributions = np.array(
            [exposures(corpus, sample_name) for sample_name in CS.list_samples(corpus)]
        )

    contributions /= contributions.sum(axis=1, keepdims=True)

    # 1. Find the regions that overlap with the specified region
    regions = zip(
        corpus.regions.chrom,
        corpus.regions.start,
        corpus.regions.end,
        range(len(corpus.regions.chrom)),
    )
    roi = RegionOverlapComparitor(chrom, start, end)
    regions = filter(lambda region: RegionOverlapComparitor(*region) == roi, regions)
    regions_idx = [region[3] for region in regions]

    if len(regions_idx) == 0:
        raise ValueError("No regions found that overlap with the specified region")

    subset = corpus.isel(locus=regions_idx)

    # 2. Unstack the regions
    start, end, idx = unstack_regions(
        subset.coords["locus"].values,
        os.path.join(
            os.path.dirname(corpus.attrs["filename"]), corpus.attrs["regions_file"]
        ),
        np.arange(len(subset.coords["locus"])),
    )

    component_dists = subset.varm["component_distributions"]
    region_mutation_rate = (
        component_dists.isel(locus=idx)
        .sum(dim=dims_except_for(component_dists.dims, "locus", "component", "context"))
        .transpose("locus", "component", "context")
        .values
    )

    nloci, ncomp, *_ = region_mutation_rate.shape
    region_mutation_rate = (
        region_mutation_rate
        .reshape(nloci, ncomp, 32, 3)
        .sum(axis=-1)
    )

    # 3. Extract the mutation rates per sample
    positions = []
    mutrates = []
    nucleotides = []

    with Fasta(fasta) as fa:

        for _start, _end, _mutrates in zip(start, end, region_mutation_rate):

            seq = fa[chrom][(_start - 1) : (_end + 2)].seq.upper()
            nucleotides.extend(seq[1:-2])

            for pos_m1 in range(len(seq) - 3 + 1):

                context = seq[pos_m1 : pos_m1 + 3]
                if context[1] in "AG":
                    context = _revcomp(context)

                site_component_rates = _mutrates.T[CONTEXTS.index(context)]  # 32xK -> K
                sample_mutation_rates = (
                    contributions @ site_component_rates
                )  # NxK @ K -> N

                positions.append(_start + pos_m1)
                mutrates.append(sample_mutation_rates)

    return (np.array(positions), np.array(nucleotides), np.array(mutrates))


def annot_empirical_marginal(
    corpus,
):
    check_structure(corpus)

    X_emp = corpus.X.sum(dim=("sample",))
    X_emp = X_emp.asdense() if X_emp.is_sparse() else X_emp

    logger.info('Added key to varm: "empirical_marginal"')
    corpus.varm["empirical_marginal"] = (
        X_emp / corpus.regions.context_frequencies
    ).fillna(0.0)

    logger.info('Added key to varm: "empirical_locus_marginal"')
    corpus.varm["empirical_locus_marginal"] = (
        X_emp.sum(dim=dims_except_for(X_emp.dims, "locus")) / corpus.regions.length
    )

    return corpus


def _get_state(model):
    return model.model_state_


def _infer_mixture_model(
    tau=None,
    alpha=None,
    seed=None,
    max_iter=10000,
    tol=5e-5,
    *,
    update_fn,
    model_pairs,
    conditional_likelihood,
    weights,
    fraction_map,
):

    # 1. initialize the priors
    if tau is None:
        tau = np.ones(len(model_pairs), order="C")
    elif not len(tau) == len(model_pairs):
        raise ValueError("Length of tau must match the number of models")

    if alpha is None:
        # (D*K,)
        alpha = np.ascontiguousarray(
            np.concatenate(
                [
                    _get_state(model).locals_model.alpha[CS.get_name(corpus)]
                    for model, corpus in model_pairs
                ]
            )
        )
    elif not len(alpha) == sum(model.n_components for model, _ in model_pairs):
        raise ValueError(
            "Length of alpha must match the number of components in the models"
        )

    update_fn = partial(
        update_fn,
        conditional_likelihood=conditional_likelihood,
        weights=weights,
        fraction_map=fraction_map,
        tau=tau,
        alpha=alpha,
    )

    # 2. initialize the variational parameters
    random_state = np.random.RandomState(seed or 42)

    delta = random_state.gamma(100.0, 1.0 / 100.0, size=(len(model_pairs),))  # D

    gamma = np.ascontiguousarray(
        np.concatenate(
            [
                random_state.gamma(100.0, 1.0 / 100.0, size=(model.n_components,))
                for model, _ in model_pairs
            ]
        )
    )  # (D*K,)

    # 3. run the update steps until convergence
    scores = []
    for i in range(max_iter):

        gamma, delta, elbo = update_fn(gamma, delta)
        scores.append(elbo)

        if i % 10 == 0:
            logger.info(f"Iteration {i+1} - ELBO: {elbo:.3f}")

        if i > 0 and np.abs(scores[-1] - scores[-2]) < tol:
            break
    
    # 4. summarize the results
    changepoints = np.cumsum([0,] + [model.n_components for model, _ in model_pairs])
    contributions={}
    for i, (_, corpus) in enumerate(model_pairs):
        gamma_hat = gamma[changepoints[i]:changepoints[i+1]]
        contributions[CS.get_name(corpus)] = gamma_hat / gamma_hat.sum()

    return {
        'estimated_fractions' : delta/delta.sum(),
        'contributions' : contributions,
        'scores' : scores,
    }


def setup_mixture_model(
    sample,
    *model_pairs,
):

    # extract sample dictionary using the first model
    sample_dict = _get_state(model_pairs[0][0]).locals_model._convert_sample(sample)

    weights = np.ascontiguousarray(sample_dict["weights"])

    likelihoods = []
    for model, corpus in model_pairs:

        state = _get_state(model)
        locals_model = state.locals_model

        model.setup_corpus(corpus)

        likelihoods.append(
            locals_model._conditional_observation_likelihood(
                corpus,
                state,
                **sample_dict,
                logsafe=False,
            )
        )

    likelihoods = np.ascontiguousarray(np.vstack(likelihoods))

    fraction_map = (
        idx_array_to_design(
            np.array(
                [
                    j
                    for j, (state, _) in enumerate(model_pairs)
                    for _ in range(state.n_components)
                ]
            ),
            len(model_pairs),
        )
        .todense()
        .T
    )

    return partial(
        _infer_mixture_model,
        conditional_likelihood=likelihoods,
        weights=weights,
        fraction_map=fraction_map,
        model_pairs=model_pairs,
    )


def simulate_from_model(model, corpus, seed=None):
    pass
