from .modalities._sbs_nucdata import CONTEXTS
from .model.corpus_state import CorpusState as CS
from .model.model_components.base import idx_array_to_design
from .utils import dims_except_for, check_structure, logger, ParContext
from .genome_utils.fancy_iterators import RegionOverlapComparitor
from .genome_utils.bed12_utils import unstack_regions
from tqdm import tqdm
from sparse import COO
import xarray as xr
import os
from pyfaidx import Fasta
import numpy as np
from functools import reduce


COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def _revcomp(seq):
    return "".join(reversed([COMPLEMENT[nuc] for nuc in seq]))


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
        corpus["component_distributions"]
    except KeyError:
        corpus = model.annot_component_distributions(corpus)

    contributions = _fetch_contributions(corpus, exposures)
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

    component_dists = subset["component_distributions"]
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
            nucleotides.extend(seq[1:-1])

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
    bar=True,
):
    check_structure(corpus)

    X_emp = reduce(
        lambda x,y : x + y,
        (
            corpus.fetch_sample(sample_name).ascoo() 
            for sample_name in tqdm(
                CS.list_samples(corpus)[1:],
                desc="Reducing samples",
            )
        ),
        corpus.fetch_sample(CS.list_samples(corpus)[0]).asdense()
    )

    X_emp = X_emp.asdense() if X_emp.is_sparse() else X_emp

    logger.info('Added key: "empirical_marginal"')
    corpus["empirical_marginal"] = (
        X_emp / corpus.regions.context_frequencies
    ).fillna(0.0)

    logger.info('Added key: "empirical_locus_marginal"')
    corpus["empirical_locus_marginal"] = (
        X_emp.sum(dim=dims_except_for(X_emp.dims, "locus")) / corpus.regions.length
    ).fillna(0.0).astype(np.float32)

    return corpus


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
            np.array([
                j
                for j, _ in enumerate(corpuses)
                for _ in range(model.n_components)
            ]),
            len(corpuses),
        )
        .todense()
        .T
    )

    component_map = np.vstack([
        np.eye(model.n_components)
        for _ in corpuses
    ]).T

    log_conditional_likelihood = np.ascontiguousarray(
        np.log(conditional_likelihood).T
    )

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
    model, 
    corpus, 
    seed=None,
    exposures=None,
    scale_num_mutations=1.
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
            scale_num_mutations*contributions.sum() - model_state.locals_model.alpha[CS.get_name(corpus)].sum()
        )

        p_vec = np.exp(log_marginal_mutrate).data.ravel()
        p_vec /= p_vec.sum()

        dense_obs = (
            random_state.multinomial(
                n_mutations,
                p_vec,
            )
            .reshape(
                log_marginal_mutrate.shape
            )
            .astype(np.float32)
        )

        return xr.DataArray(
            COO.from_numpy(dense_obs),
            dims=log_marginal_mutrate.dims,
        )
    
    contributions = _fetch_contributions(corpus, exposures=exposures)

    if not CS.has_corpusstate(corpus):
        raise ValueError(
            'The provided G-Tensor is not annotated. '
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
        dim='sample'
    )

    corpus_sampled = corpus.copy()
    corpus_sampled["X"] = X_sampled

    return corpus_sampled
