from mutopia.analysis import load_model
from mutopia.model import GtensorInterface as CS
from mutopia.modalities import SBS
from mutopia.modalities.sbs._sbs_clustering import *
from mutopia.modalities.sbs._sbs_ingestion import featurize_mutations
from mutopia.modalities.sbs._sbs_nucdata import *
from mutopia.gtensor import disk_interface as disk
from mutopia.gtensor import lazy_load
from mutopia.model.model.latent_var_models.base import iterative_update, calc_local_variables
from ..genome_utils.bed12_utils import stream_bed12
import click
from functools import partial
from typing import Union, TextIO
import numpy as np
from scipy.special import logsumexp
import xarray as xr
from sparse import COO
import pandas as pd
import sys
import os
import subprocess
from types import SimpleNamespace


@click.group("SBS commands")
def sbs():
    pass


@sbs.command("background-rate")
@click.argument("vcf_files", nargs=-1)
@click.option(
    "-g",
    "--genome-file",
    required=True,
    type=click.Path(exists=True),
    help="The reference genome fasta file.",
)
@click.option("-chr", "--chr-prefix", default="")
@click.option(
    "--pass-only/--no-pass-only",
    type=bool,
    default=True,
    help="Whether to only ingest passing mutations",
)
def marginal_rate(
    vcf_files: list,
    genome_file: str,
    chr_prefix="",
    pass_only=True,
):
    get_marginal_mutation_rate(
        genome_file,
        *vcf_files,
        chr_prefix=chr_prefix,
        pass_only=pass_only,
    )


@sbs.command("cluster")
@click.argument("vcf_file")
@click.option(
    "-g",
    "--mutation-rate-bedgraph",
    required=True,
    type=click.Path(exists=True),
    help="The mutation rate bedgraph file.",
)
@click.option("-chr", "--chr-prefix", default="")
@click.option(
    "--pass-only/--no-pass-only",
    type=bool,
    default=True,
    help="Whether to only ingest passing mutations",
)
@click.option(
    "-name",
    "--sample-name",
    type=str,
    default=None,
    help="VCFS ONLY: Name of sample in multi-sample VCF.",
)
def _cluster_vcfs(
    sample_name=None,
    pass_only=True,
    *,
    vcf_file,
    chr_prefix,
    mutation_rate_bedgraph,
):
    query_fn = partial(
        stream_passed_SNVs,
        sample=sample_name,
        pass_only=pass_only,
        chr_prefix=chr_prefix,
    )

    cluster_vcf(
        mutation_rate_bedgraph=mutation_rate_bedgraph,
        query_fn=partial(query_fn, vcf_file),
        vcf_file=vcf_file,
        chr_prefix=chr_prefix,
    )


@sbs.command("annotate-vcf")
@click.argument(
    'model_file',
    type=click.Path(exists=True),
)
@click.argument(
    'dataset_file',
    type=click.Path(exists=True),
)
@click.argument(
    'sample_file',
    type=click.Path(exists=True),
)
@click.option(
    '-o',
    '--output',
    type=click.File('w'),
    metavar='FILE',
    default=sys.stdout,
    help='Output file to write the annotated VCF to. If not provided, will write to stdout.',
)
@click.option(
    '-m',
    '--mutation-rate-file',
    type=str,
    help='VCFS ONLY: File containing mutation rates',
)
@click.option(
    '-chr',
    '--chr-prefix',
    type=str,
    default='',
    help='VCFS ONLY: Prefix to add to chromosome names',
)
@click.option(
    '--pass-only/--no-pass-only',
    type=bool,
    default=True,
    help='VCFS ONLY: Whether to only ingest passing mutations',
)
@click.option(
    '-w',
    '--weight-col',
    type=str,
    default=None,
    help='VCFS ONLY: Column to use for mutation weights',
)
@click.option(
    '-name',
    '--sample-name',
    type=str,
    default=None,
    help='VCFS ONLY: Name of sample in multi-sample VCF.',
)
@click.option(
    '--cluster/--no-cluster',
    type=bool,
    default=True,
    help='VCFS ONLY: Whether to cluster the mutations in the sample, requires a mutation rate file.',
)
@click.option(
    '--skip-sort/--no-skip-sort',
    type=bool,
    default=False,
    help='Whether to skip sorting the VCF file. This will fail if the VCF is not in lexigraphical sorted order (chr1, chr10, ...).',
)
@click.option(
    '-fa',
    '--fasta',
    type=click.Path(exists=True),
    default=None,
    help='The reference genome fasta file.',
)
@click.option(
    '-@',
    '--threads',
    type=click.IntRange(1, 64),
    default=1,
    help='Number of threads to use.',
)
@click.option(
    "--white-list",
    type=click.Path(exists=True),
    help="File containing a list of whitelisted mutations.",
)
def mutation_ll(
    sample_file: str,
    model_file: str,
    dataset_file: str,
    threads : int = 1,
    output : TextIO = sys.stdout,
    white_list : Union[str, None] = None,
    fasta : Union[str, None] = None,
    **ingest_kwargs
):
    """Compute per-mutation log posterior probabilities under the model.

    For each mutation in the input VCF, computes the log posterior probability
    of assignment to each model component (signature). Results are written as
    new INFO fields in the output VCF.

    Example usage:

    \b
    mutopia sbs annotate-vcf \\
        /path/to/model.pkl \\
        /path/to/dataset.nc \\
        /path/to/sample.vcf \\
        -m /path/to/mutation-rate.bedgraph.gz \\
        -o /path/to/sample_annotated.vcf

    The “whitelist” determines what regions of the genome should be considered
    for modeling. Training is conducted on whole genomes, but perhaps you
    collected mutations using an exome capture kit or a panel. If you provide
    the assayed regions as a whitelist BED file, MuTopia adjusts the per-region
    exposure terms to reflect only the covered fraction of each window, improving
    signature assignment accuracy.
    """
    click.echo("Loading model ...", err=True)
    model = load_model(model_file)
    locals_model = model.locals_model_
    factor_model = model.factor_model_

    attrs = disk.read_attrs(dataset_file)
    regions_file = disk.fetch_regions_path(dataset_file)
    num_regions = sum(1 for _ in stream_bed12(regions_file))
    locus_coords = disk.read_coords(dataset_file)["locus"]
    fasta = fasta or attrs["fasta_file"]
    click.echo(f"Dataset has {num_regions} regions, {len(locus_coords)} after coord subsetting.", err=True)

    # --- Ingest mutations ---------------------------------------------------
    click.echo("Ingesting mutations ...", err=True)
    mut_ids, raw_coords, weights = featurize_mutations(
        sample_file,
        regions_file,
        fasta,
        mutation_rate_file=ingest_kwargs.get('mutation_rate_file'),
        chr_prefix=ingest_kwargs.get('chr_prefix', ''),
        pass_only=ingest_kwargs.get('pass_only', True),
        weight_col=ingest_kwargs.get('weight_col'),
        sample_name=ingest_kwargs.get('sample_name'),
        cluster=ingest_kwargs.get('cluster', True),
        skip_sort=ingest_kwargs.get('skip_sort', False),
    )
    click.echo(f"Ingested {len(mut_ids)} mutations.", err=True)

    # Build sparse sample DataArray and subset to dataset loci
    sample_arr = xr.DataArray(
        COO(
            raw_coords, weights,
            shape=(2, len(MUTOPIA_ORDER), num_regions),
            has_duplicates=False, sorted=True, prune=False, cache=False,
        ),
        dims=("configuration", "context", "locus"),
    ).isel(locus=locus_coords)

    # After isel, the sparse library re-sorts nnz entries into canonical order.
    # Rebuild mut_ids aligned to the post-isel COO so z_logpost columns match.
    locus_coords_arr = np.asarray(locus_coords)
    coord_to_mut_id = {
        (int(raw_coords[0, i]), int(raw_coords[1, i]), int(raw_coords[2, i])): mut_ids[i]
        for i in np.where(np.isin(raw_coords[-1, :], locus_coords_arr))[0]
    }
    coo = sample_arr.ascoo().data
    original_loci = locus_coords_arr[coo.coords[2]]
    mut_ids = [
        coord_to_mut_id[(int(c), int(ctx), int(l))]
        for c, ctx, l in zip(coo.coords[0], coo.coords[1], original_loci)
    ]

    sample = SimpleNamespace(X=sample_arr)
    click.echo(f"Built sparse sample, {coo.nnz} nonzero entries.", err=True)

    # --- Load dataset and apply whitelist ------------------------------------
    click.echo("Loading dataset ...", err=True)
    dataset = lazy_load(dataset_file)

    if white_list is not None:
        exposures = _compute_whitelist_exposures(regions_file, white_list)
        dataset["Regions/exposures"] = xr.DataArray(
            exposures[locus_coords].astype(np.float32), dims=("locus",)
        )
        click.echo(
            f"Whitelist applied: {(exposures[locus_coords] > 0).sum()}/{len(locus_coords)} "
            f"regions have nonzero exposure.",
            err=True,
        )

    click.echo("Setting up dataset (this may take a while) ...", err=True)
    dataset = model.setup_corpus(dataset, threads=threads)

    if white_list is not None:
        click.echo("Recomputing normalizers for updated exposures ...", err=True)
        factor_model.init_normalizers([dataset])

    # --- Compute per-mutation z-posteriors ------------------------------------
    click.echo("Computing per-mutation posteriors ...", err=True)
    sample_dict = locals_model._convert_sample(sample)
    cond = locals_model._conditional_observation_likelihood(
        dataset, factor_model,
        logsafe=True, renormalize=False,
        sample_dict=sample_dict,
    )

    # E-step: estimate per-component counts, then compute phi matrix
    alpha = np.ascontiguousarray(locals_model.get_alpha(dataset), dtype=np.float32)
    obs_weights = np.ascontiguousarray(locals_model._get_weights(sample))
    cond_c = np.ascontiguousarray(cond)
    Nk = iterative_update(
        alpha, cond_c, obs_weights,
        locals_model.estep_iterations, locals_model.difference_tol,
        alpha.copy(),
    )
    phi = calc_local_variables(alpha, cond_c, obs_weights, Nk)  # (K, I)
    z_logpost = np.log(phi + 1e-30) - logsumexp(np.log(phi + 1e-30), axis=0, keepdims=True)

    # --- Pretty-print Nk estimates to stderr ---------------------------------
    Nk_total = Nk.sum()
    Nk_frac = Nk / Nk_total if Nk_total > 0 else Nk
    click.echo("Estimated component fractions:", err=True)
    max_name_len = max(len(n) for n in model.component_names)
    sorted_pairs = sorted(zip(model.component_names, Nk_frac), key=lambda x: -x[1])
    for name, frac in sorted_pairs:
        click.echo(f"  {name:<{max_name_len}}  {frac:.4f}", err=True)

    # --- Write annotated VCF -------------------------------------------------
    nk_header_str = ",".join(
        f"{name.replace(' ', '_')}={nk_val:.4f}"
        for name, nk_val in zip(model.component_names, Nk)
    )

    annotations_df = pd.DataFrame(
        {f"logp_{name.replace(' ', '_')}": z_logpost[k, :]
         for k, name in enumerate(model.component_names)},
        index=pd.MultiIndex.from_tuples(mut_ids, names=["CHROM", "POS"]),
    ).reset_index()

    click.echo("Annotating VCF ...")
    transfer_annotations_to_vcf(
        annotations_df,
        vcf_file=sample_file,
        chr_prefix=ingest_kwargs.get('chr_prefix', ''),
        description={
            f"logp_{name.replace(' ', '_')}":
                f"Log posterior probability that this mutation was generated by component {name}."
            for name in model.component_names
        },
        output=output,
        extra_headers=[
            f'##mutopia_Nk=<Description="Estimated per-component mutation counts for this sample",{nk_header_str}>',
        ],
    )


def _compute_whitelist_exposures(regions_file: str, whitelist_file: str) -> np.ndarray:
    """
    Compute per-region exposure as the fraction covered by the whitelist.

    Uses ``bedtools coverage -a <regions> -b <whitelist> -split`` which reports,
    for each BED12 region, the fraction of its exonic bases covered by the
    whitelist intervals. Regions fully outside the whitelist get exposure 0.
    """
    result = subprocess.run(
        ["bedtools", "coverage", "-a", regions_file, "-b", whitelist_file, "-split"],
        capture_output=True, text=True, check=True,
    )
    # bedtools coverage appends 4 columns: count, bases_covered, region_length, fraction
    # The fraction (last column) is exactly what we need.
    fractions = []
    for line in result.stdout.strip().split("\n"):
        if line:
            fractions.append(float(line.rstrip().split("\t")[-1]))
    return np.array(fractions, dtype=np.float32) + 1e-6  # add small pseudocount to avoid zero exposures