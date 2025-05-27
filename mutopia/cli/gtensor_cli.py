import click
from typing import *
import os
import numpy as np
import mutopia.ingestion as ingest
from mutopia.gtensor import *
import netCDF4 as nc
from shutil import copyfile

import mutopia.gtensor.disk_interface as disk
from ..gtensor import *
from ..ingestion.gtf_parsing import query_gtf
from ..ingestion import gene_features
from ..modalities import *
from ..genome_utils.bed12_utils import stream_bed12
from ..utils import FeatureType, logger
from ..dtypes import get_mode_config
from .core import *
from .pipeline_tasks import run_pipeline


@click.group("G-tensor commands")
def gtensor_cli():
    pass

@gtensor_cli.command("compose", short_help="Run the G-Tensor construction pipeline")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "-w",
    "--workers",
    type=click.IntRange(1, 100),
    default=1,
    help="Number of workers to use for the pipeline",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    is_flag=True,
    help="Whether to run the pipeline in dry-run mode",
)
def _run_pipeline(*args, **kwargs):
    run_pipeline(*args, **kwargs)


@gtensor_cli.command("split", short_help="Split a G-Tensor into training and test sets")
@click.argument("filename", type=click.Path(exists=True))
@click.argument("test_contigs", nargs=-1, type=str, required=True)
@click.option(
    "-min",
    "--min-region-size",
    type=click.IntRange(1, 100000000),
    default=5,
)
def _train_test_split(*args, **kwargs):
    train_test_split(*args, **kwargs)


@gtensor_cli.command("slice", short_help="Slice a G-Tensor by region")
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("output", type=click.Path(writable=True))
@click.option(
    "-r",
    "--query-region",
    required=True,
    multiple=True,
    type=(str, int, int),
)
def slice_loci(
    *,
    output,
    dataset,
    query_region: list[tuple[str, int, int]],
):
    dataset = lazy_load(dataset)
    dataset = slice_regions(dataset, *query_region, lazy=True)
    disk.write_dataset(dataset, output, bar=True)


@gtensor_cli.command("create", short_help="Create a new G-Tensor")
@click.option(
    "-cut",
    "--cutout-regions",
    type=(str, click.Path(exists=True)),
    multiple=True,
    help="Regions to cut out of the genome.",
)
@click.option(
    "-n",
    "--name",
    type=str,
    required=True,
    help="Name to assign the G-Tensor",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True),
    required=True,
    help="Output file to write the G-Tensor to",
)
@click.option(
    "-dtype",
    "--dtype",
    type=str,
    required=True,
    help="Modality to use for the G-Tensor",
)
@click.option(
    "-g",
    "--genome-file",
    type=click.Path(exists=True),
    required=True,
    help="Genome file to use for the regions",
)
@click.option(
    "-v",
    "--blacklist-file",
    type=click.Path(exists=True),
    required=True,
    help="File containing loci to exclude from the regions",
)
@click.option(
    "-fa",
    "--fasta-file",
    type=click.Path(exists=True),
    required=True,
    help="Fasta file to use for calculating context frequencies",
)
@click.option(
    "-s",
    "--region-size",
    type=click.IntRange(1, 1000000),
    default=10000,
    help="Size of the regions to create",
)
@click.option(
    "-min",
    "--min-region-size",
    type=click.IntRange(1, 1000000),
    default=25,
    help="Minimum size of the regions to create",
)
@click.option(
    "-b",
    "--base-regions",
    type=click.Path(exists=True),
    default=None,
    help="File containing regions to use as a base for the G-Tensor.\nIf none provided, uniform regions of size `region-size` will be created.",
)
def _create_gtensor(*args, **kwargs):
    create_gtensor(*args, **kwargs)


@gtensor_cli.command("convert", short_help="Convert a G-Tensor to a different modality")
@click.argument(
    "input",
    type=click.Path(exists=True),
)
@click.argument(
    "output",
    type=click.Path(writable=True),
)
@click.option(
    "-dtype",
    "--dtype",
    type=str,
    required=True,
    help="Modality to convert the corpus to",
)
@click.option(
    "-fa",
    "--fasta-file",
    type=click.Path(exists=True),
    required=True,
    help="Fasta file to use for calculating context frequencies",
)
def convert(
    *,
    input,
    dtype: str,
    output: str,
    fasta_file: str,
):

    old_dataset = disk.load_dataset(input, with_samples=False)
    attrs = old_dataset.attrs

    input_regions_file = attrs["regions_file"]
    output_regions_file = output + ".regions.bed"
    copyfile(input_regions_file, output_regions_file)

    modality = get_mode_config(dtype)

    context_freqs = modality.get_context_frequencies(
        regions_file=output_regions_file,
        fasta_file=fasta_file,
    )

    chrom, start, end = list(
        zip(
            *((s.chromosome, s.start, s.end) for s in stream_bed12(output_regions_file))
        )
    )

    new_dataset = GTensor(
        modality,
        name=attrs["name"],
        chrom=chrom,
        start=start,
        end=end,
        context_frequencies=context_freqs,
        exposures=old_dataset.regions.exposures.values,
        dtype=dtype,
    )

    for k, v in attrs.items():
        if not k in ("regions_file", "dtype"):
            new_dataset.attrs[k] = v

    new_dataset.attrs["regions_file"] = output_regions_file

    # transfer features from old to new
    new_dataset = new_dataset.merge(old_dataset.sections["Features"])

    write_dataset(new_dataset, output)


@gtensor_cli.command("set-attr", short_help="Set attributes on a G-Tensor")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.option(
    "-set",
    "--set-attribute",
    "attrs",
    type=(str, str),
    multiple=True,
)
def set_attrs(
    *,
    dataset,
    attrs,
):
    with nc.Dataset(dataset, "a") as dset:
        for k, v in attrs:
            try:
                setattr(dset, k, v)
            except AttributeError:
                dset.setncattr(k, v)


@gtensor_cli.command("info", short_help="Get information about a G-Tensor")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
def info(dataset):

    attrs = disk.read_attrs(dataset)
    try:
        n_features = len(disk.list_features(dataset))
    except disk.NoFeaturesError:
        n_features = 0

    try:
        n_samples = len(disk.list_samples(dataset))
    except disk.NoSamplesError:
        n_samples = 0

    click.echo(f"Num features: {n_features}")
    click.echo(f"Num samples: {n_samples}")
    click.echo(f'Epigenome name: {attrs["name"]}')

    click.echo("Dataset dims:")
    for k, v in disk.read_dims(dataset).items():
        if not k == "sample":
            click.echo(f"\t{k}: {v}")

    click.echo("Dataset attributes:")
    for k, v in attrs.items():
        if not k == "name":
            click.echo(f"\t{k}: {v}")


@gtensor_cli.group("offsets", short_help="Manage locus offsets")
def offsets():
    pass


@offsets.command("add")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "offsets_file",
    type=click.Path(exists=True),
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column in the bedfile to use for the class",
)
def add_locus_offsets(
    dataset,
    offsets_file,
    column=4,
):
    exp_offsets = read_continuous_file(
        dataset,
        offsets_file,
        normalization="log1p_cpm",
        column=column,
    ).astype(np.float32)

    exp_offsets /= exp_offsets.mean()

    disk.write_locus_offsets(
        dataset,
        exp_offsets,
    )


@offsets.command("rm")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
def rm_locus_offsets(
    dataset,
):
    exp_exposures = np.ones(disk.read_dims(dataset)["locus"], dtype=np.float32)

    disk.write_locus_offsets(
        dataset,
        exp_exposures,
    )


@gtensor_cli.group("feature", short_help="Manage features in a G-Tensor")
def featurecmds():
    pass


@featurecmds.group("add")
def add_feature():
    pass


@add_feature.command("continuous")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "ingest_file",
    type=click.Path(exists=True),
)
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature",
)
@click.option(
    "-norm",
    "--normalization",
    type=click.Choice([x.value for x in FeatureType.continuous_types()]),
    default="log1p_cpm",
    help="Normalization to apply to the feature",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Group to assign the feature to",
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column in the bedfile to use for the class",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source to assign the feature to",
)
def _add_continuous_feature(*args, **kwargs):
    add_continuous_feature(*args, **kwargs)



@add_feature.command("discrete")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "ingest_file",
    type=click.Path(exists=True),
)
@click.argument(
    "classes",
    type=str,
    nargs=-1,
)
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature",
)
@click.option(
    "--mesoscale/--no-mesoscale",
    default=True,
    help="Whether to treat the feature as mesoscale",
    type=bool,
    is_flag=True,
)
@click.option(
    "-null",
    "--null",
    type=str,
    default="none",
    help="Value to assign to null regions",
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column in the bedfile to use for the class",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Group to assign the feature to",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source to assign the feature to",
)
def _add_discrete_feature(*args, **kwargs):
    add_discrete_feature(*args, **kwargs)



@add_feature.command("strand")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "ingest_file",
    type=click.Path(exists=True),
)
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Group to assign the feature to",
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column in the bedfile to use for the class",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source to assign the feature to",
)
def _add_strand_feature(*args, **kwargs):
    add_strand_feature(*args, **kwargs)


@add_feature.command("vector")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "ingest_file",
    type=click.Path(exists=True),
)
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Group to assign the feature to",
)
@click.option(
    "-norm",
    "--normalization",
    type=click.Choice([x.value for x in FeatureType]),
    default="log1p_cpm",
    help="Normalization to apply to the feature",
)
def _add_vector_feature(*args, **kwargs):
    add_vector_feature(*args, **kwargs)



@featurecmds.command("ls")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
def list_features(
    dataset: str,
):
    feature_info = disk.list_features(dataset)

    if "sample" in feature_info:
        del feature_info["sample"]

    if len(feature_info) == 0:
        click.echo("No features found in the dataset.")
        return

    headers = ["Feature name", "normalization", "group"]
    rows = [
        [feature_name] + [str(attrs.get(header, "")) for header in headers[1:]]
        for feature_name, attrs in feature_info.items()
    ]

    col_widths = [
        max(len(str(item)) for item in col) for col in zip(*([headers] + rows))
    ]

    def format_row(row):
        return " | ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row))

    click.echo(format_row(headers))
    click.echo("-+-".join("-" * width for width in col_widths))
    for row in rows:
        click.echo(format_row(row))


@featurecmds.command("rm")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "feature_names",
    type=str,
    nargs=-1,
)
def rm_features(
    dataset: str,
    feature_names: List[str],
):
    for feature_name in feature_names:
        disk.rm_feature(
            dataset,
            feature_name,
        )
        click.echo(f"Removed feature: {feature_name}")


@featurecmds.command("edit")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "feature_name",
    type=str,
)
@click.option(
    "-g",
    "--group",
    type=str,
    default=None,
    help="Group to assign the feature to",
)
@click.option(
    "-norm",
    "--normalization",
    type=click.Choice([x.value for x in FeatureType]),
    help="Normalization to apply to the feature",
    default=None,
)
def edit_feature(
    *,
    dataset: str,
    feature_name: str,
    group: Union[None, str] = None,
    normalization: Union[None, str] = None,
):

    disk.edit_feature_attrs(
        dataset,
        feature_name,
        group=group,
        normalization=FeatureType(normalization),
    )


@gtensor_cli.group("sample", short_help="Manage samples in a G-Tensor")
def samplecmds():
    pass


@samplecmds.command("add")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "sample_file",
    type=click.Path(exists=True),
)
@click.option(
    "-id",
    "--sample-id",
    type=str,
    required=True,
    help="ID to assign the sample",
)
@click.option(
    "-sw",
    "--sample-weight",
    type=click.FloatRange(0.0, 1000000, min_open=False),
    default=1.0,
    help="Weight to assign to the sample",
)
@click.option(
    "-m",
    "--mutation-rate-file",
    type=str,
    help="VCFS ONLY: File containing mutation rates",
)
@click.option(
    "-chr",
    "--chr-prefix",
    type=str,
    default="",
    help="VCFS ONLY: Prefix to add to chromosome names",
)
@click.option(
    "--pass-only/--no-pass-only",
    type=bool,
    default=True,
    help="VCFS ONLY: Whether to only ingest passing mutations",
)
@click.option(
    "-w",
    "--weight-col",
    type=str,
    help="VCFS ONLY: Column to use for mutation weights",
)
@click.option(
    "-name",
    "--sample-name",
    type=str,
    default=None,
    help="VCFS ONLY: Name of sample in multi-sample VCF.",
)
@click.option(
    "--cluster/--no-cluster",
    type=bool,
    default=True,
    help="VCFS ONLY: Whether to cluster the mutations in the sample, requires a mutation rate file.",
)
@click.option(
    "--skip-sort/--no-skip-sort",
    type=bool,
    default=False,
    help="Whether to skip sorting the VCF file. This will fail if the VCF is not in lexigraphical sorted order (chr1, chr10, ...).",
)
@click.option(
    "-fa",
    "--fasta",
    type=click.Path(exists=True),
    default=None,
    help="Fasta file to use for calculating context frequencies",
)
@click.option(
    "-cn",
    "--copy-number",
    type=click.Path(exists=True),
    default=None,
    help="Bed file containing copy number information for the sample.",
)
def _add_sample(*args, **kwargs):
    add_sample(*args, **kwargs)



@samplecmds.command("rm")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
@click.argument(
    "sample_names",
    type=str,
    nargs=-1,
)
def rm_samples(
    dataset,
    sample_names: List[str],
):

    for sample_name in sample_names:
        disk.rm_sample(
            dataset,
            sample_name,
        )
        click.echo(f"Removed sample: {sample_name}")


@samplecmds.command("ls")
@click.argument(
    "dataset",
    type=click.Path(exists=True),
)
def list_samples(dataset: str):

    samples = disk.list_samples(dataset)

    if len(samples) == 0:
        click.echo("No samples found in the dataset.")
        return

    print(*samples, sep="\n")


@gtensor_cli.group("utils", short_help="Utility functions for G-Tensors")
def utils():
    pass


@utils.command("linearize-beds")
@click.argument(
    "bed_files",
    type=click.Path(exists=True),
    nargs=-1,
)
def linearize_beds(
    bed_files: List[str],
    max_region_size=25000,
):
    ingest.linearize_beds(
        *bed_files,
        max_region_size=max_region_size,
    )


@utils.command("query-gtf")
@click.option(
    "--input", "-i", type=click.File("r"), default="-", help="Input GTF or GFF file"
)
@click.option("--output", "-o", type=click.File("w"), default="-", help="Output file")
@click.option("--type-filter", "-type", help="Only include records of this type")
@click.option(
    "--attribute-key", "-attr", help="Only include records with this attribute key"
)
@click.option(
    "--attribute-values",
    "-vals",
    multiple=True,
    default=None,
    help="Only include records with these attribute values under the specified key.",
)
@click.option(
    "--is-gff/--is-gtf",
    default=False,
    help="Input file is in GFF format (default: GTF)",
)
@click.option(
    "--header/--no-header",
    default=False,
    help="Print a header line with the column names.",
)
@click.option(
    "--format-str",
    "-f",
    default=None,
    help="Format string for output. Use {column_name} to insert values from the GFF record. Use {attributes[key]} to insert values from the attributes dictionary.",
)
@click.option(
    "--as-regions",
    is_flag=True,
    help='Output in regions format. Default format is "{chrom}:{start}-{end}\n"',
)
@click.option(
    "--as-gtf",
    is_flag=True,
)
@click.option(
    "--zero-based/--one-based",
    default=False,
    is_flag=True,
    help="Use 0-based coordinates instead of 1-based, default is 1-based.",
)
def _query_gtf(*args, **kwargs):
    query_gtf(*args, **kwargs)


@utils.command("make-expression-bedfile")
@click.argument(
    "quantitation_files",
    type=click.Path(exists=True),
    nargs=-1,
)
@click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
)
@click.option(
    "--join-on",
    type=click.Choice(["gene_id", "gene_name"]),
    default="gene_id",
    help="Column to join on. Default is 'gene_id'.",
)
@click.option(
    "--gtf-file",
    type=click.Path(exists=True),
    default=None,
    help="GTF file to use for annotation. If not provided, will download the latest GTF file.",
)
def make_quant_file(
    output,
    quantitation_files: List[str],
    gtf_file: str = None,
    join_on="gene_id",
):

    gtf_file = gtf_file or "MANE.GRCh38.v1.3.ensembl_genomic.gtf"

    if not os.path.exists(gtf_file):
        logger.info(f"Downloading GTF file: {gtf_file} ...")
        gene_features.download_gtf(gtf_file)

    annotation_file = "MANE.GRCh38.annotation.bed"
    if not os.path.exists(annotation_file):
        logger.info(f"Creating annotation file: {annotation_file} ...")
        gene_features.make_annotation(gtf_file, annotation_file)

    quant = gene_features.join_quantitation(
        annotation_file,
        *quantitation_files,
        join_on=join_on,
    )

    logger.info(f"Writing quantitation file: {output} ...")
    quant.to_csv(
        output,
        sep="\t",
        header=None,
        index=False,
    )
