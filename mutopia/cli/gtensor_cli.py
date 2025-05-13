import click
from typing import *
from functools import partial
import os
import numpy as np
from sparse import COO
import mutopia.ingestion as ingest
from mutopia.gtensor import *
import netCDF4 as nc
from shutil import copyfile
from ..gtensor.interfaces import *
import mutopia.gtensor.disk_interface as disk
from ..ingestion import make_continuous_features_bed
from ..modalities import *
from ..genome_utils.bed12_utils import stream_bed12
from ..utils import FeatureType, logger


def _read_continuous_file(
    dataset,
    ingest_file,
    normalization: FeatureType,
    column=4,
):

    file_type = ingest.FileType.from_extension(ingest_file)

    if not file_type in (
        ingest.FileType.BEDGRAPH,
        ingest.FileType.BIGWIG,
        ingest.FileType.BED,
    ):
        raise ValueError(
            f"File type {file_type} not supported for continuous feature ingestion."
        )

    if not FeatureType(normalization) in file_type.allowed_normalizations:
        raise ValueError(
            f"Normalization {normalization} not supported for file type {file_type}"
        )

    corpus_attrs = disk.read_attrs(dataset)

    feature_vals = file_type.get_ingestion_fn(
        is_distance_feature=False,
        is_discrete_feature=False,
    )(
        ingest_file,
        disk.fetch_regions_path(dataset),
        genome_file=corpus_attrs["genome_file"],
        column=column,
    )

    return feature_vals


@click.group("G-tensor commands")
def gtensor_cli():
    pass


@gtensor_cli.group("utils")
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


@gtensor_cli.command("split")
@click.argument("filename", type=click.Path(exists=True))
@click.argument("test_contigs", nargs=-1, type=str, required=True)
@click.option(
    "-min",
    "--min-region-size",
    type=click.IntRange(1, 100000000),
    default=5,
)
def train_test_split(
    filename: str,
    test_contigs: list[str],
    min_region_size=5,
):

    if not len(test_contigs) > 0:
        raise click.BadParameter("Must provide at least one contig to use for testing.")

    outprefix = ".".join(filename.split(".")[:-1])

    dataset = disk.load_dataset(filename, with_samples=False)
    dataset.attrs["regions_file"] = "none"

    test_mask = np.isin(dataset.regions.chrom.values, test_contigs)
    include_region = dataset.regions.length.values >= min_region_size

    train_mask = ~test_mask & include_region
    test_mask = test_mask & include_region

    if not np.any(test_mask) or not np.any(train_mask):
        raise ValueError(
            "Was not able to produce a valid test or training partition given these parameters.\n"
            "Either check to make sure the corpus contains the contigs you provided, or try using a smaller min-region-size."
        )

    train = LazySlicer(LazySampleLoader(dataset), locus=train_mask)
    test = LazySlicer(LazySampleLoader(dataset), locus=test_mask)

    disk.write_dataset(train, outprefix + ".train.nc", bar=True)
    disk.write_dataset(test, outprefix + ".test.nc", bar=True)


@gtensor_cli.command("slice")
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


@gtensor_cli.command("create")
@click.argument(
    "cutout_regions",
    type=click.Path(exists=True),
    nargs=-1,
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
def create(
    *,
    name: str,
    dtype: str,
    output: str,
    genome_file,
    fasta_file,
    blacklist_file,
    cutout_regions: List[str] = [],
    min_region_size=25,
    region_size: int = 10000,
    base_regions: Union[None, str] = None,
):
    logger.info("Creating genomic regions ...")
    regions_file = output + ".regions.bed"
    with open(regions_file, "w") as r:
        ingest.make_regions(
            *cutout_regions,
            genome_file=genome_file,
            window_size=region_size,
            blacklist_file=blacklist_file,
            min_windowsize=min_region_size,
            output=r,
            base_regions=base_regions,
        )

    logger.info("Calculating context frequencies ...")

    modality = get_config(dtype)

    context_freqs = modality.get_context_frequencies(
        regions_file=regions_file,
        fasta_file=fasta_file,
    )

    logger.info("Formatting G-Tensor ...")

    chrom, start, end = list(
        zip(*((s.chromosome, s.start, s.end) for s in stream_bed12(regions_file)))
    )

    gtensor = GTensor(
        modality,
        name=name,
        chrom=chrom,
        start=start,
        end=end,
        context_frequencies=context_freqs,
        dtype=dtype,
    )

    gtensor.attrs["regions_file"] = os.path.basename(regions_file)
    gtensor.attrs["genome_file"] = genome_file
    gtensor.attrs["fasta_file"] = fasta_file
    gtensor.attrs["blacklist_file"] = blacklist_file
    gtensor.attrs["region_size"] = region_size

    logger.info(
        f"Writing G-Tensor to {output}, and accompanying regions file to {regions_file} ...\n"
        "Make sure not to leave the regions file behind if you move the G-Tensor!."
    )

    write_dataset(gtensor, output)


@gtensor_cli.command("convert")
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

    modality = get_config(dtype)

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


@gtensor_cli.command("set-attr")
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


@gtensor_cli.command("info")
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


@gtensor_cli.group("offsets")
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
    exp_offsets = _read_continuous_file(
        dataset,
        offsets_file,
        normalization="log1p_cpm",
        column=column,
    ).astype(np.float32)

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


@gtensor_cli.group("feature")
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
def continuous_feature(
    dataset: str,
    ingest_file: List[str],
    normalization: str = "log1p_cpm",
    group="all",
    column: int = 4,
    *,
    feature_name: str,
):

    logger.info(f"Ingesting data from {ingest_file} ...")
    
    read_file = partial(
        _read_continuous_file,
        dataset,
        normalization=normalization,
        column=column,
    )

    vals = np.mean([read_file(f) for f in ingest_file], axis=0)

    def arr_render(arr):
        return ", ".join("{:.2f}".format(x) for x in arr)

    logger.info(
        f"First values of feature {feature_name}: {arr_render(vals[:5])}\n"
        f"First non-null values: {arr_render(vals[~np.isnan(vals)][:5])}"
    )

    disk.write_feature(
        dataset,
        vals,
        group=group,
        name=feature_name,
        normalization=FeatureType(normalization),
    )

    logger.info(f"Added feature: {feature_name}")


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
def add_discrete_feature(
    ingest_file: str,
    group: str = "all",
    mesoscale: bool = True,
    null: str = "None",
    column: int = 4,
    classes: List[str] = [],
    *,
    dataset: str,
    feature_name: str,
):

    if not ingest.FileType.from_extension(ingest_file) == ingest.FileType.BED:
        raise ValueError("Discrete features must be ingested from Bed files.")
    
    logger.info(f"Ingesting data from {ingest_file} ...")

    feature_vals, classes = ingest.make_discrete_features(
        ingest_file,
        disk.fetch_regions_path(dataset),
        column=column,
        null=null,
        class_priority=classes if len(classes) > 0 else None,
    )

    disk.write_feature(
        dataset,
        feature_vals,
        group=group,
        name=feature_name,
        normalization=FeatureType.MESOSCALE if mesoscale else FeatureType.CATEGORICAL,
        classes=classes,
    )

    logger.info(f"Added feature: {feature_name}")


@add_feature.command("distance")
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
def add_distance_feature(
    ingest_file: str,
    group: str = "all",
    *,
    dataset: str,
    feature_name: str,
):
    if not ingest.FileType.from_extension(ingest_file) == ingest.FileType.BED:
        raise ValueError("Discrete features must be ingested from Bed files.")

    (progress_between, distance_between) = ingest.make_distance_features(
        ingest_file,
        disk.fetch_regions_path(dataset),
    )

    write_fn = partial(
        disk.write_feature,
        dataset,
        group=group,
        normalization=FeatureType.QUANTILE,
    )

    write_fn(
        progress_between,
        name=f"{feature_name}_positionBetween",
    )

    write_fn(
        distance_between,
        name=f"{feature_name}_distanceBetween",
    )


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
def add_strand_feature(
    ingest_file: str,
    group: str = "all",
    column: int = 4,
    *,
    dataset: str,
    feature_name: str,
):
    if not ingest.FileType.from_extension(ingest_file) == ingest.FileType.BED:
        raise ValueError("Discrete features must be ingested from Bed files.")

    feature_vals = ingest.make_strand_features(
        ingest_file,
        disk.fetch_regions_path(dataset),
        column=column,
    )

    disk.write_feature(
        dataset,
        feature_vals,
        group=group,
        name=feature_name,
        normalization=FeatureType.STRAND,
    )


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
def add_vector(
    ingest_file: str,
    group: str = "all",
    normalization: str = "log1p_cpm",
    *,
    dataset: str,
    feature_name: str,
):

    with open(ingest_file) as f:
        feature_vals = np.array([float(x.strip()) for x in f])

    dims = disk.read_dims(dataset)
    if not len(feature_vals) == dims["locus"]:
        raise ValueError(
            f'Feature vector length ({len(feature_vals)}) does not match the number of loci in the dataset ({dims["locus"]}).'
        )

    disk.write_feature(
        dataset,
        feature_vals,
        group=group,
        name=feature_name,
        normalization=FeatureType(normalization),
    )


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


@gtensor_cli.group("sample")
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
def ingest_sample(
    dataset: str,
    sample_file: str,
    sample_name: str,
    chr_prefix: str = "",
    pass_only: bool = True,
    weight_col: Union[None, str] = None,
    mutation_rate_file: Union[None, str] = None,
    sample_weight: Union[None, float] = 1.0,
    skip_sort: bool = False,
    cluster: bool = True,
    fasta: Union[None, str] = None,
    copy_number: Union[None, str] = None,
    *,
    sample_id: str,
):

    attrs = disk.read_attrs(dataset)
    regions_file = disk.fetch_regions_path(dataset)
    num_regions = sum(1 for _ in stream_bed12(regions_file))
    locus_coords = disk.read_coords(dataset)["locus"]

    modality = get_config(attrs["dtype"])

    fasta = fasta or attrs["fasta_file"]
    if not os.path.exists(fasta):
        raise click.FileError(
            f"No such file exists: {fasta}, provide a valid fasta file using the `--fasta/-fa` argument."
        )

    X = modality.ingest_observations(
        sample_file,
        #
        regions_file=regions_file,
        locus_dim=num_regions,
        locus_coords=locus_coords,
        fasta_file=fasta,
        #
        chr_prefix=chr_prefix,
        pass_only=pass_only,
        weight_col=weight_col,
        sample_weight=sample_weight,
        sample_name=sample_name,
        #
        mutation_rate_file=mutation_rate_file,
        skip_sort=skip_sort,
        cluster=cluster,
    )

    if not copy_number is None:
        ploidy = make_continuous_features_bed(
            copy_number,
            regions_file,
            null=2.,
        )
        ploidy = ploidy/2 - 1 # normalize for diploid and center around 0
        ploidy = COO(ploidy)

        sample = xr.Dataset({
            "X": X,
            "ploidy": ploidy,
        })

    disk.write_sample(
        dataset,
        sample,
        sample_name=sample_id,
    )


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
