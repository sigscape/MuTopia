import click
from typing import *
from functools import partial
import os
import numpy as np
import mutopia.ingestion as ingest
from mutopia.gtensor import *
import mutopia.gtensor.disk_interface as disk
from ..modalities import ModeConfig
from ..gtensor import *
from ..genome_utils.bed12_utils import stream_bed12
from ..utils import FeatureType, logger
from ..dtypes import get_mode_config
from ..ingestion import make_continuous_features_bed
from sparse import COO

def create_gtensor(
    *,
    name: str,
    dtype: str,
    output: str,
    genome_file,
    fasta_file,
    blacklist_file,
    cutout_regions=[],
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

    modality = get_mode_config(dtype)

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


def read_continuous_file(
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


def add_continuous_feature(
    dataset: str,
    ingest_file: str,
    normalization: str = "log1p_cpm",
    group="all",
    column: int = 4,
    source=None,
    *,
    feature_name: str,
    **kw,
):

    logger.info(f"Ingesting data from {ingest_file} ...")

    read_file = partial(
        read_continuous_file,
        dataset,
        normalization=normalization,
        column=column,
    )

    vals = read_file(ingest_file)

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
        name=feature_name if source is None else os.path.join(source, feature_name),
        normalization=FeatureType(normalization),
    )

    logger.info(f"Added feature: {feature_name}")



def add_discrete_feature(
    ingest_file: str,
    group: str = "all",
    mesoscale: bool = True,
    null: str = "None",
    column: int = 4,
    classes: List[str] = [],
    source: Union[None, str] = None,
    *,
    dataset: str,
    feature_name: str,
    **kw,
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
        name=feature_name if source is None else os.path.join(source, feature_name),
        normalization=FeatureType.MESOSCALE if mesoscale else FeatureType.CATEGORICAL,
        classes=classes,
    )

    logger.info(f"Added feature: {feature_name}")


def add_vector_feature(
    ingest_file: str,
    group: str = "all",
    normalization: str = "log1p_cpm",
    source: Union[None, str] = None,
    *,
    dataset: str,
    feature_name: str,
    **kw,
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
        name=feature_name if source is None else os.path.join(source, feature_name),
        normalization=FeatureType(normalization),
    )


def add_strand_feature(
    ingest_file: str,
    group: str = "all",
    column: int = 4,
    source: Union[None, str] = None,
    *,
    dataset: str,
    feature_name: str,
    **kw,
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
        name=feature_name if source is None else os.path.join(source, feature_name),
        normalization=FeatureType.STRAND,
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

    test_mask = np.isin(dataset["Regions/chrom"].values, test_contigs)
    include_region = dataset["Regions/length"].values >= min_region_size

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



def add_sample(
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

    modality : ModeConfig = get_mode_config(attrs["dtype"])

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
            null=2.0,
        )
        ploidy = ploidy / 2 - 1  # normalize for diploid and center around 0
        ploidy = COO(ploidy)

        sample = xr.Dataset(
            {
                "X": X,
                "ploidy": ploidy,
            }
        )

    disk.write_sample(
        dataset,
        sample,
        sample_name=sample_id,
    )