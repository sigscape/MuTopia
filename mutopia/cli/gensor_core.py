import click
from typing import Optional, Union, List
from functools import partial
import os
import numpy as np
import xarray as xr
from sparse import COO
import netCDF4 as nc
from shutil import copyfile

import mutopia.ingestion as ingest
import mutopia.gtensor.disk_interface as disk
from mutopia.gtensor import *
from mutopia.genome_utils.bed12_utils import stream_bed12
from mutopia.utils import FeatureType, logger
from mutopia.gtensor.dtypes import get_mode_config
from mutopia.modalities import ModeConfig


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
        chrom=list(chrom),
        start=list(start),
        end=list(end),
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

    featuretype = FeatureType(normalization)
    if not (featuretype in file_type.allowed_normalizations):
        raise ValueError(
            f"Normalization {normalization} not supported for file type {file_type}"
        )

    corpus_attrs = disk.read_attrs(dataset)

    feature_vals = file_type.get_ingestion_fn(featuretype)(
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
        source_file=ingest_file,
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
        source_file=ingest_file,
    )

    logger.info("Found classes: " + ", ".join(['"{}"'.format(str(c)) for c in classes]))

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
        source_file=ingest_file,
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
        source_file=ingest_file,
    )


def train_test_split(
    filename: str,
    test_contigs: list[str],
    min_region_size=5,
    output_prefix: Optional[str] = None,
):

    if not len(test_contigs) > 0:
        raise click.BadParameter("Must provide at least one contig to use for testing.")
    
    if output_prefix is None:
        outprefix = ".".join(filename.split(".")[:-1]) + "."
    else:
        outprefix = output_prefix

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

    train = LocusSlice(LazySampleLoader(dataset), locus=train_mask)
    test = LocusSlice(LazySampleLoader(dataset), locus=test_mask)

    disk.write_dataset(train, outprefix + "train.nc", bar=True)
    disk.write_dataset(test, outprefix + "test.nc", bar=True)


def add_sample(
    dataset: str,
    sample_file: str,
    sample_name: Optional[str] = None,
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

    modality: ModeConfig = get_mode_config(attrs["dtype"])

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
        ploidy = ingest.make_continuous_features_bed(
            copy_number,
            regions_file,
            null="2",
        )

        ploidy = ploidy / 2 - 1  # normalize for diploid and center around 0

        ploidy = COO(ploidy)

        sample = xr.Dataset(
            {
                "X": X,
                "ploidy": ploidy,
            }
        )
    else:
        sample = xr.Dataset(
            {
                "X": X,
            }
        )

    disk.write_sample(
        dataset,
        sample,
        sample_name=sample_id,
    )


def slice_gtensor(
    dataset: str,
    output: str,
    query_regions: list[str],
):
    """Slice a G-Tensor by genomic regions."""
    dataset = lazy_load(dataset)
    dataset = slice_regions(dataset, *query_regions, lazy=True)
    disk.write_dataset(dataset, output, bar=True)


def slice_samples(
    dataset: str,
    output: str,
    sample_id_file: str,
):
    """Slice a G-Tensor by sample names."""
    with open(sample_id_file) as f:
        sample_names = [line.strip() for line in f if len(line.strip()) > 0]

    if not len(sample_names) > 0:
        raise click.BadParameter("Must provide at least one sample name.")

    d = lazy_load(dataset)
    d = interfaces.SampleSlice(d, sample_names)
    disk.write_dataset(d, output, bar=True)


def convert_gtensor(
    input: str,
    output: str,
    dtype: str,
    fasta_file: str,
):
    """Convert a G-Tensor to a different modality."""
    old_dataset = disk.load_dataset(input, with_samples=False)
    attrs = old_dataset.attrs

    input_regions_file = attrs["regions_file"]
    output_regions_file = output + ".regions.bed"
    copyfile(input_regions_file, output_regions_file)

    modality: ModeConfig = get_mode_config(dtype)

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


def set_gtensor_attrs(
    dataset: str,
    attrs: List[tuple[str, str]],
):
    """Set attributes on a G-Tensor."""
    with nc.Dataset(dataset, "a") as dset:
        for k, v in attrs:
            try:
                setattr(dset, k, v)
            except AttributeError:
                dset.setncattr(k, v)


def get_gtensor_info(dataset: str) -> dict:
    """Get information about a G-Tensor."""
    attrs = disk.read_attrs(dataset)
    try:
        n_features = len(disk.list_features(dataset))
    except disk.NoFeaturesError:
        n_features = 0

    try:
        n_samples = len(disk.list_samples(dataset))
    except disk.NoSamplesError:
        n_samples = 0

    dims = disk.read_dims(dataset)

    return {
        "n_features": n_features,
        "n_samples": n_samples,
        "name": attrs["name"],
        "dims": {k: v for k, v in dims.items() if k != "sample"},
        "attrs": {k: v for k, v in attrs.items() if k != "name"},
    }


def add_locus_offsets_to_gtensor(
    dataset: str,
    offsets_file: str,
    column: int = 4,
):
    """Add locus offsets to a G-Tensor from a file."""
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


def remove_locus_offsets_from_gtensor(dataset: str):
    """Remove locus offsets from a G-Tensor."""
    exp_exposures = np.ones(disk.read_dims(dataset)["locus"], dtype=np.float32)

    disk.write_locus_offsets(
        dataset,
        exp_exposures,
    )


def list_gtensor_features(dataset: str) -> dict:
    """List features in a G-Tensor with formatted output."""
    feature_info = disk.list_features(dataset)

    if "sample" in feature_info:
        del feature_info["sample"]

    return feature_info


def remove_gtensor_features(
    dataset: str,
    feature_names: List[str],
):
    """Remove features from a G-Tensor."""
    for feature_name in feature_names:
        disk.rm_feature(
            dataset,
            feature_name,
        )


def edit_gtensor_feature(
    dataset: str,
    feature_name: str,
    group: Union[None, str] = None,
    normalization: Union[None, str] = None,
):
    """Edit feature attributes in a G-Tensor."""
    disk.edit_feature_attrs(
        dataset,
        feature_name,
        group=group,
        normalization=FeatureType(normalization) if normalization else None,
    )


def remove_gtensor_samples(
    dataset: str,
    sample_names: List[str],
):
    """Remove samples from a G-Tensor."""
    for sample_name in sample_names:
        disk.rm_sample(
            dataset,
            sample_name,
        )


def list_gtensor_samples(dataset: str) -> List[str]:
    """List samples in a G-Tensor."""
    return disk.list_samples(dataset)


def make_annotation_bedfile(gtf_file: Optional[str] = None, output: Optional[str] = None):
    """Create an expression bedfile from quantitation files."""

    from mutopia.ingestion import gene_features

    gtf_file = gtf_file or "MANE.GRCh38.v1.3.ensembl_genomic.gtf"
    if not os.path.exists(gtf_file):
        logger.info(f"Downloading GTF file: {gtf_file} ...")
        gene_features.download_gtf(gtf_file)

    annotation_file = output or "MANE.GRCh38.annotation.bed"
    if not os.path.exists(annotation_file):
        logger.info(f"Creating annotation file: {annotation_file} ...")
        gene_features.make_annotation(gtf_file, annotation_file)

    return annotation_file


def make_expression_bedfile(
    quantitation_file: str,
    output_file,
    gtf_file: Optional[str] = None,
    join_on: str = "gene_id",
):

    from mutopia.ingestion import gene_features
    
    annotation_file = make_annotation_bedfile(gtf_file)

    quant = gene_features.join_quantitation(
        annotation_file,
        quantitation_file,
        join_on=join_on,
    )

    logger.info(f"Writing quantitation file: {output_file} ...")
    quant.to_csv(
        output_file,
        sep="\t",
        header=None,
        index=False,
    )
