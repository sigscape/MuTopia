import click
import sys
from typing import List, Union, Optional
from mutopia.utils import FeatureType


@click.group("G-tensor commands")
def gtensor_cli():
    """
    Manage G-Tensor datasets for genomic data analysis.

    G-Tensors are structured genomic datasets that organize samples, features,
    and observations across genomic loci for machine learning applications.
    """
    pass


@gtensor_cli.command("compose", short_help="Run the G-Tensor construction pipeline")
@click.argument("config_file", type=click.Path(exists=True), metavar="CONFIG_FILES...", nargs=-1)
@click.option(
    "-n", "--name",
    type=str,
    required=True,
    help="Name of the G-Tensor dataset."
)
@click.option(
    "-d", "--dtype",
    type=str,
    default="sbs",
    help="Data type of the G-Tensor dataset."
)
@click.option(
    "--region-size", 
    type=int,
    default=10000,
    help="Max size of the genomic regions to create."
)
@click.option(
    "--min-region-size",
    type=int,
    default=75,
    help="Min size of the genomic regions to create."
)
@click.option(
    "-w",
    "--workers",
    type=click.IntRange(1, 100),
    default=1,
    help="Number of parallel workers to use for the pipeline execution.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    is_flag=True,
    help="Run in dry-run mode to validate configuration without executing pipeline tasks.",
)
def _run_pipeline(
    config_file: List[str],
    **kwargs,
):
    """
    Execute the G-Tensor construction pipeline from a configuration file.

    CONFIG_FILE should be a YAML or JSON file containing pipeline configuration
    including data sources, processing steps, and output specifications.

    The pipeline can ingest various genomic data formats (BED, BigWig, etc.)
    and create structured G-Tensor datasets with features and samples.
    """
    from .pipeline_tasks import run_pipeline
    if len(config_file) == 0:
        click.echo("Error: At least one CONFIG_FILE must be provided.", err=True)
        sys.exit(1)
        
    run_pipeline(tuple(config_file), **kwargs)


@gtensor_cli.command("split", short_help="Split a G-Tensor into training and test sets")
@click.argument("filename", type=click.Path(exists=True), metavar="DATASET")
@click.argument("test_contigs", nargs=-1, type=str, required=True, metavar="CONTIG...")
@click.option("-o", "--output-prefix", type=str, default=None, help="Prefix for output files, defaults to input filename")
@click.option(
    "-min",
    "--min-region-size",
    type=click.IntRange(1, 100000000),
    default=5,
    help="Minimum region size in base pairs to include in splits.",
)
def _train_test_split(*args, **kwargs):
    """
    Split a G-Tensor dataset into training and test sets based on chromosomes/contigs.

    DATASET is the path to the G-Tensor file to split.
    CONTIG... are one or more chromosome/contig names to use for the test set.

    Creates two new files: DATASET.train.nc and DATASET.test.nc
    The training set contains all contigs except those specified for testing.
    Only regions meeting the minimum size requirement are included.

    Example:
        gtensor split my_dataset.nc chr21 chr22 --min-region-size 100
    """
    from .gensor_core import train_test_split
    train_test_split(*args, **kwargs)

@gtensor_cli.group("slice", short_help="Slice a G-Tensor by samples or regions")
def slice():
    pass

@slice.command("samples")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("output", type=click.Path(writable=True), metavar="OUTPUT")
@click.argument("sample_id_file", type=click.Path(exists=True, dir_okay=False), metavar="SAMPLE_ID_FILE")
def _slice_samples(*args, **kwargs):
    """
    Extract specific samples from a G-Tensor dataset into a new file.

    DATASET is the input G-Tensor file.
    OUTPUT is the output path for the sliced G-Tensor.
    SAMPLE_ID_FILE is a text file with one sample ID per line to extract.

    Example:
        gtensor slice-samples input.nc output.nc sample_ids.txt
    """
    from .gensor_core import slice_samples
    slice_samples(*args, **kwargs)


@slice.command("regions")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("output", type=click.Path(writable=True), metavar="OUTPUT")
@click.argument("query_regions", type=str, nargs=-1, metavar="REGIONS...")
def _slice_regions(*args, **kwargs):
    """
    Extract specific genomic regions from a G-Tensor dataset.

    DATASET is the input G-Tensor file.
    OUTPUT is the output path for the sliced G-Tensor.

    Each region is specified as three space-separated values:
    chromosome (string), start position (int), end position (int).

    Example:
        gtensor slice input.nc output.nc chr1:1000-20000 chr2
    """
    from .gensor_core import slice_gtensor
    slice_gtensor(*args, **kwargs)


@gtensor_cli.command("create", short_help="Create a new G-Tensor")
@click.option(
    "-cut",
    "--cutout-regions",
    type=(str, click.Path(exists=True)),
    multiple=True,
    help="Regions to cut out of the genome as (description, bed_file) pairs.",
)
@click.option(
    "-n",
    "--name",
    type=str,
    required=True,
    help="Name to assign the G-Tensor dataset",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True),
    required=True,
    help="Output file path to write the G-Tensor to",
)
@click.option(
    "-dtype",
    "--dtype",
    type=str,
    required=True,
    help="Modality type to use for the G-Tensor (e.g., 'sbs', 'indel')",
)
@click.option(
    "-g",
    "--genome-file",
    type=click.Path(exists=True),
    required=True,
    help="Genome sizes file (.genome format) defining chromosome lengths",
)
@click.option(
    "-v",
    "--blacklist-file",
    type=click.Path(exists=True),
    required=True,
    help="BED file containing genomic regions to exclude (e.g., repetitive elements)",
)
@click.option(
    "-fa",
    "--fasta-file",
    type=click.Path(exists=True),
    required=True,
    help="Reference genome FASTA file for calculating sequence context frequencies",
)
@click.option(
    "-s",
    "--region-size",
    type=click.IntRange(1, 1000000),
    default=10000,
    help="Default size in base pairs for genomic regions",
)
@click.option(
    "-min",
    "--min-region-size",
    type=click.IntRange(1, 1000000),
    default=25,
    help="Minimum size in base pairs for genomic regions to be included",
)
@click.option(
    "-b",
    "--base-regions",
    type=click.Path(exists=True),
    default=None,
    help="BED file containing custom regions to use as the G-Tensor base. If not provided, uniform regions of --region-size will be created.",
)
def _create_gtensor(*args, **kwargs):
    """
    Create a new G-Tensor dataset from genomic reference files.
    
    This command initializes a new G-Tensor with defined genomic regions and calculates
    sequence context frequencies based on the specified modality. The resulting dataset
    can then be populated with features and samples.
    
    The modality (--dtype) determines what type of genomic variations will be analyzed:
    - 'sbs': Single base substitutions 
    
    Example:
        gtensor create -n "MyDataset" -o dataset.nc -dtype sbs \\
                      -g genome.sizes -v blacklist.bed -fa reference.fa
    """
    from .gensor_core import create_gtensor

    create_gtensor(*args, **kwargs)


@gtensor_cli.command("convert", short_help="Convert a G-Tensor to a different modality")
@click.argument("input", type=click.Path(exists=True), metavar="INPUT")
@click.argument("output", type=click.Path(writable=True), metavar="OUTPUT")
@click.option(
    "-dtype",
    "--dtype",
    type=str,
    required=True,
    help="Target modality to convert the G-Tensor to (e.g., 'sbs', 'indel')",
)
@click.option(
    "-fa",
    "--fasta-file",
    type=click.Path(exists=True),
    required=True,
    help="Reference genome FASTA file for calculating context frequencies in the new modality",
)
def _convert_gtensor(
    *,
    input,
    dtype: str,
    output: str,
    fasta_file: str,
):
    """
    Convert an existing G-Tensor dataset to a different modality.

    INPUT is the path to the source G-Tensor file.
    OUTPUT is the path for the converted G-Tensor file.

    This command preserves all features and attributes from the original dataset
    while recalculating context frequencies for the new modality. The genomic
    regions remain unchanged, but the observation space is adapted to the new
    modality type.

    Example:
        gtensor convert input_sbs.nc output_indel.nc -dtype indel -fa reference.fa
    """
    from .gensor_core import convert_gtensor

    convert_gtensor(input, output, dtype, fasta_file)


@gtensor_cli.command("set-attr", short_help="Set attributes on a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.option(
    "-set",
    "--set-attribute",
    "attrs",
    type=(str, str),
    multiple=True,
    help="Set attribute as key-value pair. Can be used multiple times.",
)
def _set_attrs(
    *,
    dataset,
    attrs,
):
    """
    Set or modify attributes on a G-Tensor dataset.

    DATASET is the path to the G-Tensor file to modify.

    Attributes are metadata key-value pairs stored with the dataset.
    This command allows you to add new attributes or modify existing ones.

    Example:
        gtensor set-attr dataset.nc -set fasta_file "path/to/fasta.fa"
    """
    from .gensor_core import set_gtensor_attrs

    set_gtensor_attrs(dataset, attrs)


@gtensor_cli.command("info", short_help="Display information about a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
def _info(dataset):
    """
    Display comprehensive information about a G-Tensor dataset.

    DATASET is the path to the G-Tensor file to examine.

    Shows dataset dimensions, number of features and samples, dataset name,
    and all stored attributes. This is useful for quickly understanding
    the structure and contents of a G-Tensor dataset.

    Example:
        gtensor info my_dataset.nc
    """
    from .gensor_core import get_gtensor_info

    info = get_gtensor_info(dataset)

    click.echo(f"Num features: {info['n_features']}")
    click.echo(f"Num samples: {info['n_samples']}")
    click.echo(f"Dataset name: {info['name']}")

    click.echo("Dataset dimensions:")
    for k, v in info["dims"].items():
        click.echo(f"\t{k}: {v}")

    click.echo("Dataset attributes:")
    for k, v in info["attrs"].items():
        click.echo(f"\t{k}: {v}")


@gtensor_cli.group("offsets", short_help="Manage genomic locus exposure offsets")
def offsets():
    """
    Manage exposure offsets for genomic loci in G-Tensor datasets.

    Locus offsets are used to account for varying sequencing depth, mappability,
    or other technical factors that affect the expected observation rate across
    different genomic regions.
    """
    pass


@offsets.command("add", short_help="Add locus offsets from a file")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("offsets_file", type=click.Path(exists=True), metavar="OFFSETS_FILE")
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column number in the BED file containing offset values (1-based indexing)",
)
def _add_locus_offsets(
    dataset,
    offsets_file,
    column=4,
):
    """
    Add locus-specific exposure offsets to a G-Tensor from a BED file.

    DATASET is the G-Tensor file to modify.
    OFFSETS_FILE is a BED file with offset values for each genomic region.

    Offset values are normalized to have a mean of 1.0 and are used to adjust
    the expected rate of observations in each genomic region. Higher offsets
    indicate regions with higher expected observation rates.

    Example:
        gtensor offsets add dataset.nc mappability.bed --column 5
    """
    from .gensor_core import add_locus_offsets_to_gtensor

    add_locus_offsets_to_gtensor(dataset, offsets_file, column)


@offsets.command("rm", short_help="Remove locus offsets")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
def _rm_locus_offsets(
    dataset,
):
    """
    Remove locus offsets from a G-Tensor dataset.

    DATASET is the G-Tensor file to modify.

    This resets all locus offsets to 1.0, effectively removing any
    region-specific exposure adjustments.

    Example:
        gtensor offsets rm dataset.nc
    """
    from .gensor_core import remove_locus_offsets_from_gtensor

    remove_locus_offsets_from_gtensor(dataset)


@gtensor_cli.group("feature", short_help="Manage features in a G-Tensor")
def featurecmds():
    """
    Manage features in G-Tensor datasets.

    Features represent genomic annotations or experimental data values
    associated with each genomic region in the G-Tensor.
    """
    pass


@featurecmds.group("add", short_help="Add features to a G-Tensor")
def add_feature():
    """
    Add various types of features to a G-Tensor dataset.

    Features can be continuous values (e.g., ChIP-seq signal), discrete
    categories (e.g., chromatin states), or specialized types like strand
    information or pre-computed vectors.
    """
    pass


@add_feature.command("continuous", short_help="Add a continuous-valued feature")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("ingest_file", type=click.Path(exists=True), metavar="INGEST_FILE")
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature in the dataset",
)
@click.option(
    "-norm",
    "--normalization",
    type=click.Choice([x.value for x in FeatureType.continuous_types()]),
    default="log1p_cpm",
    help="Normalization method to apply to feature values",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Feature group for organization (default: 'all')",
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column number in BED file containing values (1-based indexing)",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source identifier to prepend to feature name",
)
def _add_continuous_feature(*args, **kwargs):
    """
    Add a continuous-valued feature from BED, bedGraph, or BigWig files.

    DATASET is the G-Tensor file to modify.
    INGEST_FILE is the input file containing feature values.

    Continuous features represent numeric values like ChIP-seq signal intensity,
    conservation scores, or other quantitative genomic annotations.

    Supported file formats:
    - BED files with numeric values in specified column
    - bedGraph files with 4-column format
    - BigWig files with continuous signal data

    Example:
        gtensor feature add continuous dataset.nc chipseq.bw -name H3K4me3 -norm log1p_cpm
    """
    from .gensor_core import add_continuous_feature

    add_continuous_feature(*args, **kwargs)


@add_feature.command("discrete", short_help="Add a discrete/categorical feature")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("ingest_file", type=click.Path(exists=True), metavar="INGEST_FILE")
@click.argument("classes", type=str, nargs=-1, metavar="CLASS...")
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature in the dataset",
)
@click.option(
    "--mesoscale/--no-mesoscale",
    default=True,
    help="Treat feature as mesoscale (continuous encoding) vs categorical",
    type=bool,
    is_flag=True,
)
@click.option(
    "-null",
    "--null",
    type=str,
    default="none",
    help="Value to assign to regions with no annotation",
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column number in BED file containing class labels",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Feature group for organization (default: 'all')",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source identifier to prepend to feature name",
)
def _add_discrete_feature(*args, **kwargs):
    """
    Add a discrete/categorical feature from a BED file.
    
    DATASET is the G-Tensor file to modify.
    INGEST_FILE is a BED file with categorical annotations.
    CLASS... are optional class names to define priority order.
    
    Discrete features represent categorical genomic annotations like
    chromatin states, gene types, or other discrete classifications.
    
    When multiple overlapping annotations exist for a region, classes
    listed earlier in CLASS... take priority. If no classes are specified,
    priority is determined automatically.
    
    Example:
        gtensor feature add discrete dataset.nc chromhmm.bed -name ChromState \\
                          Promoter Enhancer Quiescent --mesoscale
    """
    from .gensor_core import add_discrete_feature

    add_discrete_feature(*args, **kwargs)


@add_feature.command("strand", short_help="Add strand orientation feature")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("ingest_file", type=click.Path(exists=True), metavar="INGEST_FILE")
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature in the dataset",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Feature group for organization (default: 'all')",
)
@click.option(
    "-col",
    "--column",
    type=click.IntRange(4, 1000),
    default=4,
    help="Column number in BED file containing strand information",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source identifier to prepend to feature name",
)
def _add_strand_feature(*args, **kwargs):
    """
    Add a strand orientation feature from a BED file.

    DATASET is the G-Tensor file to modify.
    INGEST_FILE is a BED file with strand information ('+', '-', or '.').

    Strand features encode the directionality of genomic elements like
    genes, transcripts, or other oriented annotations. Values are encoded
    as: '+' = 1, '-' = -1, '.' or missing = 0.

    Example:
        gtensor feature add strand dataset.nc genes.bed -name GeneStrand -col 6
    """
    from .gensor_core import add_strand_feature

    add_strand_feature(*args, **kwargs)


@add_feature.command("vector", short_help="Add a pre-computed feature vector")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("ingest_file", type=click.Path(exists=True), metavar="INGEST_FILE")
@click.option(
    "-name",
    "--feature-name",
    type=str,
    required=True,
    help="Name to assign the feature in the dataset",
)
@click.option(
    "-g",
    "--group",
    default="all",
    type=str,
    help="Feature group for organization (default: 'all')",
)
@click.option(
    "-norm",
    "--normalization",
    type=click.Choice([x.value for x in FeatureType]),
    default="log1p_cpm",
    help="Normalization method to apply to feature values",
)
@click.option(
    "-s",
    "--source",
    type=str,
    default=None,
    help="Source identifier to prepend to feature name",
)
def _add_vector_feature(*args, **kwargs):
    """
    Add a pre-computed feature vector from a text file.

    DATASET is the G-Tensor file to modify.
    INGEST_FILE is a text file with one numeric value per line.

    Vector features are pre-computed numeric values for each genomic
    region in the dataset. The file must contain exactly as many values
    as there are regions in the G-Tensor, in the same order.

    This is useful for adding externally computed features like
    chromatin accessibility scores, conservation metrics, or
    machine learning-derived features.

    Example:
        gtensor feature add vector dataset.nc accessibility.txt -name ATAC_signal
    """
    from .gensor_core import add_vector_feature

    add_vector_feature(*args, **kwargs)


@featurecmds.command("ls", short_help="List features in a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
def _list_features(
    dataset: str,
):
    """
    List all features in a G-Tensor dataset with their attributes.

    DATASET is the G-Tensor file to examine.

    Displays a formatted table showing feature names, normalization methods,
    and group assignments for all features in the dataset.

    Example:
        gtensor feature ls dataset.nc
    """
    from .gensor_core import list_gtensor_features

    feature_info = list_gtensor_features(dataset)

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


@featurecmds.command("rm", short_help="Remove features from a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument(
    "feature_names", type=str, nargs=-1, required=True, metavar="FEATURE..."
)
def _rm_features(
    dataset: str,
    feature_names: List[str],
):
    """
    Remove one or more features from a G-Tensor dataset.

    DATASET is the G-Tensor file to modify.
    FEATURE... are the names of features to remove.

    This permanently deletes the specified features and their data
    from the dataset. Use with caution.

    Example:
        gtensor feature rm dataset.nc old_feature1 old_feature2
    """
    from .gensor_core import remove_gtensor_features

    remove_gtensor_features(dataset, feature_names)
    for feature_name in feature_names:
        click.echo(f"Removed feature: {feature_name}")


@featurecmds.command("edit", short_help="Edit feature attributes")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("feature_name", type=str, metavar="FEATURE_NAME")
@click.option(
    "-g",
    "--group",
    type=str,
    default=None,
    help="New group assignment for the feature",
)
@click.option(
    "-norm",
    "--normalization",
    type=click.Choice([x.value for x in FeatureType]),
    help="New normalization method for the feature",
    default=None,
)
def _edit_feature(
    *,
    dataset: str,
    feature_name: str,
    group: Union[None, str] = None,
    normalization: Union[None, str] = None,
):
    """
    Edit the attributes of an existing feature in a G-Tensor.

    DATASET is the G-Tensor file to modify.
    FEATURE_NAME is the name of the feature to edit.

    This command allows you to change the group assignment or
    normalization method of an existing feature without re-ingesting
    the data.

    Example:
        gtensor feature edit dataset.nc my_feature -g regulatory -norm zscore
    """
    from .gensor_core import edit_gtensor_feature

    edit_gtensor_feature(dataset, feature_name, group, normalization)


@gtensor_cli.group("sample", short_help="Manage samples in a G-Tensor")
def samplecmds():
    """
    Manage samples in G-Tensor datasets.

    Samples represent individual observations (e.g., tumor samples, cell lines)
    that contain genomic variations or other observations mapped to the G-Tensor's
    genomic regions.
    """
    pass


@samplecmds.command("add", short_help="Add a sample to a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("sample_file", type=click.Path(exists=True), metavar="SAMPLE_FILE")
@click.option(
    "-id",
    "--sample-id",
    type=str,
    required=True,
    help="Unique identifier to assign to the sample",
)
@click.option(
    "-sw",
    "--sample-weight",
    type=click.FloatRange(0.0, 1000000, min_open=False),
    default=1.0,
    help="Weight to assign to the sample for analysis (default: 1.0)",
)
@click.option(
    "-m",
    "--mutation-rate-file",
    type=str,
    help="VCF ONLY: File containing mutation rates for clustering analysis",
)
@click.option(
    "-chr",
    "--chr-prefix",
    type=str,
    default="",
    help="VCF ONLY: Prefix to add to chromosome names for coordinate matching",
)
@click.option(
    "--pass-only/--no-pass-only",
    type=bool,
    default=True,
    help="VCF ONLY: Whether to only ingest variants marked as PASS",
)
@click.option(
    "-w",
    "--weight-col",
    type=str,
    help="VCF ONLY: INFO field name to use for variant weights",
)
@click.option(
    "-name",
    "--sample-name",
    type=str,
    default=None,
    help="VCF ONLY: Name of sample in multi-sample VCF to extract",
)
@click.option(
    "--cluster/--no-cluster",
    type=bool,
    default=True,
    help="VCF ONLY: Whether to cluster mutations (requires mutation rate file)",
)
@click.option(
    "--skip-sort/--no-skip-sort",
    type=bool,
    default=False,
    help="VCF ONLY: Skip sorting VCF file (use only if already sorted)",
)
@click.option(
    "-fa",
    "--fasta",
    type=click.Path(exists=True),
    default=None,
    help="Reference FASTA file (overrides dataset default)",
)
@click.option(
    "-cn",
    "--copy-number",
    type=click.Path(exists=True),
    default=None,
    help="BED file containing copy number variations for the sample",
)
def _add_sample(*args, **kwargs):
    """
    Add a sample with genomic variations to a G-Tensor dataset.
    
    DATASET is the G-Tensor file to modify.
    SAMPLE_FILE is a VCF file containing variants for the sample.
    
    This command ingests genomic variations (SNVs, indels) from VCF files
    and maps them to the G-Tensor's genomic regions. The variations are
    processed according to the dataset's modality (e.g., SBS components).
    
    For VCF files, various filtering and processing options are available:
    - Filter by PASS status
    - Add sample weights
    - Cluster mutations using mutation rates
    - Handle copy number variations
    
    Example:
        gtensor sample add dataset.nc sample.vcf -id SAMPLE_001 \\
                          -m mutation_rates.bed --cluster
    """
    from .gensor_core import add_sample

    add_sample(*args, **kwargs)


@samplecmds.command("rm", short_help="Remove samples from a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
@click.argument("sample_names", type=str, nargs=-1, required=True, metavar="SAMPLE...")
def _rm_samples(
    dataset,
    sample_names: List[str],
):
    """
    Remove one or more samples from a G-Tensor dataset.

    DATASET is the G-Tensor file to modify.
    SAMPLE... are the names/IDs of samples to remove.

    This permanently deletes the specified samples and their data
    from the dataset. Use with caution.

    Example:
        gtensor sample rm dataset.nc SAMPLE_001 SAMPLE_002
    """
    from .gensor_core import remove_gtensor_samples

    remove_gtensor_samples(dataset, sample_names)
    for sample_name in sample_names:
        click.echo(f"Removed sample: {sample_name}")


@samplecmds.command("ls", short_help="List samples in a G-Tensor")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET")
def _list_samples(dataset: str):
    """
    List all samples in a G-Tensor dataset.

    DATASET is the G-Tensor file to examine.

    Displays the names/IDs of all samples contained in the dataset.

    Example:
        gtensor sample ls dataset.nc
    """
    from .gensor_core import list_gtensor_samples

    samples = list_gtensor_samples(dataset)

    if len(samples) == 0:
        click.echo("No samples found in the dataset.")
        return

    print(*samples, sep="\n")


@gtensor_cli.group("utils", short_help="Utility functions for G-Tensors")
def utils():
    """
    Utility functions for working with genomic data files and G-Tensors.

    These commands provide helpful data processing and format conversion
    utilities for preparing data for G-Tensor ingestion.
    """
    pass


@utils.command("query-gtf", short_help="Query and filter GTF/GFF files")
@click.option(
    "--input",
    "-i",
    type=click.File("r"),
    default="-",
    help="Input GTF or GFF file (default: stdin)",
)
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout)",
)
@click.option(
    "--type-filter",
    "-type",
    help="Only include records of this feature type (e.g., 'gene', 'exon')",
)
@click.option(
    "--attribute-key", "-attr", help="Only include records with this attribute key"
)
@click.option(
    "--attribute-values",
    "-vals",
    multiple=True,
    default=None,
    help="Only include records with these attribute values for the specified key",
)
@click.option(
    "--is-gff/--is-gtf",
    default=False,
    help="Input file is in GFF format (default: GTF)",
)
@click.option(
    "--header/--no-header",
    default=False,
    help="Include header line with column names in output",
)
@click.option(
    "--format-str",
    "-f",
    default=None,
    help="Custom format string for output. Use {column_name} for fields, {attributes[key]} for attributes",
)
@click.option(
    "--as-regions",
    is_flag=True,
    help="Output in regions format: chromosome:start-end",
)
@click.option(
    "--as-gtf",
    is_flag=True,
    help="Output in GTF format",
)
@click.option(
    "--zero-based/--one-based",
    default=False,
    is_flag=True,
    help="Use 0-based coordinates instead of 1-based (default: 1-based)",
)
def _query_gtf(*args, **kwargs):
    """
    Query and filter GTF/GFF annotation files.
    
    This utility allows filtering and reformatting GTF/GFF files based on
    feature types, attributes, and values. It's useful for extracting
    specific annotations for G-Tensor feature creation.
    
    The tool supports various output formats including BED-like regions,
    custom formatted text, or filtered GTF files.
    
    Examples:
        # Extract all genes
        gtensor utils query-gtf -i genes.gtf --type-filter gene
        
        # Extract protein-coding genes in BED format
        gtensor utils query-gtf -i genes.gtf --type-filter gene \\
                                --attribute-key gene_type \\
                                --attribute-values protein_coding \\
                                --as-regions
    """
    from mutopia.ingestion.gtf_parsing import query_gtf

    query_gtf(*args, **kwargs)


@utils.command(
    "make-expression-bedfile", short_help="Create expression BED from quantification"
)
@click.argument(
    "quantitation_file", type=click.File("r"), default=sys.stdin, metavar="QUANT_FILE"
)
@click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
    help="Output BED file (default: stdout)",
)
@click.option(
    "--join-on",
    type=click.Choice(["gene_id", "gene_name"]),
    default="gene_id",
    help="Column to join annotation and quantification on",
)
@click.option(
    "--gtf-file",
    type=click.Path(exists=True),
    default=None,
    help="GTF file for gene annotation (downloads MANE if not provided)",
)
def _make_quant_file(
    output,
    quantitation_file: str,
    gtf_file: Optional[str] = None,
    join_on="gene_id",
):
    """
    Create a BED file with gene expression values from quantification files.
    
    QUANT_FILE is a TSV where the first column are Gene IDs
    and the second column are expression values (e.g., TPM, FPKM).
    
    This utility combines gene annotation from GTF files with expression
    quantification to create BED files suitable for adding as continuous
    features to G-Tensors.
    
    The output BED file contains genomic coordinates for genes along with
    their expression values, which can then be ingested as continuous
    features in G-Tensor datasets.
    
    Example:
        gtensor utils make-expression-bedfile sample1.quant sample2.quant \\
                                             -o expression.bed --join-on gene_id
    """
    from .gensor_core import make_expression_bedfile
    make_expression_bedfile(quantitation_file, output, gtf_file, join_on)


@utils.command("download-gex-annotation", short_help="Create GEX annotation BED file")
@click.option(
    "-gtf",
    "--gtf-file",
    type=click.Path(exists=True),
    default=None,
    help="GTF file for gene annotation (default: downloads MANE GTF)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True),
    default=None,
    help="Output BED file for GEX annotation (default: gex_annotation.bed)",
)
def _make_annotation_bedfile(gtf_file=None, output=None):
    from .gensor_core import make_annotation_bedfile

    make_annotation_bedfile(gtf_file, output)


@utils.command("fill-template", short_help="Fill a configuration template using Jinja2")
@click.argument("template_file", type=click.Path(exists=True), metavar="TEMPLATE_FILE")
@click.argument("variables_file", type=click.Path(exists=True), metavar="VARIABLES_FILE")
def fill_template(template_file: str, variables_file: str) -> None:
    import yaml
    from mutopia.utils import fill_jinja_template

    with open(variables_file, "r") as vf:
        variables = yaml.safe_load(vf)
    
    with open(template_file, "r") as tf:
        template_str = tf.read()

    filled_template = fill_jinja_template(template_str, **variables)
    click.echo(filled_template)