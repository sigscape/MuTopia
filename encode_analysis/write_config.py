import pydantic
from typing import Any, List, Dict, Iterable, Optional
from download_encode import EncodeExperimentConfig
from mutopia.utils import logger
from mutopia.cli.pipeline_config import FeatureConfig, ProcessingConfig, GenomeConfig

class GTensorConfigPresets(pydantic.BaseModel):
    genome: GenomeConfig
    processing: Dict[str, ProcessingConfig]
    features: Dict[str, FeatureConfig]

class NoExperimentFoundError(Exception):
    pass

def _first_file(experiment: EncodeExperimentConfig, file_format: Optional[str] = None) -> str:
        files = experiment.files.values()
        if file_format is not None:
            files = filter(lambda f : f.file_format == file_format, files)
        first_file = next(iter(files), None)
        if first_file is None:
            raise NoExperimentFoundError(f"No files found for experiment {experiment.accession} with file_format={file_format}")
        return first_file.url

def _experiment_summary(experiment: EncodeExperimentConfig) -> str:
    return "|".join(
        f"{key}={getattr(experiment, key)}"
        for key in ["accession", "assay_title", "target", "biosample_term_id","audit_flag"]
    )

def _get_assay(
    experiments: Iterable[EncodeExperimentConfig],
    assay_title: List[str],
    biosample_term_ids: List[str],
    target: Optional[str] = None,
    file_format: Optional[str] = None,
) -> EncodeExperimentConfig:
    """Get the best experiment for a given assay title and biosample term IDs"""
    filtered_experiments = list(filter(
        lambda e : (
            e.assay_title in assay_title
            and e.biosample_term_id in biosample_term_ids
            and (target is None or e.target == target)
            and e.species == "Homo_sapiens"
            and (file_format is None or any(f.file_format == file_format for f in e.files.values()))
        ),
        experiments,
    ))
    if not filtered_experiments:
        raise NoExperimentFoundError(f"No experiments found for assay_title={assay_title}, target={target} in biosamples {biosample_term_ids}")
    
    # first try exact match
    exact = next(
        filter(
            lambda e : (
                e.assay_title == assay_title[0]
                and e.biosample_term_id == biosample_term_ids[0]
                and e.audit_flag >= 3
            ),
            filtered_experiments,
        ),
        None,
    )
    if exact is not None:
        logger.info(f"Found exact match for assay_title={assay_title[0]}, target={target}, biosample={exact.biosample_term_name}")
        return exact

    priority_match = max(
        filtered_experiments,
        key=lambda e : (
            e.description is not None and "adult" in e.description, # prioritize adult samples
            e.audit_flag >= 3,                             # is a high quality experiment
            -biosample_term_ids.index(e.biosample_term_id), # prioritize biosample term ID order
            -assay_title.index(e.assay_title), # prioritize assay title order
            e.audit_flag, # if we get this far, prefer higher audit flag
        ),
        default=None
    )
    assert priority_match is not None

    logger.info(f"Selected experiment for assay_title={priority_match.assay_title}, target={target}, biosample={priority_match.biosample_term_name}")
    return priority_match


HISTONE_MARKS = {"H3K4me1", "H3K27ac", "H3K4me3", "H3K36me3", "H3K27me3", "H3K9me3"}
GC_FILE = "/Users/allen/data/genomes/hg38/hg38.gc.bed"
REPLICATION_STRAND_FILE = "/Users/allen/data/replication_strand/hg38_replication_strand.bed"

def get_config_dict(
    *,
    encode_configs: List[EncodeExperimentConfig],
    biosample_term_ids: List[str],
    repliseq_term_id: str,
) -> GTensorConfigPresets:

    genome = GenomeConfig(
        blacklist="https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/hg38-blacklist.v2.bed.gz",
        chromsizes="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
        fasta="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa",
    )
    
    processing_functions = {
        "liftover_repliseq" : ProcessingConfig(
            output_extension="bed",
            function="{input} {output}",
        ),
        "extract_encode_gex" : ProcessingConfig(
            output_extension="tsv",
            function="{input} {output}",
        ),
        "extract_atac_peaks" : ProcessingConfig(
            output_extension="bed",
            function="{input} {output}",
        ),
        "get_gene_position" : ProcessingConfig(
            output_extension="bed",
            function="{input} {output}",
        ),
        "get_gc_content" : ProcessingConfig(
            output_extension="bed",
            function="{input} {output}",
        ),
    }
    
    gex: EncodeExperimentConfig = _get_assay(
        encode_configs,
        ["total RNA-seq", "polyA RNA-seq"],
        biosample_term_ids,
    )
    atac: EncodeExperimentConfig = _get_assay(
        encode_configs,
        ["ATAC-seq", "DNase-seq"],
        biosample_term_ids,
        file_format="bed",
    )
    dnase: EncodeExperimentConfig = _get_assay(
        encode_configs,
        ["DNase-seq", "ATAC-seq"],
        biosample_term_ids,
        file_format="bigWig",
    )

    base_features = {
        "GCContent" : FeatureConfig(
            normalization="standardize",
            column=4,
            sources=[GC_FILE],
            processing="get_gc_content",
        ),
        "GeneExpression" : FeatureConfig(
            normalization="gex",
            column=5,
            sources=[_first_file(gex)],
            processing="extract_encode_gex",
            description=_experiment_summary(gex),
        ),
        "GeneStrand" : FeatureConfig(
            normalization="strand",
            column=6,
            sources=[_first_file(gex)],
            processing="extract_encode_gex",
            description=_experiment_summary(gex),
        ),
        "GenePosition" : FeatureConfig(
            normalization="quantile",
            column=4,
            sources=[_first_file(gex)],
            processing="get_gene_position",
            description=_experiment_summary(gex),
        ),
        "ReplicationStrand" : FeatureConfig(
            normalization="strand",
            column=4,
            sources=[REPLICATION_STRAND_FILE],
        ),
        "AccessiblePeak" : FeatureConfig(
            normalization="categorical",
            sources=[_first_file(atac, file_format="bed")],
            processing="extract_atac_peaks",
            description=_experiment_summary(atac),
        ),
        "ChromatinAccessibility" : FeatureConfig(
            normalization="log1p_cpm",
            sources=[_first_file(dnase, file_format="bigWig")],
            description=_experiment_summary(dnase),
        ),
    }

    def _get_phase(experiment: EncodeExperimentConfig) -> str:
        if experiment.description is None:
            raise ValueError("Experiment description is required for Repliseq phase extraction")
        return (
            experiment.description
            .removesuffix(" phase").strip()
        )
    
    repliseq_features = {
        "Repliseq" + _get_phase(experiment) : FeatureConfig(
            normalization="quantile",
            column=4,
            sources=[_first_file(experiment)],
            processing="liftover_repliseq",
        )
        for experiment in filter(
            lambda e : (
                e.biosample_term_id == repliseq_term_id
                and e.assay_title == "Repli-seq"
            ),
            encode_configs,
        )
    }

    if not repliseq_features:
        raise NoExperimentFoundError(f"No Repliseq experiments found for biosample term ID {repliseq_term_id}")
    logger.info(f"Found Repliseq features: {list(repliseq_features.keys())}")

    mark_experiments = {}
    for mark in HISTONE_MARKS:
        try:
            mark_experiments[mark] = _get_assay(
                encode_configs,
                ["Histone ChIP-seq"],
                biosample_term_ids,
                target=mark,
            )
        except NoExperimentFoundError as e:
            logger.error(e)

    histone_features = {
        mark: FeatureConfig(
            normalization="log1p_cpm",
            sources=[_first_file(experiment)],
            description=_experiment_summary(experiment),
        )
        for mark, experiment in mark_experiments.items()
    }

    return GTensorConfigPresets(
        genome=genome,
        processing=processing_functions,
        features={**base_features, **histone_features, **repliseq_features},
    )

def main():
    import argparse
    import yaml
    import sys

    parser = argparse.ArgumentParser(description="Generate GTensor config from ENCODE experiments")
    parser.add_argument(
        "ENCODE_CONFIGS",
        type=str,
        help="Path to YAML file with list of ENCODE experiment configurations",
    )
    parser.add_argument(
        "--biosample-term-ids",
        "-ids",
        type=str,
        nargs="+",
        required=True,
        help="List of biosample term IDs to consider for feature extraction",
    )
    parser.add_argument(
        "--repliseq-term-id",
        "-repliseq",
        type=str,
        required=True,
        help="Biosample term ID to use for Repliseq features",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Path to output GTensor config YAML file",
    )

    args = parser.parse_args()

    with open(args.ENCODE_CONFIGS, "r") as f:
        encode_configs_data: Dict[str, Any] = yaml.safe_load(f)
    
    encode_configs = [
        EncodeExperimentConfig(**data)
        for data in encode_configs_data.values()
    ]

    config_presets = get_config_dict(
        encode_configs=encode_configs,
        biosample_term_ids=args.biosample_term_ids,
        repliseq_term_id=args.repliseq_term_id,
    )

    args.output.write("# GTensor configuration generated from ENCODE experiments\n")
    args.output.write("# biosample term IDs: " + " ".join(map(lambda s : f"'{s}'", args.biosample_term_ids)) + "\n")
    args.output.write(f"# repliseq biosample term ID: '{args.repliseq_term_id}'\n")
    args.output.write("\n")

    yaml.dump(
        config_presets.model_dump(exclude_defaults=True),
        args.output,
        sort_keys=False,
    )

if __name__ == "__main__":
    main()
