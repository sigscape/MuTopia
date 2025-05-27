import luigi
import yaml
import os
import click

from ..utils import FeatureType, logger
from ..gtensor.disk_interface import (
    NoFeaturesError,
    list_features,
)
from .pipeline_config import GTensorConfig
from .core import (
    create_gtensor,
    add_continuous_feature,
    add_strand_feature,
    add_discrete_feature,
)

class GTensorFeatureTarget(luigi.Target):
    """Target that checks if a feature exists in a GTensor with matching metadata"""
    
    def __init__(
        self, 
        gtensor_path: str, 
        feature_name: str,
        source: str,
    ):
        self.path = gtensor_path
        self.feature_name = feature_name
        self.source = source

    def exists(self) -> bool:
        """Check if feature exists with matching configuration"""
        stored_feature_name = (
            os.path.join(self.source, self.feature_name).replace('/','.')
            if not self.source is None 
            else self.feature_name
        )

        if not os.path.exists(self.path):
            return False

        try:
            return stored_feature_name in list_features(self.path)
        except (FileNotFoundError, OSError, NoFeaturesError):
            return False


def load_config(config_path: str) -> GTensorConfig:
    """Load and validate GTensor configuration from file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return GTensorConfig.model_validate(config_dict)


class CreateGTensorTask(luigi.Task):
    """Task to create a new GTensor object"""
    
    config_path = luigi.Parameter(
        description="Path to GTensor configuration file",
        significant=True
    )
    
    def output(self):
        config = load_config(self.config_path)
        return luigi.LocalTarget(f"{config.name}.nc")
    
    def run(self):
        c = load_config(self.config_path)
        
        logger.info(
            "Creating regions with cutouts for: " 
            + ", ".join(set([fn for fn, _ in c.bed_cuts]))
        )
        
        create_gtensor(
            name=c.name,
            dtype=c.dtype,
            output=f"{c.name}.nc",
            genome_file=c.chromsizes,
            blacklist_file=c.blacklist,
            fasta_file=c.fasta,
            region_size=c.region_size,
            cutout_regions=c.bed_cuts,
            min_region_size=c.min_region_size,
        )


class IngestFeatureTask(luigi.Task):
    """Task to ingest a single feature into an existing GTensor"""
    
    config_path = luigi.Parameter(
        description="Path to GTensor configuration file",
        significant=True
    )
    feature_name = luigi.Parameter(
        description="Name of the feature to ingest"
    )
    idx = luigi.Parameter(
        description="Idx of the feature to ingest"
    )

    def requires(self):
        return CreateGTensorTask(config_path=self.config_path)
    
    def output(self):
        config = load_config(self.config_path)
        return GTensorFeatureTarget(
            gtensor_path=f"{config.name}.nc",
            feature_name=self.feature_name,
            source=(
                config
                .features[self.feature_name]
                .urls[int(self.idx)]
                .source
            )
        )
    
    def run(self):
        config = load_config(self.config_path)
        gtensor_path = f"{config.name}.nc"
        fc = config.features[self.feature_name]
        url_config = fc.urls[int(self.idx)]

        kw = {
            'dataset': gtensor_path,
            'feature_name': self.feature_name,
            'ingest_file' : url_config.url, 
            'source': url_config.source,
            'group': fc.group,
            'mesoscale': FeatureType(fc.normalization) == FeatureType.MESOSCALE,
            'null': fc.null,
            'column': fc.column,
            'classes': fc.classes,
        }

        if FeatureType(fc.normalization).is_continuous:
            add_continuous_feature(**kw)
        elif FeatureType(fc.normalization) == FeatureType.STRAND:
            add_strand_feature(**kw)
        else:
            add_discrete_feature(**kw)        


class GTensorPipeline(luigi.WrapperTask):
    """Main pipeline task that processes all features"""
    
    config_path = luigi.Parameter(
        description="Path to GTensor configuration file",
        significant=True
    )
    
    def requires(self):
        config = load_config(self.config_path)
        return [
            IngestFeatureTask(
                config_path=self.config_path,
                feature_name=feature_name,
                idx=idx,
            )
            for feature_name, feature_config in config.features
            for idx, _ in enumerate(feature_config.urls)
        ]

def run_pipeline(
    *,
    config_path: str, 
    workers: int, 
    remote_scheduler: bool, 
    dry_run: bool
):
    """Run the GTensor pipeline with the given configuration file."""
    
    # Validate configuration first to fail fast if invalid
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        GTensorConfig.model_validate(config_dict)
    except Exception as e:
        click.echo(f"Error in configuration file:\n{e}", err=True)
        raise click.Abort()
    
    # Run pipeline
    success = luigi.build(
        [GTensorPipeline(config_path=config_path)],
        workers=workers * (not dry_run),
        local_scheduler=not remote_scheduler,
        detailed_summary=True,
        log_level="WARNING",
    )
    
    if not success:
        click.echo("Pipeline failed", err=True)
        raise click.Abort()
