import luigi
import yaml
import os
import click
from functools import cache

from ..utils import FeatureType, logger
from ..gtensor.disk_interface import (
    NoFeaturesError,
    list_features,
)
from .pipeline_config import GTensorConfig
from urllib.parse import urlparse
from pathlib import Path
import requests
import shutil
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


@cache
def load_config(config_path: str) -> GTensorConfig:
    """Load and validate GTensor configuration from file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return GTensorConfig.model_validate(config_dict)


class DownloadTask(luigi.Task):

    url = luigi.Parameter(
        description="Path to download the file to",
        significant=True
    )

    @property
    def download_path(self):
        """Determine the local path to download the file to."""
        parsed_url = urlparse(self.url)
        filename = os.path.basename(parsed_url.path)
        return os.path.join("downloads", filename)

    def output(self):
        return luigi.LocalTarget(self.download_path)
    
    def run(self):
        
        logger.info(f"Downloading {self.url}")
        os.makedirs("downloads", exist_ok=True)
        
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
        
        with open(self.download_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


class FileExistsTask(luigi.Task):
    """Task that checks if a file exists"""
    path = luigi.Parameter(
        description="Path to the file to check",
        significant=True
    )
    
    def output(self):
        return luigi.LocalTarget(self.path)


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
        # First require the GTensor to exist
        config = load_config(self.config_path)
        file = config.features[self.feature_name].files[int(self.idx)].file
        
        return {
            "gtensor" : CreateGTensorTask(config_path=self.config_path),
            "file" : FileExistsTask(file)
        }
    
    def output(self):
        config = load_config(self.config_path)
        return GTensorFeatureTarget(
            gtensor_path=f"{config.name}.nc",
            feature_name=self.feature_name,
            source=(
                config
                .features[self.feature_name]
                .files[int(self.idx)]
                .celltype
            )
        )
    
    def run(self):
        config = load_config(self.config_path)
        gtensor_path = f"{config.name}.nc"
        fc = config.features[self.feature_name]
        file_config = fc.files[int(self.idx)]
        
        file_path = self.input()["file"].path

        kw = {
            'dataset': gtensor_path,
            'feature_name': self.feature_name,
            'ingest_file' : file_path, 
            'source': file_config.celltype,
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
            for feature_name, feature_config in config.features.items()
            for idx, _ in enumerate(feature_config.files)
        ]

def run_pipeline(
    config_file: str,
    workers=1,
    remote_scheduler=False,
    dry_run=False,
):
    """Run the GTensor pipeline with the given configuration file."""
    
    # Validate configuration first to fail fast if invalid
    try:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
        GTensorConfig.model_validate(config_dict)
    except Exception as e:
        click.echo(f"Error in configuration file:\n{e}", err=True)
        raise click.Abort()
    
    # Run pipeline
    success = luigi.build(
        [GTensorPipeline(config_path=config_file)],
        workers=workers * (not dry_run),
        local_scheduler=not remote_scheduler,
        detailed_summary=True,
        log_level="WARNING",
    )
    
    if not success:
        click.echo("Pipeline failed", err=True)
        raise click.Abort()
