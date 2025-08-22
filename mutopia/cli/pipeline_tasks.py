import luigi
import yaml
import os
import click
from functools import cache

from mutopia.utils import FeatureType, logger
from mutopia.gtensor.disk_interface import (
    NoFeaturesError,
    NoSamplesError,
    list_features,
    list_samples,
)
from .pipeline_config import GTensorConfig
from urllib.parse import urlparse
import shutil
from mutopia.cli.gensor_core import (
    create_gtensor,
    add_continuous_feature,
    add_strand_feature,
    add_discrete_feature,
    add_sample,
)


@cache
def load_config(config_path: str) -> GTensorConfig:
    """Load and validate GTensor configuration from file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return GTensorConfig.model_validate(config_dict)


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
            os.path.join(self.source, self.feature_name).replace("/", ".")
            if not self.source is None
            else self.feature_name
        )

        if not os.path.exists(self.path):
            return False

        try:
            return stored_feature_name in list_features(self.path)
        except (FileNotFoundError, OSError, NoFeaturesError):
            return False


class GTensorSampleTarget(luigi.Target):
    """Target that checks if a sample exists in a GTensor with matching metadata"""

    def __init__(
        self,
        gtensor_path: str,
        sample_id: str,
    ):
        self.path = gtensor_path
        self.sample_id = sample_id

    def exists(self) -> bool:
        """Check if sample exists with matching configuration"""
        if not os.path.exists(self.path):
            return False

        try:
            return self.sample_id in list_samples(self.path)
        except (FileNotFoundError, OSError, NoSamplesError):
            return False


class DownloadTask(luigi.Task):

    url = luigi.Parameter(description="Path to download the file to", significant=True)

    @property
    def download_path(self):
        """Determine the local path to download the file to."""
        parsed_url = urlparse(self.url)
        filename = os.path.basename(parsed_url.path)
        return os.path.join("downloads", filename)

    def output(self):
        return luigi.LocalTarget(self.download_path)

    def run(self):

        import requests
        import tempfile

        # Create downloads directory if it doesn't exist
        os.makedirs("downloads", exist_ok=True)

        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_path = temp_file.name
            try:
                with requests.get(self.url, stream=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)

                # If download successful, move temp file to final location
                shutil.move(temp_path, self.download_path)

            except Exception as e:
                # Clean up temp file if download failed
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                logger.error(f"Download failed: {e}")
                raise


class ExistingFileTask(luigi.Task):
    """Task that checks if a file exists"""

    path = luigi.Parameter(description="Path to the file to check", significant=True)

    def output(self):
        return luigi.LocalTarget(self.path)


def url_or_file_task(file):

    def _is_url(path: str) -> bool:
        """Check if a path is a URL"""
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https", "ftp", "ftps")

    return DownloadTask(file) if _is_url(file) else ExistingFileTask(file)


class CreateGTensorTask(luigi.Task):
    """Task to create a new GTensor object"""

    config_path = luigi.Parameter(
        description="Path to GTensor configuration file", significant=True
    )

    def output(self):
        config = load_config(self.config_path)
        return luigi.LocalTarget(f"{config.name}.nc")

    def requires(self):
        config = load_config(self.config_path)
        return {k: url_or_file_task(v) for k, v in config.bed_cuts}

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
            cutout_regions=[(k, v.path) for k, v in self.input().items()],
            min_region_size=c.min_region_size,
        )


class IngestFeatureTask(luigi.Task):
    """Task to ingest a single feature into an existing GTensor"""

    config_path = luigi.Parameter(
        description="Path to GTensor configuration file", significant=True
    )
    feature_name = luigi.Parameter(description="Name of the feature to ingest")
    idx = luigi.IntParameter(
        description="Idx of the feature to ingest",
    )

    def requires(self):
        # First require the GTensor to exist
        config = load_config(self.config_path)
        file = config.features[self.feature_name].sources[int(self.idx)].file

        requirements = {
            "gtensor": CreateGTensorTask(config_path=self.config_path),
            "file": url_or_file_task(file),
        }
        return requirements

    def output(self):
        config = load_config(self.config_path)
        return GTensorFeatureTarget(
            gtensor_path=f"{config.name}.nc",
            feature_name=self.feature_name,
            source=(config.features[self.feature_name].sources[int(self.idx)].celltype),
        )

    def run(self):
        config = load_config(self.config_path)
        gtensor_path = f"{config.name}.nc"
        fc = config.features[self.feature_name]
        file_config = fc.sources[int(self.idx)]

        # Get the actual file path (either local or downloaded)
        file_path = self.input()["file"].path
        featuretype = FeatureType(fc.normalization)

        kw = {
            "ingest_file": file_path,
            "dataset": gtensor_path,
            "feature_name": self.feature_name,
            "source": file_config.celltype,
            "group": fc.group,
            "normalization": fc.normalization,
            "mesoscale": featuretype == FeatureType.MESOSCALE,
            "null": fc.null,
            "column": fc.column,
            "classes": fc.classes,
        }

        if featuretype.is_continuous:
            add_continuous_feature(**kw)
        elif featuretype == FeatureType.STRAND:
            add_strand_feature(**kw)
        else:
            add_discrete_feature(**kw)


class IngestSampleTask(luigi.Task):

    config_path = luigi.Parameter(
        description="Path to GTensor configuration file", significant=True
    )
    sample_id = luigi.Parameter(
        description="ID of the sample to ingest", significant=True
    )

    def requires(self):
        config = load_config(self.config_path)
        return {
            "gtensor": CreateGTensorTask(config_path=self.config_path),
            "file": url_or_file_task(config.samples[self.sample_id].file),
        }

    def output(self):
        config = load_config(self.config_path)
        return GTensorSampleTarget(
            gtensor_path=f"{config.name}.nc",
            sample_id=self.sample_id,
        )

    def run(self):
        config = load_config(self.config_path)
        gtensor_path = f"{config.name}.nc"
        sample_config = config.samples[self.sample_id]
        file_path = self.input()["file"].path

        params = config.sample_params.model_dump()
        params.update(sample_config.model_dump())
        params.update(
            {
                "dataset": gtensor_path,
                "sample_file": file_path,
                "fasta": config.fasta,
                "sample_id": self.sample_id,
            }
        )
        params.pop("file", None)  # Remove file as it's not needed here
        print(params)
        add_sample(**params)


class GTensorPipeline(luigi.WrapperTask):
    """Main pipeline task that processes all features"""

    config_path = luigi.Parameter(
        description="Path to GTensor configuration file", significant=True
    )

    def requires(self):
        config = load_config(self.config_path)
        features_ingested = [
            IngestFeatureTask(
                config_path=self.config_path,
                feature_name=feature_name,
                idx=idx,
            )
            for feature_name, feature_config in config.features.items()
            for idx, _ in enumerate(feature_config.sources)
        ]
        samples_ingested = [
            IngestSampleTask(config_path=self.config_path, sample_id=sample_id)
            for sample_id in config.samples.keys()
        ]
        return features_ingested + samples_ingested


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
