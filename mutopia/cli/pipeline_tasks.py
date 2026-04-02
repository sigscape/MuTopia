import luigi
import yaml
import os
import click
from functools import cache
import tempfile
from collections import defaultdict
from typing import Tuple

from mutopia.utils import FeatureType, logger
from mutopia.gtensor.disk_interface import (
    NoFeaturesError,
    NoSamplesError,
    list_features,
    list_samples,
)
from .pipeline_config import GTensorConfig, ProcessingConfig, FeatureConfig
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
        return os.path.join("gtensor__tempfiles/downloads", filename)

    def output(self):
        return luigi.LocalTarget(self.download_path)

    def run(self):

        import requests
        import tempfile

        logger.info(f"Downloading file from {self.url} to {self.download_path}")
        
        # Create downloads directory if it doesn't exist
        os.makedirs("gtensor__tempfiles/downloads", exist_ok=True)

        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                with requests.get(self.url, stream=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)

                    # Ensure all buffered data is written to disk before moving.
                    # On some systems, moving/copying an open file can copy only
                    # the already-flushed portion, resulting in truncated files.
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                # Close the temp file explicitly before moving to ensure all
                # data is visible to the mover (and to avoid platform-specific
                # issues when copying open files across filesystems).
                try:
                    temp_file.close()
                except Exception:
                    # ignore - context manager will close on exit if needed
                    pass

                # If download successful, move temp file to final location
                shutil.move(temp_path, self.download_path)

            except Exception as e:
                # Clean up temp file if download failed
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if os.path.exists(self.download_path):
                    os.unlink(self.download_path)
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


class ProcessFeatureTask(luigi.Task):
    """Task to process a single feature file before ingestion"""

    config_path = luigi.Parameter(
        description="Path to GTensor configuration file", significant=True
    )
    function_name = luigi.Parameter(
        description="Name of the processing function to apply", significant=True
    )
    source_file = luigi.Parameter(
        description="Path to the source feature file", significant=True
    )

    @property
    def processing_config(self) -> ProcessingConfig:
        config = load_config(self.config_path)
        return config.processing[self.function_name]
    
    def requires(self):
        return url_or_file_task(self.source_file)

    def output(self):
        ext = self.processing_config.output_extension
        config = load_config(self.config_path)
        input_filename = os.path.basename(self.input().path)
        processed_path = os.path.join("gtensor__tempfiles", "processed", config.name, f"{input_filename}-{self.function_name}.{ext}")
        return luigi.LocalTarget(processed_path)

    def run(self):
        logger.info(
            f"Processing feature file {self.source_file} with function "
            f"'{self.function_name}'"
        )

        output_file = self.output().path
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        self.processing_config.run(
            input_file=self.input().path,
            output_file=output_file,
        )


def feature_sourcedata_task(
    config_path: str,
    feature_name: str,
    file_path: str,
) -> luigi.Task:
    # First require the GTensor to exist
    config = load_config(config_path)
    feature_config = config.features[feature_name]

    file = (
        url_or_file_task(file_path)
        if feature_config.processing is None
        else ProcessFeatureTask(
            config_path=config_path,
            function_name=feature_config.processing,
            source_file=file_path,
        )
    )
    return file


class CreateGTensorTask(luigi.Task):
    """Task to create a new GTensor object"""

    config_path = luigi.Parameter(
        description="Path to GTensor configuration file", significant=True
    )

    def output(self):
        config = load_config(self.config_path)
        return luigi.LocalTarget(config.gtensor_file)

    def requires(self):
        config = load_config(self.config_path)
        
        cutout_features = {}
        for k, v in config.bed_cuts:
            cutout_features[(k,v)] = feature_sourcedata_task(str(self.config_path), k, v)

        genome_requirements = {
            "fasta": url_or_file_task(config.genome.fasta),
            "chromsizes": url_or_file_task(config.genome.chromsizes),
            "blacklist": url_or_file_task(config.genome.blacklist),
        }
        return {
            "genome" : genome_requirements,
            "cutouts": cutout_features,
        }

    def run(self):
        c = load_config(self.config_path)

        if len(c.bed_cuts) > 0:
            logger.info(
                "Creating regions with cutouts for: "
                + ", ".join(set([fn for fn, _ in c.bed_cuts]))
            )
        logger.info(f"Output GTensor will be: {self.output().path}")

        genome = self.input()["genome"]

        create_gtensor(
            name=c.name,
            dtype=c.dtype,
            output=self.output().path,
            region_size=c.region_size,
            min_region_size=c.min_region_size,
            genome_file=genome["chromsizes"].path,
            blacklist_file=genome["blacklist"].path,
            fasta_file=genome["fasta"].path,
            cutout_regions=[
                (featurename, task.path) 
                for (featurename, _), task in self.input()["cutouts"].items()
            ],
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
        source_filepath = (
            load_config(self.config_path)
            .features[str(self.feature_name)]
            .sources[int(self.idx)]
            .file
        )
        requirements = {
            "gtensor": CreateGTensorTask(config_path=self.config_path),
            "file": feature_sourcedata_task(
                config_path=self.config_path,
                feature_name=self.feature_name,
                file_path=source_filepath,
            ),
        }
        return requirements

    def output(self):
        config = load_config(self.config_path)
        return GTensorFeatureTarget(
            gtensor_path=config.gtensor_file,
            feature_name=self.feature_name,
            source=(config.features[self.feature_name].sources[int(self.idx)].celltype),
        )

    def run(self):
        config = load_config(self.config_path)
        gtensor_path = config.gtensor_file
        feature_config: FeatureConfig = config.features[str(self.feature_name)]
        file_config = feature_config.sources[int(self.idx)]

        # Get the actual file path (either local or downloaded)
        file_path = self.input()["file"].path
        featuretype = FeatureType(feature_config.normalization)

        logger.info(
            f"Ingesting feature '{self.feature_name}' from file '{file_path}' "
            f"into GTensor '{gtensor_path}'"
        )

        kw = {
            "ingest_file": file_path,
            "dataset": gtensor_path,
            "feature_name": self.feature_name,
            "source": file_config.celltype,
            "group": feature_config.group,
            "normalization": feature_config.normalization,
            "mesoscale": featuretype == FeatureType.MESOSCALE,
            "null": feature_config.null,
            "column": feature_config.column,
            "classes": feature_config.classes,
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
            "fasta": url_or_file_task(config.genome.fasta),
        }

    def output(self):
        config = load_config(self.config_path)
        return GTensorSampleTarget(
            gtensor_path=config.gtensor_file,
            sample_id=self.sample_id,
        )

    def run(self):
        config = load_config(self.config_path)
        sample_config = config.samples[str(self.sample_id)]
        file_path = self.input()["file"].path
        gtensor_path = self.input()["gtensor"].path
        fasta_path = self.input()["fasta"].path
        mutation_rate_file = config.genome.mutation_rate_file

        params = config.sample_params.model_dump()
        params.update(sample_config.model_dump(exclude_defaults=True))
        params["file"] = params.pop("url")
        
        params.update(
            {
                "dataset": gtensor_path,
                "sample_file": file_path,
                "fasta": fasta_path,
                "sample_id": self.sample_id,
                "mutation_rate_file": mutation_rate_file,
            }
        )
        params.pop("file", None)  # Remove file as it's not needed here
        logger.info(
            f"Ingesting sample '{self.sample_id}' from file '{file_path}' "
            f"into GTensor '{gtensor_path}'"
        )
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
    output_prefix: str, 
    config_files: Tuple[str, ...],
    *,
    workers=1,
    remote_scheduler=False,
    dry_run=False,
):
    """Run the GTensor pipeline with the given configuration file."""
    config_dict = {
        "output_prefix": output_prefix,
    }
    try:
        for config_file in config_files:
            with open(config_file) as f:
                config_dict.update(yaml.safe_load(f))
        GTensorConfig.model_validate(config_dict)
    except Exception as e:
        click.echo(f"Error in configuration file:\n{e}", err=True)
        raise click.Abort()
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=True) as temp_config:
        yaml.dump(config_dict, temp_config)
        temp_config.flush()
        temp_config.seek(0)

        result = luigi.build(
            [GTensorPipeline(config_path=temp_config.name)],
            workers=workers * (not dry_run),
            local_scheduler=not remote_scheduler,
            detailed_summary=True,
            log_level="WARNING",
        )

        # Luigi returns LuigiStatusCode: SUCCESS, SUCCESS_WITH_RETRY, FAILED, FAILED_AND_SCHEDULING_FAILED, etc.
        # Check if any tasks failed
        from luigi.execution_summary import LuigiStatusCode
        if result.status not in (LuigiStatusCode.SUCCESS, LuigiStatusCode.SUCCESS_WITH_RETRY):
            logger.error("Pipeline failed. Check the logs for details.")
            raise click.Abort()
            
        logger.info("Pipeline completed successfully")
