from typing import Any, Optional, Dict, List, Union
import os
from pydantic import BaseModel, Field
import pydantic
from mutopia.utils import logger
from ..utils import FeatureType
from ..ingestion import FileType


class URLConfig(BaseModel):
    file: str = Field(..., description="URL to the data source")
    celltype: Optional[str] = Field(default=None, description="Name of the data source")

    def __str__(self) -> str:
        return f"{self.celltype}: {self.file}"
    

class ProcessingConfig(BaseModel):
    output_extension: str = Field(
        ..., description="File extension for the processed output"
    )
    function: str = Field(
        ..., description="Run the command on the data, must include wildcard {input} and {output}"
    )

    @pydantic.field_validator("function")
    def validate_function(cls, v):
        if "{input}" not in v or "{output}" not in v:
            raise ValueError("Function must include {input} and {output} wildcards")
        return v

    def run(self, input_file: str, output_file: str) -> None:
        if not output_file.endswith(self.output_extension):
            raise ValueError(
                f"Output file must end with {self.output_extension}, got {output_file}"
            )
        command = self.function.format(input=input_file, output=output_file)
        logger.info(f"Running processing command: {command}")
        import subprocess
        # Use bash with pipefail so that failures anywhere in a pipeline
        # cause a non-zero exit status (and thus raise when check=True).
        # This preserves shell semantics while ensuring pipeline errors
        # are not silently ignored.
        bash_cmd = ["bash", "-o", "pipefail", "-c", command]
        try:
            # capture_output and text=True so stderr is available on exception
            subprocess.run(bash_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_file):
                os.remove(output_file)
            stderr = getattr(e, "stderr", None)
            if stderr:
                logger.error(
                    f"Processing command '{command}' failed with exit code {e.returncode}. stderr:\n{stderr}"
                )
            else:
                logger.error(f"Processing command '{command}' failed with exit code {e.returncode}. No stderr captured.")
            raise

class FeatureConfig(BaseModel):
    normalization: str = Field(..., description="Type of normalization to apply")
    classes: List[str] = Field(
        default=[], description="List of classes for the feature"
    )
    sources: Union[List[URLConfig], List[str]] = Field(
        ..., description="List of URLConfig objects or URLs for feature data"
    )
    column: int = Field(
        default=4, description="Column number to use from input file (if applicable)"
    )
    group: str = Field(default="all", description="Group name for the feature")
    null: Optional[str] = Field(default=None, description="Value to use for null values")
    processing: Optional[str] = Field(
        default=None, description="Name of the processing function to apply to the feature"
    )
    description: Optional[str] = Field(
        default=None, description="Description of the feature"
    )

    def validate_extension(self, file_type: FileType) -> None:
        """Validate that the file type supports the chosen normalization"""
        norm_type = FeatureType(self.normalization)
        if not (norm_type in file_type.allowed_normalizations):
            raise ValueError(
                f"Normalization {norm_type} not allowed for file type {file_type}. "
                f"Allowed normalizations: {file_type.allowed_normalizations}"
            )

    def model_post_init(self, *args, **kwargs) -> None:
        """Validate after model initialization"""
        # convert all sources to URL configs
        self.sources = [URLConfig(file=src) if isinstance(src, str) else src for src in self.sources]
        super().model_post_init(*args, **kwargs)


class SampleParams(BaseModel):
    cluster: Optional[bool] = Field(default=False, description="Whether to cluster the samples")
    chr_prefix: Optional[str] = Field(default="", description="Prefix for chromosome names")
    pass_only: Optional[bool] = Field(
        default=True, description="Whether to only pass the sample through"
    )
    weight_col: Optional[str] = Field(default=None, description="Column to use for weights")
    skip_sort: Optional[bool] = Field(
        default=False, description="Whether to skip sorting the samples"
    )


class SampleConfig(SampleParams):
    url: str = Field(..., description="Path to the sample file (e.g., VCF)")
    sample_weight: float = Field(default=1.0, description="Weight of the sample")
    copy_number: Optional[str] = Field(default=None, description="Path to copy number file")
    sample_name: Optional[str] = Field(
        default=None, description="Name of the sample, for VCFs with multiple samples."
    )

    @property
    def file(self) -> str:
        """Get the sample file path"""
        return self.url

class GenomeConfig(BaseModel):
    chromsizes: str = Field(..., description="Path to chromosome sizes file")
    blacklist: str = Field(..., description="Path to blacklist file")
    fasta: str = Field(..., description="Path to FASTA file")
    mutation_rate_file: Optional[str] = Field(
        default=None, description="Path to mutation rate file"
    )


class GTensorConfig(BaseModel):
    output_prefix: str = Field(..., description="Prefix for output GTensor files")
    name: str = Field(..., description="Name of the GTensor object")
    dtype: str = Field(..., description="Data type for the GTensor")
    region_size: int = Field(..., description="Size of regions")
    min_region_size: int = Field(100, description="Minimum size of regions")
    genome: GenomeConfig = Field(
        ..., description="Genome configuration for GTensor"
    )
    processing: Dict[str, ProcessingConfig] = Field(
        {}, description="Processing functions to apply to features"
    )
    features: Dict[str, FeatureConfig] = Field(
        ..., description="Dictionary of features to process"
    )
    sample_params: SampleParams = Field(
        SampleParams(), description="Default parameters for sample ingestion."
    )
    samples: Dict[str, SampleConfig] = Field(
        {}, description="List of samples to process"
    )

    @property
    def gtensor_file(self) -> str:
        """Get the GTensor output file name"""
        return f"{self.output_prefix}.nc"
    
    def get_filetype(self, feature_name: str) -> FileType:
        """Get the FileType for a given feature"""
        feature: FeatureConfig = self.features[feature_name]
        source_file: List[str] = (
            [src.file for src in feature.sources]
            if feature.processing is None
            else [f".{self.processing[feature.processing].output_extension}"]
        )
        exts = [FileType.from_extension(f) for f in source_file]
        if not all(ext == exts[0] for ext in exts):
            raise ValueError(
                f"Feature '{feature_name}' has mixed file types: {exts}"
            )
        return exts[0]

    @property
    def bed_cuts(self) -> list[tuple[str, str]]:
        """Get list of BED files to be cut during GTensor creation"""
        cuts = []
        for name, feature in self.features.items():
            if self.get_filetype(name) == FileType.BED:
                cuts.extend((name, conf.file) for conf in feature.sources)
        return cuts

    def model_post_init(self, *args, **kwargs) -> None:
        """Validate after model initialization"""
        super().model_post_init(*args, **kwargs)
        for sample_id in self.samples.keys():
            if "/" in sample_id:
                raise ValueError(f"Sample ID '{sample_id}' cannot contain slashes")
            
        for feature_name, feature in self.features.items():
            if feature.processing and not (feature.processing in self.processing):
                raise ValueError(
                    f"Feature '{feature_name}' specifies unknown processing function "
                    f"'{feature.processing}'. Available functions: "
                    f"{list(self.processing.keys())}"
                )
            file_type = self.get_filetype(feature_name)
            feature.validate_extension(file_type)
