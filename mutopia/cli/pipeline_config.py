from typing import Any, Optional, Dict, List, Union
from pydantic import BaseModel, Field
from ..utils import FeatureType
from ..ingestion import FileType


class URLConfig(BaseModel):
    file: str = Field(..., description="URL to the data source")
    celltype: Optional[str] = Field(None, description="Name of the data source")

    def __str__(self) -> str:
        return f"{self.celltype}: {self.file}"


class FeatureConfig(BaseModel):
    normalization: str = Field(..., description="Type of normalization to apply")
    classes: List[str] = Field(
        [], description="List of classes for the feature"
    )
    sources: Union[List[URLConfig], List[str]] = Field(
        ..., description="List of URLConfig objects or URLs for feature data"
    )
    column: Optional[int] = Field(
        4, description="Column number to use from input file (if applicable)"
    )
    group: Optional[str] = Field("all", description="Group name for the feature")
    null: Optional[str] = Field(None, description="Value to use for null values")

    @property
    def file_type(self) -> FileType:
        """Determine file type from the first file's extension"""
        return FileType.from_extension(self.sources[0].file)

    def validate_extension(self) -> None:
        """Validate that the file type supports the chosen normalization"""
        file_type = self.file_type
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
        self.validate_extension()
        super().model_post_init(*args, **kwargs)


class SampleParams(BaseModel):
    chr_prefix: Optional[str] = Field("", description="Prefix for chromosome names")
    pass_only: Optional[bool] = Field(
        True, description="Whether to only pass the sample through"
    )
    weight_col: Optional[str] = Field(None, description="Column to use for weights")
    mutation_rate_file: Optional[str] = Field(
        None, description="Path to mutation rate file"
    )
    skip_sort: Optional[bool] = Field(
        False, description="Whether to skip sorting the samples"
    )
    cluster: Optional[bool] = Field(True, description="Whether to cluster the samples")


class SampleConfig(BaseModel):
    file: str = Field(..., description="Path to the sample file (e.g., VCF)")
    sample_name: Optional[str] = Field(
        None, description="Name of the sample, for VCFs with multiple samples."
    )
    sample_weight: Optional[float] = Field(1.0, description="Weight of the sample")
    copy_number: Optional[str] = Field(None, description="Path to copy number file")


class GTensorConfig(BaseModel):
    name: str = Field(..., description="Name of the GTensor object")
    dtype: str = Field(..., description="Data type for the GTensor")
    chromsizes: str = Field(..., description="Path to chromosome sizes file")
    blacklist: str = Field(..., description="Path to blacklist file")
    fasta: str = Field(..., description="Path to FASTA file")
    region_size: int = Field(..., description="Size of regions")
    min_region_size: Optional[int] = Field(100, description="Minimum size of regions")
    features: Dict[str, FeatureConfig] = Field(
        ..., description="Dictionary of features to process"
    )
    sample_params: Optional[SampleParams] = Field(
        SampleParams(), description="Default parameters for sample ingestion."
    )
    samples: Optional[Dict[str, SampleConfig]] = Field(
        {}, description="List of samples to process"
    )

    @property
    def bed_cuts(self) -> list[tuple[str, str]]:
        """Get list of BED files to be cut during GTensor creation"""
        cuts = []
        for name, feature in self.features.items():
            if feature.file_type == FileType.BED:
                cuts.extend((name, conf.file) for conf in feature.sources)
        return cuts

    def model_post_init(self, *args, **kwargs) -> None:
        """Validate after model initialization"""
        super().model_post_init(*args, **kwargs)
        for sample_id in self.samples.keys():
            if "/" in sample_id:
                raise ValueError(f"Sample ID '{sample_id}' cannot contain slashes")
