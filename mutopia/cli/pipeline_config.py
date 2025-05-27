from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from ..utils import FeatureType
from ..ingestion import FileType

class URLConfig(BaseModel):
    celltype: Optional[str] = Field(None, description="Name of the data source")
    url: str = Field(..., description="URL to the data source")
    
    def __str__(self) -> str:
        return f"{self.source}: {self.url}"

class FeatureConfig(BaseModel):
    normalization: str = Field(..., description="Type of normalization to apply")
    column: Optional[int] = Field(
        None, 
        description="Column number to use from input file (if applicable)"
    ) 
    group: Optional[str] = Field(
        None,
        description="Group name for the feature"
    )
    null: Optional[str] = Field(
        None,
        description="Value to use for null values"
    )
    classes: Optional[List[str]] = Field(
        None,
        description="List of classes for the feature"
    )
    urls: List[URLConfig] = Field(
        ..., 
        description="Mapping of source names to URLs for feature data"
    )
    
    @property
    def file_type(self) -> FileType:
        """Determine file type from the first URL's extension"""
        return FileType.from_extension(self.urls[0].url)
    
    def validate_extension(self) -> None:
        """Validate that the file type supports the chosen normalization"""
        file_type = self.file_type
        norm_type = FeatureType(self.normalization)
        if norm_type not in file_type.allowed_normalizations:
            raise ValueError(
                f"Normalization {norm_type} not allowed for file type {file_type}. "
                f"Allowed normalizations: {file_type.allowed_normalizations}"
            )
    
    def model_post_init(self, *args, **kwargs) -> None:
        """Validate after model initialization"""
        super().model_post_init(*args, **kwargs)
        self.validate_extension()


class GTensorConfig(BaseModel):
    name: str = Field(..., description="Name of the GTensor object")
    dtype: str = Field(..., description="Data type for the GTensor")
    chromsizes: str = Field(..., description="Path to chromosome sizes file")
    blacklist: str = Field(..., description="Path to blacklist file")
    fasta: str = Field(..., description="Path to FASTA file")
    region_size: int = Field(..., description="Size of regions")
    min_region_size : Optional[int] = Field(
        100,
        description="Minimum size of regions"
    )
    features: Dict[str, FeatureConfig] = Field(
        ..., 
        description="Dictionary of features to process"
    )
    
    @property
    def bed_cuts(self) -> list[tuple[str, str]]:
        """Get list of BED files to be cut during GTensor creation"""
        cuts = []
        for name, feature in self.features.items():
            if feature.file_type == FileType.BED:
                cuts.extend((name, url) for url in feature.urls.values())
        return cuts 