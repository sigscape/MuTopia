from enum import Enum
from .features import *
from ..utils import FeatureType


class FileType(Enum):
    BIGWIG = 'bigwig'
    BEDGRAPH = 'bedgraph'
    BED = 'bed'
    
    @property
    def allowed_normalizations(self):
        if self == FileType.BIGWIG:
            return FeatureType.continuous_types()
        elif self == FileType.BEDGRAPH:
            return FeatureType.continuous_types()
        elif self == FileType.BED:
            return (
                FeatureType.MESOSCALE,
                FeatureType.CATEGORICAL,
                *FeatureType.continuous_types(),
            )
    

    @classmethod
    def from_extension(cls, filename):
        exts = {
            FileType.BIGWIG: ['.bw', '.bigwig'],
            FileType.BEDGRAPH: ['.bedgraph', '.bg','.bg.gz','.bedgraph.gz'],
            FileType.BED: ['.bed', '.bed.gz']
        }
        for k, v in exts.items():
            if any(filename.lower().endswith(ext) for ext in v):
                return k
        else:
            raise ValueError(f'Unknown extension: {filename}')
    

    def get_ingestion_fn(self,
        is_distance_feature=False,
        is_discrete_feature=False
    ):
        if self == FileType.BIGWIG:
            return make_continous_features_bigwig
        elif self == FileType.BEDGRAPH:
            return make_continous_features_bedgraph
        elif self == FileType.BED:
            if is_distance_feature:
                return make_distance_features
            elif is_discrete_feature:
                return make_discrete_features
            else:
                return make_continuous_features_bed
        else:
            raise ValueError(f'Unknown file type: {self}')

