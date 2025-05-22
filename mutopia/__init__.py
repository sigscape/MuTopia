import mutopia.gtensor as gt
import mutopia.model as model
import mutopia.plot as pl
import mutopia.tuning as tune
import mutopia.tools as tl
from mutopia.utils import (
    FeatureType, 
    diverging_palette, 
    categorical_palette, 
    using_exposures_from
)
import mutopia.mixture_model as mm
from mutopia.modalities import SBS
from mutopia.dtypes import get_mode, MODALITIES
from joblib import load as load_model

MODALITIES.append(SBS)

__version__ = "1.0.0a5"
