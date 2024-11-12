
from .base import ModeConfig
from .sbs import SBSMode, SBSModel
from .fragment_motif import FragmentMotif, MotifModel
from .fragment_length import FragmentLength, FragmentLengthModel

DTYPE_MAP = {
    'sbs' : SBSMode,
    'fragment-motif-out5p' : FragmentMotif,
    'fragment-motif-in5p' : FragmentMotif,
    'fragment-length' : FragmentLength,
}

def register_new_mode(config : ModeConfig):
    DTYPE_MAP[config.MODE_ID] = config
    return DTYPE_MAP
