from .mode_config import ModeConfig
from .sbs.sbs import SBSMode

SBS = SBSMode()  # initialize the modality model

__all__ = [
    "ModeConfig",
    "SBS",
]
