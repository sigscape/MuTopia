from .sbs import SBSMode
from .fragment_motif import InFragmentMotif, OutFragmentMotif
from .fragment_length import FragmentLength
from enum import Enum

def get_mode(corpus):
    return Modality(
        corpus.attrs['dtype'].upper()
    ).get_config()


class Modality(Enum):
    
    SBS = 'SBS'
    FRAGMENT_MOTIF_OUT5P = 'FRAGMENT_MOTIF_OUT5P'
    FRAGMENT_MOTIF_IN5P = 'FRAGMENT_MOTIF_IN5P'
    FRAGMENT_LENGTH = 'FRAGMENT_LENGTH'

    def get_config(self):
        if self == Modality.SBS:
            return SBSMode()
        elif self == Modality.FRAGMENT_MOTIF_IN5P:
            return InFragmentMotif()
        elif self == Modality.FRAGMENT_MOTIF_OUT5P:
            return OutFragmentMotif()
        elif self == Modality.FRAGMENT_LENGTH:
            return FragmentLength()
        else:
            raise ValueError(f'Unknown modality: {self}')