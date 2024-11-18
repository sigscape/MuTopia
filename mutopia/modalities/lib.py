from .sbs import SBSMode
from .fragment_motif import FragmentMotif
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
        elif self in (Modality.FRAGMENT_MOTIF_OUT5P, Modality.FRAGMENT_MOTIF_IN5P):
            return FragmentMotif()
        elif self == Modality.FRAGMENT_LENGTH:
            return FragmentLength()
        else:
            raise ValueError(f'Unknown modality: {self}')