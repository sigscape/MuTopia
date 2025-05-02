from .sbs import SBSMode
from enum import Enum
from xarray import register_dataset_accessor

class Modality(Enum):
    
    SBS = 'SBS'

    def get_config(self):
        if self == Modality.SBS:
            return SBSMode()
        else:
            raise ValueError(f'Unknown modality: {self}')
        

def get_mode(corpus):
    return Modality(
        corpus.attrs['dtype'].upper()
    ).get_config()


@register_dataset_accessor("modality")
class ModalityAccessor:
    def __init__(self, xrds):
        self._xrds = xrds
        
    def __call__(self):
        return get_mode(self._xrds)