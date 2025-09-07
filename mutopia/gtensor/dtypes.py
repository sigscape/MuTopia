import sys
import os
from functools import cache
from xarray import register_dataset_accessor, register_dataarray_accessor


@cache
def load_mode_config(filename):

    import importlib.util

    module_path, class_name = filename.rsplit(":", 1)

    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise ImportError(f"No module named '{module_path}'")
    module = importlib.util.module_from_spec(spec)
    
    sys.modules["modality"] = module
    spec.loader.exec_module(module)

    return getattr(module, class_name)


@cache
def get_mode_config(mode_name: str):

    import mutopia.modalities

    try:
        return getattr(mutopia.modalities, mode_name.upper())
    except AttributeError:
        pass

    try:
        return load_mode_config(mode_name)
    except Exception as e:
        raise ValueError(
            f"Cannot load mode config for {mode_name}. "
            "Please provide a valid mode name or path to a config file."
        ) from e


def get_mode(dataset):
    return get_mode_config(dataset.attrs["dtype"])


@register_dataset_accessor("modality")
@register_dataarray_accessor("modality")
class ModalityAccessor:
    def __init__(self, xrds):
        self._xrds = xrds

    def __call__(self):
        return get_mode(self._xrds)
