from xarray import register_dataset_accessor, register_dataarray_accessor
import importlib.util
import sys
import os
from functools import cache

MODALITIES = []


@cache
def load_mode_config(filename):

    module_path, class_name = filename.rsplit(":", 1)

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"No such file exists: {module_path}, cannot load mode config."
        )

    spec = importlib.util.spec_from_file_location("modality", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modality"] = module
    spec.loader.exec_module(module)

    return getattr(module, class_name)()


@cache
def get_mode_config(mode_name: str):

    for mode in MODALITIES:
        if mode_name.upper() == mode.MODE_ID.upper():
            return mode

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
