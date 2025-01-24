from .modalities.sbs import SBSModel
from .utils import borrow_kwargs
from joblib import load as jl_load

@borrow_kwargs(SBSModel)
def MutopiaModel(
    train_corpuses,
    test_corpuses,
    **kwargs,
):
    return train_corpuses[0].modality()\
        .make_model(
            train_corpuses=train_corpuses,
            test_corpuses=test_corpuses,
            **kwargs,
        )


def load_model(path):
    return jl_load(path)