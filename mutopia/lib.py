from .modalities import *
from .utils import borrow_kwargs
from joblib import load as jl_load

#@borrow_kwargs(SBSModel, MotifModel)
def MutopiaModel(
    train_corpuses,
    test_corpuses,
    **kwargs,
):
    return get_mode(train_corpuses[0])\
        .make_model(
            train_corpuses=train_corpuses,
            test_corpuses=test_corpuses,
            **kwargs,
        )


def load_model(path):
    return jl_load(path)