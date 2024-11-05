from .modalities import *
import inspect
from functools import wraps

def _merge_signatures(*functions):
    # Extract parameters from each function and combine them into one signature
    combined_params = {}
    for func in functions:
        sig = inspect.signature(func)
        combined_params.update(sig.parameters)
        
    # Create a new signature with combined parameters
    merged_signature = inspect.Signature(parameters=combined_params.values())
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Update the wrapper's signature
        wrapper.__signature__ = merged_signature
        return wrapper

    return decorator


@_merge_signatures(SBSModel, MotifModel)
def MutopiaModel(
    train_corpuses,
    test_corpuses,
    **kwargs,      
):
    return train_corpuses[0].modality().make_model(
        train_corpuses=train_corpuses,
        test_corpuses=test_corpuses,
        **kwargs,
    )