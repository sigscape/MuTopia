import sparse
import numpy as np
from joblib import Parallel
from contextlib import contextmanager
from enum import Enum
import inspect
from functools import wraps
import time
import logging

logger = logging.getLogger(' Mutopia')
logger.setLevel(logging.INFO)

'''def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        logger.debug(f'{func.__name__} took {time.time()-start:.2f} seconds.')
        return res
    return wrapper'''


def str_wrapped_list(x, n=4):
    x = list(x)
    return ",\n\t".join([', '.join(x[i:i+n]) for i in range(0, len(x), n)])


def borrow_kwargs(*borrow_sigs):
    
    def decorator(func):
        
        # start the signature with the incipient function
        combined_params = dict((
            (name, param)
            for name, param in inspect.signature(func).parameters.items()
            if not param.kind == inspect.Parameter.VAR_KEYWORD
        ))

        # iterate over the borrowed functions
        for f in borrow_sigs:
            sig = inspect.signature(f)
            # add the kwargs from the borrowed function
            for name, param in sig.parameters.items():
                if not param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    continue
                combined_params[name] = param
            
        # Create a new signature with combined parameters
        merged_signature = inspect.Signature(parameters=combined_params.values())

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Update the wrapper's signature
        wrapper.__signature__ = merged_signature
        return wrapper

    return decorator


class FeatureType(Enum):
    MESOSCALE = 'mesoscale'
    STRAND = 'strand'
    CATEGORICAL = 'categorical'
    POWER = 'power'
    MINMAX = 'minmax'
    QUANTILE = 'quantile'
    STANDARDIZE = 'standardize'
    ROBUST = 'robust'

    @property
    def is_continuous(self):
        return self in (
            FeatureType.POWER, 
            FeatureType.MINMAX, 
            FeatureType.QUANTILE, 
            FeatureType.STANDARDIZE, 
            FeatureType.ROBUST
        )
    
    @property
    def allowed_dtypes(self):
        if self in (FeatureType.MESOSCALE, FeatureType.CATEGORICAL):
            return (str, np.str_, int, np.int_)
        elif self == FeatureType.STRAND:
            return (np.int8, np.int_)
        elif self.is_continuous:
            return (float, np.float64, np.float32, np.double, int, np.int_)
        else:
            raise ValueError(f'FeatureType {self} not recognized.')


@contextmanager
def ParContext(n_jobs, verbose=0):
    yield Parallel(
        n_jobs=n_jobs, 
        backend='threading', 
        return_as='generator', 
        verbose=verbose,
        pre_dispatch='n_jobs',
    )


def check_dims(corpus, model_state):
    rm_dim = dims_except_for(
        corpus.X.dims,
        *model_state.requires_dims,
    )
    if not len(rm_dim) == 0:
        logger.warning(
            f'The corpus {corpus.attrs["name"]} has extra dimensions: {", ".join(rm_dim)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
            'Having extra data dimensions will increase training time and memory usage,\n'
            'remove them by summing over them: `corpus.sum(dim="extra_dim", keep_attrs=True)`.'
        )
    
    missing_dims = set(model_state.requires_dims).difference(corpus.X.dims + ('sample',))
    if not len(missing_dims) == 0:
        raise ValueError(
            f'The corpus {corpus.attrs["name"]} is missing the following dimensions: {", ".join(missing_dims)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
        )
    

def check_structure(corpus):
    structure = {
        'regions' : [
            'chrom',
            'start',
            'end',
            'length',
            'context_frequencies',
            'exposures'
        ],
        'features' : [],
        'obsm' : [],
        'varm' : [],
    }

    if not 'name' in corpus.attrs:
        raise ValueError('The corpus is missing a name attribute.')
    
    if not 'X' in corpus:
        raise ValueError('The corpus is missing the data matrix `X`.')

    for key, value in structure.items():
        if not hasattr(corpus, key):
            raise ValueError(f'The corpus is missing the {key} node.')
        
        for subkey in value:
            if not hasattr(getattr(corpus, key), subkey):
                raise ValueError(f'The corpus is missing the {key}.{subkey} node.')



def check_sample_data(corpus, dtype):

    X = corpus.X.data
    dims = corpus.X.dims

    if isinstance(X, sparse.SparseArray):
        if isinstance(X, sparse.COO):
            X = sparse.GCXS(X)

        compress = (dims.index('sample'), dims.index('locus')) if 'sample' in dims \
                else (dims.index('locus'),)
        
        if not X.compressed_axes == compress:
            X = X.change_compressed_axes(compress)

    if not X.dtype == dtype:
        X = X.astype(dtype)

    corpus.X.data = X
    return corpus


def check_feature(feature):

    if not 'normalization' in feature.attrs:
        raise ValueError('The feature is missing a normalization attribute.')
    
    normalization = feature.attrs['normalization']
    try:
        FeatureType(normalization)
    except ValueError:
        raise ValueError(
            f'Normalization type {normalization} not recognized. '
            f'Please use one of {", ".join(FeatureType.__members__)}'
        )
    

    allowed_types = FeatureType(normalization).allowed_dtypes
    dtype = feature.data.dtype

    if not any(np.issubdtype(dtype, t) for t in allowed_types):
        raise ValueError(
            f'The feature has dtype {dtype.dtype} but must be one of {", ".join(map(repr, allowed_types))}.'
        )


def check_corpus(corpus):

    check_structure(corpus)
    check_sample_data(corpus, float)

    for feature in corpus.features.values():
        check_feature(feature)    


def dims_except_for(dims, *keepdims):
    return tuple({*dims}.difference({*keepdims}))


def match_dims(y, X):
    data_dims = y.sizes
    return X.expand_dims({
        d : data_dims[d]
        for d in dims_except_for(data_dims.keys(), *X.dims)
    })


def using_exposures_from(corpus):
    try:
        corpus.obsm['exposures']
    except KeyError:
        raise KeyError('The corpus does not have exposures. Run `model.annot_exposures(corpus)` first.')

    return lambda sample_name : \
        corpus.obsm['exposures'].sel(sample=sample_name).data