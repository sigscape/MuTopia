import sparse
import numpy as np
from pandas import DataFrame
from joblib import Parallel
from contextlib import contextmanager
from enum import Enum
import inspect
from functools import wraps
from collections import defaultdict
import logging
import subprocess
from gzip import open as gzopen
import time


@contextmanager
def safe_read(filename):
    yield gzopen(filename, 'rt') if filename.endswith('.gz') \
        else open(filename, 'r')


logger = logging.getLogger(' Mutopia')
logging.basicConfig(level = logging.INFO)
logger.setLevel(logging.INFO)


def timer_wrapper(func, name=None):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.debug(f'{name or func.__name__} took {time.time() - start:.2f} seconds.')
        return result

    return wrapper


class FeatureType(Enum):
    LOG1P_CPM = 'log1p_cpm'
    MESOSCALE = 'mesoscale'
    STRAND = 'strand'
    CATEGORICAL = 'categorical'
    POWER = 'power'
    MINMAX = 'minmax'
    QUANTILE = 'quantile'
    STANDARDIZE = 'standardize'
    ROBUST = 'robust'

    @classmethod
    def continuous_types(cls):
        return (
            FeatureType.LOG1P_CPM,
            FeatureType.POWER, 
            FeatureType.MINMAX, 
            FeatureType.QUANTILE, 
            FeatureType.STANDARDIZE, 
            FeatureType.ROBUST
        )

    @property
    def is_continuous(self):
        return self in self.continuous_types()
    
    @property
    def allowed_dtypes(self):
        if self in (FeatureType.MESOSCALE, FeatureType.CATEGORICAL):
            return (str, np.str_, int, np.int_)
        elif self == FeatureType.STRAND:
            return (np.int8, np.int32, np.int_)
        elif self.is_continuous:
            return (float, np.float64, np.float32, np.double, int, np.int_)
        else:
            raise ValueError(f'FeatureType {self} not recognized.')
        
    @property
    def save_dtype(self):
        if self in (FeatureType.MESOSCALE, FeatureType.CATEGORICAL):
            return np.str_
        elif self == FeatureType.STRAND:
            return np.int8
        elif self.is_continuous:
            return np.float32
        else:
            raise ValueError(f'FeatureType {self} not recognized.')


def str_wrapped_list(x, n=4):
    x = list(map(str, x))
    return "\n\t" + ",\n\t".join([', '.join(x[i:i+n]) for i in range(0, len(x), n)])


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
    }

    if not 'name' in corpus.attrs:
        raise ValueError('The corpus is missing a name attribute.')
    
    for key, value in structure.items():
        if not hasattr(corpus, key):
            raise ValueError(f'The corpus is missing the {key} node.')
        
        for subkey in value:
            if not hasattr(getattr(corpus, key), subkey):
                raise ValueError(f'The corpus is missing the {key}.{subkey} node.')



def check_sample_data(corpus, dtype):

    if not hasattr(corpus, 'X'):
        raise ValueError('The corpus is missing the data matrix `X`.')

    X = corpus.X.data
    dims = corpus.X.dims

    if isinstance(X, sparse.SparseArray):
        if isinstance(X, sparse.COO):
            X = sparse.GCXS(X)

        compress = (dims.index('sample'), dims.index('locus')) if 'sample' in dims \
                else (dims.index('locus'),)
        
        if not X.compressed_axes == compress:
            X = X.change_compressed_axes(compress)

    else:
        X = np.ascontiguousarray(X)

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

    if not dtype in allowed_types and not any(np.issubdtype(dtype, t) for t in allowed_types):
        raise ValueError(
            f'The feature {feature.name} has dtype {dtype} but must be one of {", ".join(map(repr, allowed_types))}.'
        )


def check_corpus(corpus, enforce_sample=True):

    check_structure(corpus)
    
    if enforce_sample:
        check_sample_data(corpus, np.float32)

    for feature in corpus.features.values():
        check_feature(feature)


def check_feature_consistency(*corpuses):

    type_dict = defaultdict(set)
    for feature_name, dtype in list(
        (feature_name, FeatureType(feature.attrs['normalization']))
        for corpus in corpuses
        for feature_name, feature in corpus.features.data_vars.items()
    ):
        type_dict[feature_name].add(dtype)

    for feature_name, types in type_dict.items():
        if not len(types) == 1:
            raise ValueError(
                f'The feature {feature_name} has inconsistent normalization types across corpuses: {str_wrapped_list(types)}'
            )
        
    def _get_classes(corpus, feature):
        try:
            return feature.attrs['classes']
        except KeyError as err:
            raise KeyError(
                f'The feature {feature.name} in corpus {corpus.attrs["name"]} is missing the `classes` attribute.'
            ) from err


    priority_dict = defaultdict(list)
    for feature_name, classes in list(
        (feature_name, tuple(_get_classes(corpus, feature)))
        for corpus in corpuses
        for feature_name, feature in corpus.features.data_vars.items()
        if FeatureType(feature.attrs['normalization']) in (FeatureType.CATEGORICAL, FeatureType.MESOSCALE)
    ):
        priority_dict[feature_name].append(classes)

    for feature_name, priorities in priority_dict.items():
        if not all(p == priorities[0] for p in priorities):
            raise ValueError(
                f'The feature {feature_name} has inconsistent class priorities across corpuses:\n\t' \
                    + '\n\t'.join(map(
                        lambda p : ', '.join(map(str, p)), 
                        priorities
                    ))
            )
        
    corpus_membership = {
        corpus.attrs['name'] : {feature_name for feature_name in corpus.features.data_vars.keys()}
        for corpus in corpuses
    }
    shared_features = set.intersection(*corpus_membership.values())

    for corpus_name, features in corpus_membership.items():
        extra_features = features.difference(shared_features)
        if len(extra_features) > 0:
            logger.warning(
                f'The corpus {corpus_name} has extra features: {", ".join(extra_features)}.\n'
                'Extra features will be ignored during training.'
            )
        

def check_dim_consistency(*corpuses):
    pass


def dims_except_for(dims, *keepdims):
    return tuple({*dims}.difference({*keepdims}))


def match_dims(X,**dim_sizes):
    return X.expand_dims({
        d : dim_sizes[d]
        for d in dims_except_for(dim_sizes.keys(), *X.dims)
    })


def using_exposures_from(corpus):
    try:
        corpus.contributions
    except AttributeError:
        raise AttributeError('The corpus does not have contributions. Run `model.annot_contributions(corpus)` first.')

    return lambda _, sample_name : \
        corpus.contributions.sel(sample=sample_name).data


def using_priors_from(model_state):
    return lambda corpus, _ : model_state.alpha[corpus.attrs['name']]


def get_explanation(corpus, component):

    try:
        import shap
    except ImportError:
        raise ImportError('SHAP is required to calculate SHAP values')
    
    if not component in corpus.varm['SHAP_values'].shap_component.values:
        raise ValueError(f'The corpus does not have SHAP values for component {component}.')

    shap_values = corpus.varm['SHAP_values']

    shap_df = (
        DataFrame(
            shap_values.sel(shap_component=component).data,
            columns=shap_values.transformed_features.values,
        )
        .melt(
            ignore_index=False, 
            var_name='feature', 
            value_name='value'
        )
        .reset_index()\
        .rename(columns={'index':'locus'})
    )
    
    # handle this case to remove the convolution
    if any(shap_df.feature.str.contains(':')):
        shap_df[['feature', 'convolution']] = (
            shap_df.feature.str
            .split(':', expand=True, n=1)
            .rename(columns={0: 'feature', 1: 'convolution'})
        )

    shap_df = (
        shap_df
        .groupby(['feature', 'locus'])['value']
        .sum()
        .unstack()
        .fillna(0)
        .T
    )

    expl = shap.Explanation(
        shap_df.values,
        feature_names=shap_df.columns,
        data=corpus.state.locus_features.sel(
            feature=[
                f'{s}:0' if f'{s}:0' in corpus.state.feature.values else s
                for s in shap_df.columns
            ]
        ).values
    )

    return expl


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
        
        # Sort the parameters so that the order is valid
        sorted_params = sorted(
            combined_params.values(), 
            key=lambda p: (p.kind, p.default is not inspect.Parameter.empty)
        )
        combined_params = {param.name: param for param in sorted_params}
        # Create a new signature with combined parameters
        merged_signature = inspect.Signature(parameters=combined_params.values())

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Update the wrapper's signature
        wrapper.__signature__ = merged_signature
        return wrapper

    return decorator


def close_process(process):

    if not process.stdout is None:
        process.stdout.close()
    process.wait()

    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, 
            process.args,
        )

def stream_subprocess_output(process):
    
    while True:
        line = process.stdout.readline().strip()
        if not line:
            break
        yield line

    close_process(process)
