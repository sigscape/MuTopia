import xarray as xr
import sparse
from .corpus_state import CorpusState as CS
from joblib import Parallel
from contextlib import contextmanager

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
        CS.sample_dims(corpus),
        *model_state.requires_dims,
    )
    if not len(rm_dim) == 0:
        raise ValueError(
            f'The corpus {CS.get_name(corpus)} has extra dimensions: {", ".join(rm_dim)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.requires_dims)}.\n'
            'Having extra data dimensions will increase training time and memory usage,\n'
            'remove them by summing over them: `corpus.sum(dim="extra_dim")`.'
        )
    
    missing_dims = set(model_state.requires_dims).difference(CS.sample_dims(corpus) + ('sample',))
    if not len(missing_dims) == 0:
        raise ValueError(
            f'The corpus {CS.get_name(corpus)} is missing the following dimensions: {", ".join(missing_dims)}.\n'
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
        'layers' : ['X'],
        'obsm' : [],
        'varm' : [],
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

    X = corpus.layers.X.data
    dims = corpus.layers.X.dims

    if isinstance(X, sparse.SparseArray):
        if isinstance(X, sparse.COO):
            X = sparse.GCXS(X)

        compress = (dims.index('sample'), dims.index('locus')) if 'sample' in dims \
                else (dims.index('locus'),)
        
        if not X.compressed_axes == compress:
            X = X.change_compressed_axes(compress)

    if not X.dtype == dtype:
        X = X.astype(dtype)

    corpus.layers.X.data = X
    return corpus


def using_exposures_from(*corpuses):
    
    corpus_dict = {CS.get_name(corpus) : corpus for corpus in corpuses}
    
    return lambda corpus, sample_name : \
        CS.fetch_topic_compositions(
            corpus_dict[CS.get_name(corpus)],
            sample_name
        )

def dims_except_for(dims, *keepdims):
    return tuple({*dims}.difference({*keepdims}))


def match_dims(corpus, X):
    data_dims = corpus.modality().dims
    return X.expand_dims({
        d : data_dims[d]
        for d in dims_except_for(data_dims.keys(), *X.dims)
    })


def split_by_chrom(
    corpus,
    test_chroms = ('chr1',),
):
    corpus=corpus.load()
    train_mask = ~(corpus.regions.chrom.data.isin(test_chroms))
    train = corpus.isel(locus=train_mask)
    test = corpus.isel(locus=~train_mask)

    return train, test
