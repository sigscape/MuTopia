import xarray as xr
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
