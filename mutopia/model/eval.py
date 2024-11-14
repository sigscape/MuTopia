import numpy as np
import xarray as xr
import sparse as sp
from functools import reduce, partial
from joblib import delayed
from .corpus_state import CorpusState as CS
from ..utils import dims_except_for, match_dims
import warnings

def _reduce_sum(g):
    return reduce(lambda x,y : x+y, g)


def get_n_mutations(
        corpuses,
):
    return _reduce_sum((
        CS.fetch_sample(corpus, sample_name).data.sum()
        for corpus in corpuses 
        for sample_name in CS.list_samples(corpus)
    ))


def perplexity(num_mutations, elbo):
    return np.exp(-elbo/num_mutations)


def deviance(
    model_state,
    corpuses,
    exposures_fn = CS.fetch_topic_compositions,
    *,
    parallel_context,
    ignore_dims=[],
    save_dims=None,
):

    xr_xlogy_sparse_dense = lambda a,b : xr.apply_ufunc(
                            lambda x,y : x * np.nan_to_num(np.log(y), neginf=-1e30),
                            a,b,
                        )

    xr_xlogy_sparse_sparse = lambda a,b : xr.apply_ufunc(
                                lambda x,y : x * np.log(y),
                                a,b,
                            )
    
    def _sample_deviance(obs, marginal_mutrate):
        
        sum_dims = dims_except_for(obs.dims, *save_dims) \
                    if not save_dims is None else None
        
        ysum = obs.data.sum()
        bias = ysum*np.log( marginal_mutrate.data.sum()/ysum )

        ylogy = xr_xlogy_sparse_sparse(obs, obs)\
                    .sum(skipna=True, dim=sum_dims).data
        
        ylogmu = xr_xlogy_sparse_dense(obs, marginal_mutrate)\
                    .sum(skipna=True, dim=sum_dims).data

        return (ylogy - ylogmu + bias)
    
    
    def _corpus_fit_deviance(corpus):

        '''
        set up the marginal mutation rate tensor from the model
        '''
        marg_fn = partial(
            model_state._marginalize_mutrate,
            model_state._get_log_mutation_rate_tensor(
                corpus, 
                parallel_context=parallel_context
            )
        )

        fit_deviance = lambda obs, gamma : _sample_deviance(
                            obs.sum(dim=ignore_dims), 
                            match_dims(corpus, marg_fn(gamma)).sum(dim=ignore_dims)
                        )
        
        return _reduce_sum(parallel_context(
            delayed(fit_deviance)(
                CS.fetch_sample(corpus, sample_name),
                exposures_fn(corpus, sample_name)
            )
            for sample_name in CS.list_samples(corpus)
        ))
    
    
    def _corpus_null_deviance(corpus):

        '''
        set up the background mutation rate tensor
        '''
        background_rate = match_dims(
                            corpus, 
                            corpus.regions.exposures \
                                * corpus.regions.context_frequencies
                        ).sum(dim=ignore_dims)
        
        null_deviance = lambda obs : _sample_deviance(
                            obs.sum(dim=ignore_dims),
                            background_rate
                        )
        
        return _reduce_sum(parallel_context(
            delayed(null_deviance)(CS.fetch_sample(corpus, sample_name))
            for sample_name in CS.list_samples(corpus)
        ))


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        D_fit = _reduce_sum(map(_corpus_fit_deviance, corpuses))
        D_null = _reduce_sum(map(_corpus_null_deviance, corpuses))

    return 10*(1 - D_fit/D_null)
