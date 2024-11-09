import numpy as np
import xarray as xr
import sparse as sp
from functools import reduce, partial
from joblib import delayed
from .corpus_state import CorpusState as CS
from .utils import dims_except_for, match_dims
import warnings

def get_n_mutations(
        corpuses,
):
    samples = (
        (corpus, sample_name)
        for corpus in corpuses 
        for sample_name in corpus.samples.data_vars.keys()
    )

    return reduce(
            lambda x,y : x+y,
            (
            corpus.samples[sample_name].data.sum()
            for corpus, sample_name in samples
            )
    )


def perplexity(num_mutations, elbo):
    return np.exp(-elbo/num_mutations)


'''def elbo_score(
    model_state,
    corpuses,
    locals_weight=1.0,
    exposures_fn = CS.fetch_topic_compositions,
    *,
    parallel_context,
):
    
    bound = lambda corpus, sample_name : \
                model_state.locals_model.bound(
                    exposures_fn(corpus, sample_name),
                    corpus=corpus,
                    sample=corpus.samples[sample_name],
                    model_state=model_state,
                    locals_weight=locals_weight,
                )
    
    samples = (
        (corpus, sample_name)
        for corpus in corpuses 
        for sample_name in corpus.samples.data_vars.keys()
    )

    elbo = reduce(
            lambda x,y : x+y,
            parallel_context(
                delayed(bound)(corpus, sample_name)
                for corpus, sample_name in samples
            )
        )
    
    return elbo'''


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
        
        return sum(parallel_context(
                    delayed(fit_deviance)(obs, exposures_fn(corpus, sample_name))
                    for sample_name, obs in corpus.samples.data_vars.items()
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
        
        return sum(parallel_context(
                    delayed(null_deviance)(obs)
                    for obs in corpus.samples.data_vars.values()
                ))


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        D_fit = sum(map(_corpus_fit_deviance, corpuses))
        D_null = sum(map(_corpus_null_deviance, corpuses))

    return 10*(1 - D_fit/D_null)
