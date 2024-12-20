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


def _update_normalizers(model_state, corpuses, *, parallel_context):
    for corpus in corpuses:
        CS.update_normalizers(
            corpus, 
            model_state._calc_normalizers(
                corpus,
                parallel_context=parallel_context,
            )
        )


def _deviance(
    model_state,
    corpuses,
    exposures_fn=CS.fetch_topic_compositions,
    *,
    parallel_context,
):
    dev_fns = model_state.locals_model.get_deviance_fns(
        corpuses,
        model_state,
        exposures_fn=exposures_fn,
        parallel_context=parallel_context,
    )
    
    res = list(parallel_context(delayed(fn)() for fn in dev_fns))
    d_fit, d_null = list(zip(*res))

    return (1 - sum(d_fit)/sum(d_null))*10


def deviance_locus(
    model_state,
    corpuses,
    exposures_fn=CS.fetch_topic_compositions,
    *,
    parallel_context,
):
    # we want to make extra sure the normalizers are up to date
    # before we start computing deviance.
    
    _update_normalizers(model_state, corpuses, parallel_context=parallel_context)

    return _deviance(
        model_state,
        corpuses,
        exposures_fn=exposures_fn,
        parallel_context=parallel_context,
    )


def deviance_samples(
    model_state,
    corpuses,
    *,
    parallel_context,
):
    _update_normalizers(model_state, corpuses, parallel_context=parallel_context)
    
    for corpus in corpuses:
        exposures = model_state.locals_model.predict(
            corpus,
            model_state,
            parallel_context=parallel_context,
        )
        CS.fetch_val(corpus, 'topic_compositions')[...] = exposures.data.T

    return _deviance(
        model_state,
        corpuses,
        parallel_context=parallel_context,
    )


'''
def _deviance_residuals(
    self,
    corpus,
    sample,
    model_state,
    *,
    gamma,
    conditional_likelihood,
    weights,
    log_context_effect,
    sample_dict,
):
    
    contributions = np.ascontiguousarray(gamma/np.sum(gamma))        
    y_sum = np.sum(weights)
    
    log_y = np.log(weights)
    log_pi_hat = ( np.log(conditional_likelihood.T.dot(contributions)) - log_context_effect + np.log(y_sum) )

    resid = np.sqrt(2*(weights * log_y - weights * log_pi_hat)) * np.sign(log_y - log_pi_hat)

    return self._unconvert_sample(sample_dict, resid)
'''

def residuals(
    model_state,
    corpuses,
    exposures_fn=CS.fetch_topic_compositions,
    *,
    parallel_context,
):
    
    _update_normalizers(model_state, corpuses, parallel_context=parallel_context)

    resid_fns = model_state.locals_model.get_residual_fns(
        corpuses,
        model_state,
        exposures_fn=exposures_fn,
        parallel_context=parallel_context,
    )

    return _reduce_sum(parallel_context(delayed(fn)() for fn in resid_fns))



def _slow_deviance(
    model_state,
    corpuses,
    exposures_fn = CS.fetch_topic_compositions,
    *,
    parallel_context,
    ignore_dims=[],
    save_dims=None,
):
    '''
    This function computes the deviance of the model on the corpuses.
    It is a slow implementation that is useful for debugging and testing,
    and offers more flexibility in what you're testing.

    However, you can't use this to monitor the progress of model training
    because it takes too much time and memory. For this, use the more
    specialized `deviance` function.
    '''

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
        ylogy = xr_xlogy_sparse_sparse(obs, obs)\
                    .sum(skipna=True, dim=sum_dims).data \
                    - ysum * np.log(ysum)
        
        ylogmu = xr_xlogy_sparse_dense(obs, marginal_mutrate)\
                    .sum(skipna=True, dim=sum_dims).data \
                    - ysum * np.log(marginal_mutrate.data.sum())

        print(ylogy, ylogmu)
        return (ylogy - ylogmu)
    
    
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
                            match_dims(obs, marg_fn(gamma)).sum(dim=ignore_dims)
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
        example_sample = CS.fetch_sample(corpus, CS.list_samples(corpus)[0])

        background_rate = match_dims(
                            example_sample, 
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
