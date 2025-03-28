import numpy as np
from functools import reduce, partial
from joblib import delayed
from .corpus_state import CorpusState as CS
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


def pearson_residuals(
    model_state,
    corpuses,
    exposures_fn = CS.fetch_topic_compositions,
    *,
    parallel_context,
):
    '''
    This function computes the deviance of the model on the corpuses.
    It is a slow implementation that is useful for debugging and testing,
    and offers more flexibility in what you're testing.

    However, you can't use this to monitor the progress of model training
    because it takes too much time and memory. For this, use the more
    specialized `deviance` function.
    '''

    _update_normalizers(
        model_state, 
        corpuses, 
        parallel_context=parallel_context
    )

    # () -> F(corpus) -> F(exposures) -> y_hat
    get_log_mutation_rate_fn = lambda corpus : partial(
        model_state._log_marginalize_mutrate,
        model_state._get_log_mutation_rate_tensor(
            corpus, 
            parallel_context=parallel_context,
            with_context=True,
        )
    )

    def _sample_squared_residuals(obs, log_marginal_mutrate_fn, exposures):

        log_mutrate = log_marginal_mutrate_fn(exposures)
        
        ysum = obs.data.sum()

        #obs = obs.sum(dim=dims_except_for(log_mutrate.dims, 'locus'))
        
        y_hat = np.exp(log_mutrate + np.log(ysum))#\
            #.sum(dim=dims_except_for(log_mutrate.dims, 'locus'))
            #.transpose(*obs.dims)

        return (obs - y_hat)/np.sqrt(y_hat)


    def _corpus_squared_residuals(corpus):
        
        log_marginal_mutrate_fn = get_log_mutation_rate_fn(corpus)
        
        return _reduce_sum(parallel_context(
            delayed(_sample_squared_residuals)(
                CS.fetch_sample(corpus, sample_name),
                log_marginal_mutrate_fn,
                exposures_fn(corpus, sample_name)
            )
            for sample_name in CS.list_samples(corpus)
        ))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        return _reduce_sum(map(_corpus_squared_residuals, corpuses))