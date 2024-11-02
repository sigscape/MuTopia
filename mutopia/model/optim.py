
from .corpus_state import CorpusState as CS
from functools import reduce, partial
from joblib import Parallel, delayed
from contextlib import contextmanager
from numpy import exp

@contextmanager
def ParContext(n_jobs, verbose=0):
    yield Parallel(
        n_jobs=n_jobs, 
        backend='threading', 
        return_as='generator', 
        verbose=verbose,
        pre_dispatch='n_jobs',
    )


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


def score(
    model_state,
    corpuses,
    locals_weight=1.0,
    subsample_rate=1.0,
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
                    subsample_rate=subsample_rate,
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
    
    return elbo



def VI_step(
    model_state,
    corpuses,
    update_prior=True,
    *,
    test_score_fn,
    parallel_context,
):
    
    args = dict(
        corpuses=corpuses,
        parallel_context=parallel_context,
    )
    
    offsets = model_state.get_exp_offsets_dict(
        **args,
        norm_update_fn=model_state._update_normalizer
    )
    '''
    The normalizers are updated in the previous function
    (because the mutation rates are used to get the offsets, 
    so it's more efficient to use them to update the normalizers there).

    Therefore, this is the best time to evaluate the loss functions.
    The elbo is calculated during the E-step because it's convenient.
    '''
    stats, elbo = model_state.Estep(**args)
    '''
    We're agnostic to the form of the test set evaluation function,
    but it should be a function of the model state that returns the
    test set score.
    '''
    test_elbo = test_score_fn(
        model_state,
        parallel_context=parallel_context
    )
    
    model_state.Mstep(
        offsets=offsets,
        sstats=stats,
        update_prior=update_prior,
        **args,
    )

    for corpus in corpuses:
        CS.update_corpusstate(corpus, model_state)

    return elbo, test_elbo



def SVI_step(
    model_state,
    corpuses,
    update_prior=True,
    *,
    test_score_fn,
    parallel_context,
    batch_generator,
    learning_rate,
    subsample_rate,
):

    # Update the normalizing constants using the whole training set.
    # Otherwise, the updates have too high variance and nothing works.
    model_state.init_normalizers(
        corpuses, 
        parallel_context=parallel_context
    )

    # calculate the bound here because the mutations rates
    # have just been re-normalized during the offset calculation
    test_elbo = test_score_fn(
        model_state, 
        parallel_context=parallel_context
    )

    #use "batch_generator" to slice or subsample the corpuses
    slices = batch_generator(*corpuses)
    args = dict(
        corpuses=slices,
        parallel_context=parallel_context,
    )
    svi_kw = dict(
        learning_rate=learning_rate,
        subsample_rate=subsample_rate,
    )

    # E-step ELBO calculation is unreliable because it's only calculated
    # on a subset of the data.
    sstats, elbo = model_state.Estep(**args, **svi_kw)

    # Get the offsets on the sliced data, 
    # BUT DON'T UPDATE the normalizer using the slice!
    offsets = model_state.get_exp_offsets_dict(
        **args,
        norm_update_fn=lambda *x : None # don't update the normalizer
    )

    model_state.Mstep(
        offsets=offsets,
        sstats=sstats,
        update_prior=update_prior,
        **args,
        **svi_kw,
    )

    # Update the corpus states in the ORIGINAL corpuses
    for corpus in corpuses:
        CS.update_corpusstate(corpus, model_state)

    return elbo, test_elbo


def learning_rate_schedule(epoch, tau=1, kappa=0.5):
    return (tau + epoch)**(-kappa)


def locus_slice_generator(
        random_state,
        *corpuses,
        subsample_rate=0.125,
    ):

    n_loci = corpuses[0].dims['locus']
    
    sel_loci = random_state.choice(
        n_loci,
        int(subsample_rate*n_loci),
        replace=False
    )

    return tuple(
        corpus.isel(locus=sel_loci)
        for corpus in corpuses
    )


def perplexity(num_mutations, elbo):
    return exp(-elbo/num_mutations)
