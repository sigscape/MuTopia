
from .corpus_state import CorpusState as CS
from functools import reduce, partial
from joblib import delayed


def Estep(
    model_state,
    corpuses,
    parallel_context,
    learning_rate = 1.,
    subsample_rate = 1.,    
):
    
    latent_vars_model = model_state.locals_model

    '''
    A suffstats dictionary with the following structure:
    sstats[parameter_name][corpus_name] <- suffstats
    '''
    sstats = model_state.get_sstats_dict(corpuses)
    elbo = 0.

    '''
    Construct the update function for each sample from the corpus
    in a generator fashion. Curry the initial gamma value for the 
    update, but don't call the update function yet - we want to
    pass this expression to the multiprocessing pool.
    '''
    samples = [
        (corpus, sample_name)
        for corpus in corpuses
        for sample_name in corpus.samples.data_vars.keys()
    ]

    updates = (
        partial(
            latent_vars_model._get_update_fn(
                sample=corpus.samples[sample_name],
                corpus=corpus,
                model_state=model_state,
                learning_rate=learning_rate,
                subsample_rate=subsample_rate,
            ),
            CS.fetch_topic_compositions(corpus, sample_name),
        )   
        for (corpus, sample_name) in samples
    )
    
    for (corpus, sample_name), (suffstats, gamma_new, bound) in zip(
        samples, 
        parallel_context(delayed(update)() for update in updates)
    ):
            elbo += bound
            # Update the topic compositions for the sample
            CS.update_topic_compositions(
                corpus, 
                sample_name, 
                gamma_new
            )
            
            # After every sample, we update the suffstats
            for stat, vals in sstats.items():
                getattr(latent_vars_model.reducer, f'reduce_{stat}')(
                    vals[corpus.attrs['name']],
                    corpus=corpus,
                    **suffstats
                )

    return sstats, elbo


def Mstep(
    model_state,
    corpuses,
    sstats,
    learning_rate=1.,
    update_prior=True,
    parallel_context=None,
):
    model_state.update(
        corpuses,
        **sstats,
        learning_rate=learning_rate,
        update_prior=update_prior,
        parallel_context=parallel_context
    )


def score(
    model_state,
    corpuses,
):
    
    bound = lambda corpus, sample_name : \
                model_state.locals_model.bound(
                    CS.fetch_topic_compositions(corpus, sample_name),
                    corpus=corpus,
                    sample=corpus.samples[sample_name],
                    model_state=model_state,
                )
    
    samples = (
        (corpus, sample_name)
        for corpus in corpuses 
        for sample_name in corpus.samples.data_vars.keys()
    )

    return reduce(
        lambda x,y : x+y,
        map(
            lambda x : bound(*x),
            samples
        )
    )


def VI_step(
    *,
    parallel_context,
    model_state,
    corpuses,
    update_prior=True,
):
    
    for corpus in corpuses:
        model_state.update_normalizer(corpus)
    
    args = dict(
        model_state=model_state,
        corpuses=corpuses,
        parallel_context=parallel_context,
    )

    stats, elbo = Estep(**args)
    
    Mstep(sstats=stats,
          update_prior=update_prior,
          **args,
         )

    for corpus in corpuses:
        CS.update_corpusstate(corpus, model_state)

    return elbo


def SVI_step(
    update_prior=True,
    *,
    parallel_context,
    model_state,
    corpuses,
    batch_generator,
    learning_rate,
    subsample_rate,
):

    #use "batch_generator" to slice or subsample the corpuses
    #args = (model_state, batch_generator(*corpuses))
    slices = batch_generator(*corpuses)

    # The normalizer only matters for the E-step, so update it here.
    for slice in slices:
        model_state.update_normalizer(
            slice,
            learning_rate=learning_rate
        )

    args = dict(
        model_state=model_state,
        corpuses=slices,
        parallel_context=parallel_context,
        learning_rate=learning_rate,
    )

    sstats, elbo = Estep(
        **args, 
        subsample_rate=subsample_rate,
    )

    Mstep(
        sstats=sstats,
        update_prior=update_prior,
        **args,
    )

    # Update the corpus states in the ORIGINAL corpuses
    for corpus in corpuses:
        CS.update_corpusstate(corpus, model_state)

    return elbo


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
