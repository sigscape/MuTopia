
from .corpus_state import CorpusState as CS
from .eval import *
from .utils import *
import logging
from tqdm import trange

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
)
logger = logging.getLogger(" Mutopia ")


def VI_step(
    model_state,
    corpuses,
    update_prior=True,
    learning_rate=1.,
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

    '''
    Fit the normalizing constants on the whole training corpus - 
    it's really important that this step has low variance.
    '''
    #model_state.init_normalizers(
    #    corpuses, 
    #    parallel_context=parallel_context
    #)

    '''
    Generate the sliced corpuses for the E-step.
    '''
    slices = batch_generator(*corpuses)
    
    args = dict(
        corpuses=slices,
        parallel_context=parallel_context,
    )
    svi_kw = dict(
        learning_rate=learning_rate,
        subsample_rate=subsample_rate,
    )

    '''
    Get the offsets from the sliced data,
    but don't update the normalizers!
    '''
    offsets = model_state.get_exp_offsets_dict(
        **args,
        norm_update_fn=partial(model_state._update_normalizer, **svi_kw),
    )
    
    sstats, elbo = model_state.Estep(**args, **svi_kw)

    '''
    Calculate the bounds here because the normalizers and 
    locals have been updated for some set of model parameters.
    '''
    test_score = test_score_fn(
        model_state, 
        parallel_context=parallel_context
    )

    '''
    Update global model parameters
    '''
    model_state.Mstep(
        offsets=offsets,
        sstats=sstats,
        update_prior=update_prior,
        **args,
        **svi_kw,
    )

    '''
    Update the state of the original corpuses,
    - not the slices.
    '''
    for corpus in corpuses:
        CS.update_corpusstate(corpus, model_state)

    return elbo, test_score


def learning_rate_schedule(tau, kappa, epoch):
    return (tau + epoch)**(-kappa)


def locus_slice_generator(
        random_state,
        *corpuses,
        subsample_rate=0.25,
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


def _should_stop(stop_condition, scores):
    return len(scores) > stop_condition \
        and scores.index(max(scores)) < (len(scores) - stop_condition)


def _check_dims(corpus, model_state):
    rm_dim = dims_except_for(
        CS.sample_dims(corpus),
        model_state.requires_dims,
    )
    if not len(rm_dim) == 0:
        raise ValueError(
            f'The corpus {CS.get_name(corpus)} has extra dimensions: {", ".join(rm_dim)}.\n'
            f'The model requires the following dimensions: {", ".join(model_state.required_dims)}.\n'
            'Having extra data dimensions will increase training time and memory usage,\n'
            'remove them by summing over them: `corpus.sum(dim="extra_dim")`.'
        )


def fit_model( 
    train_corpuses,
    test_corpuses,
    model_state,
    random_state,
    # optimization settings
    empirical_bayes = True,
    begin_prior_updates = 10,
    stop_condition=50,
    # optimization settings
    num_epochs = 2000,
    locus_subsample = None,
    threads = 1,
    kappa = 0.5,
    tau = 1.,
    callback=None,
    eval_every=10,
    verbose=0,
):

    '''for corpus in train_corpuses + test_corpuses:
        _check_dims(corpus, model_state)'''

    logger.info('Preprocessing training corpuses...')
    for corpus in train_corpuses:   
        CS.init_corpusstate(corpus, model_state)

    logger.info('Preprocessing testing corpuses...')
    for test_corpus in test_corpuses:
        CS.init_corpusstate(test_corpus, model_state)

    test_score_fn = partial(
        deviance,
        corpuses=test_corpuses,
        exposures_fn=using_exposures_from(*train_corpuses)
    )

    '''
    Sometimes we'd like to skip evaluating the test data
    to decrease the computational burden.
    '''
    dummy_score_fn = lambda *x, **y : np.nan

    train_scorer = partial(
        perplexity,
        get_n_mutations(train_corpuses)
    )

    lr_schedule = partial(learning_rate_schedule, tau, kappa)

    stop_fn = partial(_should_stop, stop_condition//eval_every)

    if locus_subsample is None:
        logger.info('Using batch variational inference.')
        step_fn = partial(
            VI_step, 
            model_state=model_state,
            corpuses=train_corpuses,
        )

    else:
        logger.info(f'Using stochastic VI with a sampling rate of {locus_subsample:2f}.')
        subsampler = partial(
            locus_slice_generator, 
            random_state, 
            subsample_rate=locus_subsample
        )

        step_fn = partial(
            SVI_step, 
            model_state=model_state,
            corpuses=train_corpuses,
            batch_generator=subsampler,
            subsample_rate=locus_subsample,
        )

    logger.info(f'Training model with {threads} threads.')
    train_scores = []
    test_scores = []

    with ParContext(threads, verbose) as par:
        try:
            progress_bar = trange(
                1, num_epochs+1, 
                desc='Training',
                bar_format='Epoch: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}, {rate_fmt}] Scores{postfix}',
                position=0,
            )
            
            for epoch in progress_bar:

                evaluate_test = not eval_every is None and \
                                (
                                (epoch % eval_every == 0) \
                                or epoch == 1 \
                                or epoch == num_epochs
                                )

                train_score, test_score = step_fn(
                    parallel_context=par,
                    update_prior = epoch >= begin_prior_updates \
                        and empirical_bayes,
                    learning_rate = lr_schedule(epoch),
                    test_score_fn=test_score_fn if evaluate_test else dummy_score_fn,
                )

                if np.isnan(train_score):
                    raise ValueError(
                        'The model has diverged - some parameter is NaN. '
                        'This usually means that the model is under-regularized.\n'
                        'Try increasing one of the regularization parameters:\n'
                        '  - mutation_reg\n'
                        '  - context_reg\n'
                        '  - l2_regularization\n'
                        '  - conditioning_alpha\n'
                    )

                for test_corpus in test_corpuses:
                    CS.update_corpusstate(
                        test_corpus,
                        model_state,
                        from_scratch=False,
                    )

                train_scores.append(train_scorer(train_score))
                
                if evaluate_test:
                    test_scores.append(test_score)

                progress_bar.set_postfix({
                    'Best' : f'{max(test_scores):2f}',
                    'Recent' : ', '.join([f'{x:2f}' for x in test_scores[-5:]]),
                })

                if not callback is None:
                    callback(model_state, train_scores, test_scores)

                if stop_fn(test_scores):
                    logger.info('Early stopping criterion met. The model has converged.')
                    break
        
        except (KeyboardInterrupt, SystemError):
            # sometimes interrupting an optimizer throws a system error ...
            pass

        finally:
            progress_bar.close()

    return (
        model_state,
        train_scores,
        test_scores,
    )