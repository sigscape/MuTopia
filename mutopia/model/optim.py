
from . import corpus_state as CS
from .eval import *
from ..utils import *
from ..gtensor import *
from tqdm import trange


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
    In the previous "offsets" step, the normalizers were updated
    on a subset of the data. Now we need to update the normalizers
    on the full data set.
    '''
    for corpus in corpuses:
        CS.update_normalizers(corpus, model_state.get_normalizers(corpus))

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
        CS.update_corpusstate(
            corpus, 
            model_state, 
            from_scratch=False,
            parallel_context=parallel_context,
        )

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
    locus_subsample,
    batch_subsample,
):

    '''
    Generate the sliced corpuses for the E-step.
    '''
    slices = batch_generator(*corpuses)
    
    args = dict(
        corpuses=slices,
        parallel_context=parallel_context,
    )

    '''
    Get the offsets from the sliced data.
    '''
    offsets = timer_wrapper(model_state.get_exp_offsets_dict)(
        **args,
        norm_update_fn=partial(
            model_state._update_normalizer, 
            learning_rate=learning_rate,
            subsample_rate=locus_subsample
        ),
    )

    '''
    In the previous "offsets" step, the normalizers were updated
    on a subset of the data. Now we need to transfer this information
    to the full data set (this is just a copying step, no computation).
    '''
    for corpus in corpuses:
        CS.update_normalizers(corpus,  model_state.get_normalizers(corpus))
    
    '''
    Okay, now we can calculate the sufficient statistics.
    '''
    sstats, elbo = timer_wrapper(model_state.Estep)(
        **args, 
        learning_rate=learning_rate,
        locus_subsample=locus_subsample or 1.,
        batch_subsample=batch_subsample or 1.,
    )

    '''
    Calculate the bounds here because the normalizers and 
    locals have been updated for some set of model parameters.
    '''
    test_score = timer_wrapper(test_score_fn, 'test_score')(
        model_state, 
        parallel_context=parallel_context
    )

    '''
    Update global model parameters
    '''
    timer_wrapper(model_state.Mstep)(
        offsets=offsets,
        sstats=sstats,
        update_prior=update_prior,
        learning_rate=learning_rate,
        **args,
    )

    '''
    Update the state of the original corpuses,
    - not the slices.
    Note at this point the normalizer and the rate parameters
    are not in sync - this is on purpose, we only want to update
    the normalizers on the subset data during the offset calculation.
    '''
    for corpus in corpuses:
        timer_wrapper(CS.update_corpusstate)(
            corpus, 
            model_state, 
            from_scratch=False,
            parallel_context=parallel_context,
        )

    return elbo, test_score


def learning_rate_schedule(tau, kappa, epoch):
    return (tau + epoch)**(-kappa)


def slice_generator(
        random_state,
        *corpuses,
        locus_subsample=None,
        batch_subsample=None,
    ):

    def _subset_corpus(corpus):

        if not batch_subsample is None:

            sample_names = CS.list_samples(corpus)

            new_samples = list(
                random_state.choice(
                    sample_names,
                    int(len(sample_names)*batch_subsample),
                    replace=False,
                )
            )

            corpus = DifferentSamples(
                corpus, 
                new_samples,
            )


        if not locus_subsample is None:

            corpus = LazySlicer(
                corpus,
                keep_features=False,
                locus = random_state.choice(
                    CS.get_dims(corpus)['locus'],
                    int(locus_subsample*CS.get_dims(corpus)['locus']),
                    replace=False
                )
            )

        return corpus
    
    return tuple(_subset_corpus(corpus) for corpus in corpuses)


def _should_stop(stop_condition, scores):
    return len(scores) > stop_condition \
        and scores.index(max(scores)) < (len(scores) - stop_condition)


def fit_model( 
    train_corpuses,
    test_corpuses,
    model_state,
    random_state,
    # optimization settings
    empirical_bayes = True,
    begin_prior_updates = 50,
    stop_condition=50,
    # optimization settings
    num_epochs = 2000,
    locus_subsample = None,
    batch_subsample = None,
    threads = 1,
    kappa = 0.5,
    tau = 1.,
    callback=None,
    eval_every=10,
    verbose=0,
    time_limit=None,
):
    start_time = time.time()
    ##
    # If the data is sparse, we should make sure it's in the right format.
    # GCSX with locus and sample dimensions compressed.
    ##
    logger.info('Validating corpuses...')
    for corpus in train_corpuses + test_corpuses:
        check_dims(corpus, model_state)
        check_corpus(corpus)

    # Ensure all train corpuses have different names
    corpus_names = [CS.get_name(corpus) for corpus in train_corpuses]
    if len(corpus_names) != len(set(corpus_names)):
        raise ValueError("All train corpuses must have different names.")
    
    corpus_names = [CS.get_name(corpus) for corpus in test_corpuses]
    if len(corpus_names) != len(set(corpus_names)):
        raise ValueError("All test corpuses must have different names.")
    
    num_training_samples = sum(
        len(CS.list_samples(corpus)) 
        for corpus in train_corpuses
    )
    logger.info(f'Found n={num_training_samples} training samples across {len(train_corpuses)} corpuses.')
    
    check_feature_consistency(*train_corpuses, *test_corpuses)

    logger.info('Preprocessing training corpuses...')
    train_corpuses = [CS.init_corpusstate(corpus, model_state) for corpus in train_corpuses]

    logger.info('Preprocessing testing corpuses...')
    test_corpuses = [CS.init_corpusstate(corpus, model_state) for corpus in test_corpuses]

    test_score_fn = partial(
        deviance_locus,
        corpuses=test_corpuses,
        exposures_fn=CS.using_exposures_from(*train_corpuses)
    )
    
    '''
    Sometimes we'd like to skip evaluating the test data
    to decrease the computational burden.
    '''
    dummy_score_fn = lambda *x, **y : np.nan
    train_scorer = lambda x : -x # we don't really care about train score - test score is what matters

    lr_schedule = partial(learning_rate_schedule, tau, kappa)

    stop_fn = partial(_should_stop, stop_condition//eval_every)

    if locus_subsample is None and batch_subsample is None:
        logger.warning('Using batch variational inference.')
        step_fn = partial(
            VI_step, 
            model_state=model_state,
            corpuses=train_corpuses,
        )
    else:

        subsample_rates = dict(
            locus_subsample=locus_subsample,
            batch_subsample=batch_subsample,
        )

        logger.info(f'Using SVI.')
        subsampler = partial(slice_generator, random_state, **subsample_rates)

        step_fn = partial(
            SVI_step, 
            model_state=model_state,
            corpuses=train_corpuses,
            batch_generator=subsampler,
            **subsample_rates,
        )

    logger.info(f'Training model with {threads} threads.')
    logger.info('The first few epochs take longer as things get warmed up -\n\texpect the time per epoch to decrease about 4-fold.')
    logger.info(f'Model will stop training if no improvement in the last {stop_condition} epochs.')
    if not time_limit is None:
        logger.info(f'Training will stop after {time_limit} minutes.')

    train_scores = []
    test_scores = []
    prior_has_been_updated = False

    with ParContext(threads, verbose) as par:
        try:
            progress_bar = trange(
                1, num_epochs+1, 
                desc='Training',
                bar_format='Epoch: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}, {rate_fmt}] Scores{postfix}',
                position=0,
            )
            
            for epoch in progress_bar:

                evaluate_test = (
                    not eval_every is None and 
                    (
                        (epoch % eval_every == 0) \
                        or epoch == 1 \
                        or epoch == num_epochs
                    )
                )
                
                update_prior = (
                    epoch >= begin_prior_updates
                    and empirical_bayes
                    and not any(CS.is_marginal_corpus(corpus) for corpus in train_corpuses)
                )

                if update_prior and not prior_has_been_updated:
                    logger.info('Beginning to update priors.')
                    prior_has_been_updated = True

                train_score, test_score = timer_wrapper(step_fn, 'train_step')(
                    parallel_context=par,
                    update_prior = update_prior,
                    learning_rate = lr_schedule(epoch),
                    test_score_fn=test_score_fn if evaluate_test else dummy_score_fn,
                )

                if np.isnan(train_score):
                    raise ValueError(
                        'The model has diverged - some parameter is NaN. '
                        'This usually means that the model is under-regularized.\n'
                        'Try increasing one of the regularization parameters:\n'
                        '  - conditioning_alpha\n'
                        '  - mutation_reg\n'
                        '  - context_reg\n'
                        '  - l2_regularization\n'
                    )

                for test_corpus in test_corpuses:
                    CS.update_corpusstate(
                        test_corpus,
                        model_state,
                        from_scratch=False,
                        parallel_context=par,
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

                if not time_limit is None and (time.time() - start_time)/60 > time_limit:
                    logger.info('Time limit reached. Stopping training.')
                    break
        
        except (KeyboardInterrupt, SystemError, SystemExit):
            # sometimes interrupting an optimizer throws a system error ...
            pass

        finally:
            progress_bar.close()

    if num_epochs > 0:
        logger.info('Finalizing models ...')

        for model in model_state.nonlocals.values():
            model.post_fit(train_corpuses)

        if not empirical_bayes:
            logger.info('Updating priors ...')
            model_state.optim_prior(train_corpuses)

    return (
        model_state,
        train_scores,
        test_scores,
    )