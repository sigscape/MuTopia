from tqdm import trange
from functools import partial
from math import isnan
from ..utils import (
    logger,
    str_wrapped_list,
    timer_wrapper,
    ParContext
)
import time
from ..gtensor import (
    DifferentSamples,
    LazySlicer,
)
from ..gtensor.validation import (
    check_dims,
    check_corpus,
    check_feature_consistency,
)
from .factor_model import FactorModel
from .latent_var_models.base import LocalsModel
from .gtensor_interface import GtensorInterface


def VI_step(
    factor_model: FactorModel,
    locals_model: LocalsModel,
    datasets,
    GT: GtensorInterface,
    update_prior=True,
    *,
    test_score_fn,
    par_context,
    **kw, # just soak up the kwargs for compatibility
):

    args = dict(
        datasets=datasets,
        par_context=par_context,
    )

    offsets, normalizers = factor_model.get_exp_offsets_dict(**args)

    """
    In the previous "offsets" step, new normalizers were calculated. 
    Now we need to transer the normalizers to the full data set.
    """
    factor_model.update_normalizers(datasets, normalizers)

    """
    The normalizers are updated in the previous function
    (because the mutation rates are used to get the offsets, 
    so it's more efficient to use them to update the normalizers there).

    Therefore, this is the best time to evaluate the loss functions.
    The elbo is calculated during the E-step because it's convenient.
    """
    stats = locals_model.Estep(**args, factor_model=factor_model)
    """
    We're agnostic to the form of the test set evaluation function,
    but it should be a function of the model state that returns the
    test set score.
    """
    test_score = test_score_fn(par_context=par_context)

    factor_model.Mstep(
        offsets=offsets,
        sstats=stats,
        **args,
    )

    if update_prior:
        locals_model.Mstep(datasets)

    for dataset in datasets:
        GT.update_state(
            dataset,
            factor_model,
            from_scratch=False,
            par_context=par_context,
        )

    return 0.0, test_score


def SVI_step(
    factor_model: FactorModel,
    locals_model: LocalsModel,
    datasets,
    GT: GtensorInterface,
    update_prior=True,
    full_normalizers=False,
    *,
    test_score_fn,
    par_context,
    batch_generator,
    learning_rate,
    locus_subsample,
    batch_subsample,
):
    """
    Generate the sliced datasets for the E-step.
    """
    slices = batch_generator(*datasets)

    args = dict(
        datasets=slices,
        par_context=par_context,
    )

    """
    Get the offsets from the sliced data.
    """
    offsets, normalizers = timer_wrapper(factor_model.get_exp_offsets_dict)(**args)

    if full_normalizers:
        '''
        If the locus subsample is very small, we need to update the normalizers
        on the full datasets to reduce variance.
        '''
        _, normalizers = factor_model.get_exp_offsets_dict(
            datasets=datasets,
            par_context=par_context,
        )

        factor_model.update_normalizers(datasets, normalizers)
    else:

        factor_model.update_normalizers(
            slices,
            normalizers,
            learning_rate=learning_rate,
            subsample_rate=locus_subsample or 1.0,
        )

    """
    Okay, now we can calculate the sufficient statistics.
    """
    sstats = timer_wrapper(locals_model.Estep)(
        **args,
        factor_model=factor_model,
        learning_rate=learning_rate,
        locus_subsample=locus_subsample or 1.0,
        batch_subsample=batch_subsample or 1.0,
    )

    """
    Calculate the bounds here because the normalizers and 
    locals have been updated for some set of model parameters.
    """
    test_score = timer_wrapper(test_score_fn, "test_score")(par_context=par_context)

    """
    Update global model parameters
    """
    timer_wrapper(factor_model.Mstep)(
        offsets=offsets,
        sstats=sstats,
        learning_rate=learning_rate,
        **args,
    )

    if update_prior:
        locals_model.Mstep(datasets)

    """
    Update the state of the original datasets,
    - not the slices.
    Note at this point the normalizer and the rate parameters
    are not in sync - this is on purpose, we only want to update
    the normalizers on the subset data during the offset calculation.
    """
    for dataset in datasets:
        timer_wrapper(GT.update_state)(
            dataset,
            factor_model,
            from_scratch=False,
            par_context=par_context,
        )

    return 0.0, test_score


def learning_rate_schedule(tau, kappa, epoch):
    return (tau + epoch) ** (-kappa)


def slice_generator(
    GT: GtensorInterface,
    random_state,
    *datasets,
    locus_subsample=None,
    batch_subsample=None,
):

    def _subset_corpus(dataset):

        if not batch_subsample is None:

            sample_names = GT.list_samples(dataset)

            new_samples = list(
                random_state.choice(
                    sample_names,
                    int(len(sample_names) * batch_subsample),
                    replace=False,
                )
            )

            dataset = DifferentSamples(
                dataset,
                new_samples,
            )

        if not locus_subsample is None:

            dataset = LazySlicer(
                dataset,
                keep_features=False,
                locus=random_state.choice(
                    GT.get_dims(dataset)["locus"],
                    int(locus_subsample * GT.get_dims(dataset)["locus"]),
                    replace=False,
                ),
            )

        return dataset

    return tuple(_subset_corpus(dataset) for dataset in datasets)


def _should_stop(stop_condition, scores):
    return len(scores) > stop_condition and scores.index(max(scores)) < (
        len(scores) - stop_condition
    )


def fit_model(
    GT: GtensorInterface,  # gtensor interface
    train_datasets,
    test_datasets,
    random_state,
    factor_model: FactorModel,
    locals_model: LocalsModel,
    # optimization settings
    empirical_bayes=True,
    begin_prior_updates=50,
    stop_condition=50,
    # optimization settings
    num_epochs=2000,
    locus_subsample=None,
    batch_subsample=None,
    threads=1,
    kappa=0.5,
    tau=1.0,
    callback=None,
    eval_every=10,
    verbose=0,
    time_limit=None,
    full_normalizers=False,
    **kw,
) -> tuple[FactorModel, LocalsModel, list]:

    start_time = time.time()
    models = (factor_model, locals_model)
    ##
    # If the data is sparse, we should make sure it's in the right format.
    # GCSX with locus and sample dimensions compressed.
    ##
    logger.info("Validating datasets...")
    for dataset in train_datasets + test_datasets:
        check_dims(dataset, factor_model)
        check_corpus(dataset)

    sources = list(set([source for ds in train_datasets + test_datasets for source in GT.list_sources(ds)]))
    if len(sources) > 1:
        logger.info(f"Found sources: {str_wrapped_list(sources)}")

    # Ensure all train datasets have different names
    corpus_names = [name for name, _ in GT.expand_datasets(*train_datasets)]
    if len(corpus_names) != len(set(corpus_names)):
        raise ValueError("All train datasets must have different names.")

    num_training_samples = sum(
        len(GT.list_samples(dataset)) for dataset in train_datasets
    )
    logger.info(
        f"Found n={num_training_samples} training samples across {len(train_datasets)} datasets."
    )

    check_feature_consistency(*train_datasets, *test_datasets)

    logger.info("Preprocessing training datasets...")
    train_datasets = [
        GT.init_state(GT.prepare_data(dataset), *models) for dataset in train_datasets
    ]

    logger.info("Preprocessing testing datasets...")
    test_datasets = [
        GT.init_state(GT.prepare_data(dataset), *models) for dataset in test_datasets
    ]

    test_score_fn = partial(
        locals_model.score,
        factor_model,
        test_datasets,
        exposures_fn=GT.using_exposures_from(*train_datasets),
    )

    """
    Sometimes we'd like to skip evaluating the test data
    to decrease the computational burden.
    """
    dummy_score_fn = lambda *x, **y: float("nan")

    lr_schedule = partial(learning_rate_schedule, tau, kappa)

    stop_fn = partial(_should_stop, stop_condition // eval_every)

    if full_normalizers:
        logger.warning(
            "Calculating full factor normalizers for updates - this will increase the memory usage, but will reduce variance."
        )

    if locus_subsample is None and batch_subsample is None:
        logger.warning("Using batch variational inference.")
        step_fn = partial(
            VI_step,
            *models,
            train_datasets,
            GT,
            full_normalizers=full_normalizers,
        )
    else:

        subsample_rates = dict(
            locus_subsample=locus_subsample,
            batch_subsample=batch_subsample,
        )

        logger.info(f"Using SVI.")
        subsampler = partial(slice_generator, GT, random_state, **subsample_rates)

        step_fn = partial(
            SVI_step,
            *models,
            train_datasets,
            GT,
            batch_generator=subsampler,
            full_normalizers=full_normalizers,
            **subsample_rates,
        )

    logger.info(f"Training model with {threads} threads.")
    logger.info(
        "The first few epochs take longer as things get warmed up -\n\texpect the time per epoch to decrease about 4-fold."
    )
    logger.info(
        f"Model will stop training if no improvement in the last {stop_condition} epochs."
    )
    logger.info(f"Training usually coverges much sooner than {num_epochs} epochs.")
    if not time_limit is None:
        logger.info(f"Training will stop after {time_limit} minutes.")

    test_scores = []
    prior_has_been_updated = False

    with ParContext(threads, verbose) as par:

        #factor_model.init_normalizers(train_datasets, par_context=par)
        
        try:
            progress_bar = trange(
                1,
                num_epochs + 1,
                desc="Training",
                bar_format="Epoch: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}, {rate_fmt}] Scores{postfix}",
                position=0,
            )

            for epoch in progress_bar:

                evaluate_test = not eval_every is None and (
                    (epoch % eval_every == 0) or epoch == 1 or epoch == num_epochs
                )

                update_prior = (
                    epoch >= begin_prior_updates
                    and empirical_bayes
                    and not any(
                        GT.is_marginal_corpus(dataset) for dataset in train_datasets
                    )
                )

                if update_prior and not prior_has_been_updated:
                    logger.info("Beginning to update priors.")
                    prior_has_been_updated = True

                train_score, test_score = timer_wrapper(step_fn, "train_step")(
                    par_context=par,
                    update_prior=update_prior,
                    learning_rate=lr_schedule(epoch),
                    test_score_fn=test_score_fn if evaluate_test else dummy_score_fn,
                )

                if isnan(train_score):
                    raise ValueError(
                        "The model has diverged - some parameter is NaN. "
                        "This usually means that the model is under-regularized, or it suffered a bad initial condition.\n"
                        "First, try re-initializing the model with a different random seed.\n"
                        "If that doesn't work, try increasing one of these parameters (in order):\n"
                        "  - locus_subsample\n"
                        "  - batch_subsample (unless already at 1 or None)\n"
                        "  - conditioning_alpha\n"
                        "  - mutation_reg\n"
                        "  - context_reg\n"
                        "  - l2_regularization\n"
                    )

                for test_corpus in test_datasets:
                    GT.update_state(
                        test_corpus,
                        factor_model,
                        from_scratch=False,
                        par_context=par,
                    )

                if evaluate_test:
                    test_scores.append(test_score)

                progress_bar.set_postfix(
                    {
                        "Best": f"{max(test_scores):2f}",
                        "Recent": ", ".join([f"{x:2f}" for x in test_scores[-5:]]),
                    }
                )

                if not callback is None:
                    callback(factor_model, epoch, test_scores)

                if stop_fn(test_scores):
                    logger.info(
                        "Early stopping criterion met. The model has converged."
                    )
                    break

                if (
                    not time_limit is None
                    and (time.time() - start_time) / 60 > time_limit
                ):
                    logger.info("Time limit reached. Stopping training.")
                    break

        except (KeyboardInterrupt, SystemError, SystemExit):
            # sometimes interrupting an optimizer throws a system error ...
            pass

        finally:
            progress_bar.close()

    if num_epochs > 0:
        logger.info("Finalizing models ...")

        for model in factor_model.models.values():
            model.post_fit(GT.to_datasets(*train_datasets)[0])

        if not empirical_bayes:
            logger.info("Updating priors ...")
            locals_model.Mstep(train_datasets)

    return (
        factor_model,
        locals_model,
        test_scores,
    )
