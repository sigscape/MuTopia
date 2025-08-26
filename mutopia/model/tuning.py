"""
Hyperparameter tuning utilities for machine learning models.

This module provides functions for creating and managing Optuna studies for
hyperparameter optimization, including database storage, study management,
and model training with automated pruning.
"""

from typing import Iterable
import os
from glob import glob
from functools import partial
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from mutopia.utils import logger

def _get_study_path(study_name):
    #return os.path.join(
    #    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #    "{study_name}",
    #)
    return os.path.join("studies/", study_name)


def _get_nfs_storage(study_name):
    """
    Create a journal file storage backend for Optuna studies.

    Parameters
    ----------
    study_name : str
        Name of the study to create storage for

    Returns
    -------
    JournalStorage
        Optuna storage backend using journal file
    """
    journal = _get_study_path(study_name)
    storage = JournalStorage(JournalFileBackend(journal))
    return storage


def dashboard(study_name, storage=None):
    """
    Launch an Optuna dashboard for visualizing study progress.

    Parameters
    ----------
    study_name : str
        Name of the study to visualize
    storage : optuna.storages.BaseStorage, optional
        Custom storage backend. If None, uses default NFS storage.

    Raises
    ------
    ImportError
        If optuna_dashboard package is not installed
    """

    try:
        from optuna_dashboard import run_server
    except ImportError:
        raise ImportError(
            "To use the dashboard, you must install the optuna_dashboard package: `pip install optuna_dashboard`"
        )

    if storage is None:
        storage = _get_nfs_storage(study_name)

    run_server(storage)


def summary(study_name, storage=None):
    """
    Get a summary dataframe of all trials in a study.

    Parameters
    ----------
    study_name : str
        Name of the study to summarize
    storage : optuna.storages.BaseStorage, optional
        Custom storage backend. If None, uses default NFS storage.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing trial information including parameters,
        values, and metadata
    """

    if storage is None:
        storage = _get_nfs_storage(study_name)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    return study.trials_dataframe()


def create_study(
    seed=0,
    storage=None,
    save_model=True,
    output_dir=".",
    extensive=0,
    *,
    train,
    test,
    min_components,
    max_components,
    study_name,
    **model_kw,
):
    """
    Create a new Optuna study for hyperparameter optimization.

    Parameters
    ----------
    seed : int, default 0
        Random seed for reproducible sampling
    storage : optuna.storages.BaseStorage, optional
        Custom storage backend. If None, uses default NFS storage.
    save_model : bool, default True
        Whether to save trained models during optimization
    output_dir : str, default "."
        Directory to save models and results
    extensive : int, default 0
        Level of extensive parameter sampling
    train : str or list of str
        Path(s) to training dataset(s)
    test : str or list of str
        Path(s) to test dataset(s)
    min_components : int
        Minimum number of components to try
    max_components : int
        Maximum number of components to try
    study_name : str
        Unique name for the study
    **model_kw
        Additional model parameters to store as study attributes
    """

    if not os.path.exists("studies"):
        os.makedirs("studies")

    if storage is None:
        storage = _get_nfs_storage(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
        sampler=optuna.samplers.RandomSampler(
            seed=seed,
        ),
    )

    if not isinstance(train, Iterable):
        train = (train,)

    if not isinstance(test, Iterable):
        test = (test,)

    study.set_user_attr("min_components", min_components)
    study.set_user_attr("max_components", max_components)
    study.set_user_attr("train", list(map(os.path.abspath, train)))
    study.set_user_attr("test", list(map(os.path.abspath, test)))
    study.set_user_attr("save_model", save_model)
    study.set_user_attr("extensive", extensive)
    study.set_user_attr("output_dir", os.path.abspath(output_dir))

    for key, value in model_kw.items():
        study.set_user_attr(key, value)


def load_study(study_name, storage=None, prune=True):
    """
    Load an existing Optuna study with its configuration.

    Parameters
    ----------
    study_name : str
        Name of the study to load
    storage : optuna.storages.BaseStorage, optional
        Custom storage backend. If None, uses default NFS storage.
    prune : bool, default True
        Whether to enable hyperband pruning for the study

    Returns
    -------
    tuple
        Three-element tuple containing:
        - study : optuna.Study
            The loaded Optuna study object
        - study_attrs : dict
            Study-specific attributes (components, datasets, etc.)
        - model_attrs : dict
            Model-specific attributes and parameters
    """

    if storage is None:
        storage = _get_nfs_storage(study_name)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        pruner=(
            optuna.pruners.HyperbandPruner(
                min_resource=40,
                max_resource=2000,
                reduction_factor=3,
            )
            if prune
            else optuna.pruners.NopPruner()
        ),
    )

    model_attrs = study.user_attrs

    study_attrs = {
        "min_components": model_attrs.pop("min_components"),
        "max_components": model_attrs.pop("max_components"),
        "train": model_attrs.pop("train"),
        "test": model_attrs.pop("test"),
        "save_model": model_attrs.pop("save_model"),
        "output_dir": model_attrs.pop("output_dir"),
        "extensive": model_attrs.pop("extensive", 0),
        "test_chroms": model_attrs.pop("test_chroms", ["chr1"]),
    }

    return (
        study,
        study_attrs,
        model_attrs,
    )


def load_study_data(study, lazy=False):
    """
    Load training and test datasets from a study configuration.

    Parameters
    ----------
    study : optuna.Study
        Optuna study object containing dataset paths in user attributes
    lazy : bool, default False
        Whether to use lazy loading for datasets

    Returns
    -------
    tuple
        Two-element tuple containing:
        - train : list
            List of loaded training datasets
        - test : list
            List of loaded test datasets
    """

    from mutopia.gtensor import lazy_load, eager_load

    train = study.user_attrs["train"]
    test = study.user_attrs["test"]

    load_fn = lazy_load if lazy else eager_load
    train = list(map(load_fn, train))
    test = list(map(load_fn, test))

    return train, test


def _model_report_callback(trial, factor_model, epoch, test_scores):
    """
    Callback function for reporting trial progress to Optuna.

    Reports the latest test score to the trial and checks if the trial
    should be pruned based on intermediate results.

    Parameters
    ----------
    trial : optuna.Trial
        Current trial object
    factor_model : object
        The model being trained (unused in this callback)
    epoch : int
        Current training epoch
    test_scores : list
        List of test scores from training

    Raises
    ------
    optuna.TrialPruned
        If the trial should be pruned based on intermediate results
    """
    from numpy import isfinite

    if isfinite(test_scores[-1]):
        trial.report(test_scores[-1], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


def get_reporting_callback(trial):
    """
    Create a partial callback function for trial reporting.

    Parameters
    ----------
    trial : optuna.Trial
        Trial object to bind to the callback

    Returns
    -------
    functools.partial
        Partial callback function with trial pre-bound
    """
    return partial(_model_report_callback, trial)


def _get_save_model_fn(study):
    """
    Create a model saving function for a specific study.

    Parameters
    ----------
    study : optuna.Study
        Study object containing output directory configuration

    Returns
    -------
    callable
        Function that saves models with trial-specific naming

    Raises
    ------
    PermissionError
        If the output directory is not writable
    """

    study_name = study.study_name
    output_dir = study.user_attrs["output_dir"]

    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_dir}")

    def _save_model(trial, model):

        model_path = os.path.join(
            output_dir,
            f'{study_name.replace("/",".")}_{trial.number}.pkl',
        )

        trial.set_user_attr("model_path", model_path)
        model.save(model_path)

    return _save_model


def _objective(
    trial,
    save_model=True,
    param_sampling_fn=None,
    summary_callback=None,
    *,
    study,
    model,
    train,
    test,
):
    """
    Objective function for Optuna optimization trials.

    This function defines what happens during each trial: parameter sampling,
    model training, evaluation, and optional model saving.

    Parameters
    ----------
    trial : optuna.Trial
        Current trial object for parameter sampling
    save_model : bool, default True
        Whether to save the trained model
    param_sampling_fn : callable, optional
        Custom function for sampling model parameters.
        If None, uses model's default sampling method.
    summary_callback : callable, optional
        Callback function called after trial completion
    study : optuna.Study
        Study object containing configuration
    model : object
        Model instance to train
    train : list
        Training datasets
    test : list
        Test datasets

    Returns
    -------
    float
        Final test score for this trial
    """

    if save_model:
        save_fn = _get_save_model_fn(study)

    if param_sampling_fn is None:
        param_sampling_fn = partial(
            model.sample_params,
            extensive=study.user_attrs["extensive"],
        )

    params = {
        "num_components": trial.suggest_int(
            "num_components",
            study.user_attrs["min_components"],
            study.user_attrs["max_components"],
        )
    }

    params.update(param_sampling_fn(trial))

    logger.info(
        f"Running trial {trial.number} with params:\n\t"
        + "\n\t".join([f"{key}: {value}" for key, value in params.items()])
    )

    callback = partial(_model_report_callback, trial)

    model.set_params(
        **params,
        callback=callback,
        seed=trial.number,
    )

    model.fit(
        train,
        test_datasets=test,
    )

    if save_model:
        save_fn(trial, model)

    if not summary_callback is None:
        summary_callback(trial, model=model, study=study, train=train, test=test)

    return model.test_scores_[-1]


def run_trial(
    save_model=True,
    param_sampling_fn=None,
    summary_callback=None,
    *,
    study,
    model,
    train,
    test,
):
    """
    Run a single optimization trial.

    Parameters
    ----------
    save_model : bool, default True
        Whether to save the trained model
    param_sampling_fn : callable, optional
        Custom function for sampling model parameters
    summary_callback : callable, optional
        Callback function called after trial completion
    study : optuna.Study
        Study object to run trial on
    model : object
        Model instance to train
    train : list
        Training datasets
    test : list
        Test datasets

    Returns
    -------
    object
        Result of study.optimize() call
    """
    objective = partial(
        _objective,
        save_model=save_model,
        param_sampling_fn=param_sampling_fn,
        summary_callback=summary_callback,
        study=study,
        model=model,
        train=train,
        test=test,
    )

    return study.optimize(
        objective,
        n_trials=1,
    )


def _run_trial_cli(
    storage=None,
    lazy=False,
    *,
    study_name,
    **kwargs,
):
    """
    Command-line interface for running a single trial.

    This function loads a study configuration, sets up the model,
    and runs a single optimization trial. Intended for use in
    distributed optimization scenarios.

    Parameters
    ----------
    storage : optuna.storages.BaseStorage, optional
        Custom storage backend. If None, uses default NFS storage.
    lazy : bool, default False
        Whether to use lazy loading for datasets
    study_name : str
        Name of the study to run trial for
    **kwargs
        Additional keyword arguments passed to model configuration
    """

    study, study_attrs, model_kw = load_study(study_name, storage)
    model_kw.update(kwargs)
    model_kw["eval_every"] = 5

    train, test = load_study_data(study, lazy)

    model = train[0].modality().TopographyModel(**model_kw)

    run_trial(
        save_model=study_attrs["save_model"],
        study=study,
        model=model,
        train=train,
        test=test,
    )


def retrain(
    storage=None,
    lazy=False,
    seed=None,
    *,
    study_name,
    trial_number,
    save_name,
    **kwargs,
):
    """
    Retrain a model using parameters from a specific trial.

    This function loads the best parameters from a completed trial
    and retrains the model with those parameters, typically for
    final model deployment or further analysis.

    Parameters
    ----------
    storage : optuna.storages.BaseStorage, optional
        Custom storage backend. If None, uses default NFS storage.
    lazy : bool, default False
        Whether to use lazy loading for datasets
    seed : int, optional
        Random seed for reproducible training. If None, uses trial number.
    study_name : str
        Name of the study containing the trial
    trial_number : int
        Trial number to retrain from
    save_name : str
        Path where to save the retrained model
    **kwargs
        Additional keyword arguments to override model parameters

    Notes
    -----
    This function automatically sets max_features based on convolution_width
    and removes eval_every from parameters to ensure full training.
    """

    from mutopia.gtensor import lazy_train_test_load, eager_train_test_load

    study, study_attrs, model_kw = load_study(study_name, storage)
    model_kw.update(kwargs)
    model_kw.update(study.trials[trial_number].params)
    model_kw["max_features"] = 1 / (model_kw.get("convolution_width", 0) + 1)

    if "eval_every" in model_kw:
        model_kw.pop("eval_every")

    logger.info(
        f"Retraining trial {trial_number} with params:\n\t"
        + "\n\t".join([f"{key}: {value}" for key, value in model_kw.items()])
    )

    train, test = list(
        zip(
            *[
                (lazy_train_test_load if lazy else eager_train_test_load)(
                    corpus, *study_attrs["test_chroms"]
                )
                for corpus in study_attrs["train_corpuses"]
            ]
        )
    )
    example_corpus = train[0]

    model = (
        example_corpus.modality()
        .TopographyModel(
            **model_kw,
            seed=seed,
        )
        .fit(
            *train,
            test_datasets=test,
        )
    )

    model.save(save_name)
