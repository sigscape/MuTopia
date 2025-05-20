import optuna
import os
from glob import glob
from functools import partial
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from numpy import isfinite
from .gtensor import *
from .utils import logger


def _get_nfs_storage(study_name):

    journal = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f".journal.{study_name}.db",
    )

    storage = JournalStorage(JournalFileBackend(journal))
    return storage


def list_studies():

    glob_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".journal.{study_name}.db",
    )

    journal_files = glob(glob_script.format(study_name="*"))

    return [
        os.path.basename(journal).removeprefix(".journal.").removesuffix(".db")
        for journal in journal_files
    ]


def dashboard(study_name, storage=None):

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

    if storage is None:
        storage = _get_nfs_storage(study_name)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    return study.trials_dataframe()


def create_study(
    train_corpuses,
    seed=0,
    storage=None,
    save_model=False,
    output_dir=None,
    extensive=0,
    test_chroms=["chr1"],
    *,
    min_components,
    max_components,
    study_name,
    **model_kw,
):

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

    study.set_user_attr("min_components", min_components)
    study.set_user_attr("max_components", max_components)
    study.set_user_attr("train_corpuses", train_corpuses)
    study.set_user_attr("save_model", save_model)
    study.set_user_attr("extensive", extensive)
    study.set_user_attr("test_corpuses", test_chroms)
    study.set_user_attr("output_dir", output_dir)

    for key, value in model_kw.items():
        study.set_user_attr(key, value)


def load_study(study_name, storage=None, prune=True):

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
        "train_corpuses": model_attrs.pop("train_corpuses"),
        "test_corpuses": model_attrs.pop("test_corpuses"),
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


def _model_report_callback(trial, factor_model, epoch, test_scores):

    if isfinite(test_scores[-1]):
        trial.report(test_scores[-1], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


def _get_save_model_fn(
    study_name,
    output_dir,
):
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_dir}")

    def _save_model(trial, model):

        model_path = os.path.join(
            os.path.abspath(output_dir),
            f'{study_name.replace("/",".")}_{trial.number}.pkl',
        )

        trial.set_user_attr("model_path", model_path)
        model.save(model_path)

    return _save_model


def _sample_params(study, extra_param_fn, trial):

    params = {
        "num_components": trial.suggest_int(
            "num_components",
            study.user_attrs["min_components"],
            study.user_attrs["max_components"],
        )
    }

    params.update(extra_param_fn(study, trial))

    logger.info(
        f"Running trial {trial.number} with params:\n\t"
        + "\n\t".join([f"{key}: {value}" for key, value in params.items()])
    )

    return params


def _objective(
    trial,
    *,
    model_kw,
    train,
    test,
    param_sampling_fn,
    model_save_fn,
):
    params = param_sampling_fn(trial)
    model_kw.update(params)
    callback = partial(_model_report_callback, trial)

    model = (
        train[0]
        .modality()
        .TopographyModel(**model_kw, callback=callback, seed=trial.number)
        .fit(
            *train,
            test_datasets=test,
        )
    )

    model_save_fn(trial, model)

    return model.test_scores_[-1]


def run_trial(
    storage=None,
    lazy=False,
    *,
    study_name,
    **kwargs,
):

    study, study_attrs, model_kw = load_study(study_name, storage)
    model_kw.update(kwargs)

    train, test = list(
        zip(
            *[
                (lazy_train_test_load if lazy else eager_train_test_load)(
                    corpus, study_attrs["test_chroms"]
                )
                for corpus in study_attrs["train_corpuses"]
            ]
        )
    )

    example_corpus = train[0]
    model_cls = example_corpus.modality().TopographyModel

    if "eval_every" in model_kw:
        model_kw.pop("eval_every")

    if not study_attrs["save_model"]:
        model_save_fn = lambda *args: None
    else:
        model_save_fn = _get_save_model_fn(
            study_name,
            study_attrs["output_dir"],
        )

    param_sampling_fn = partial(
        _sample_params,
        study,
        partial(model_cls.sample_params, extensive=study_attrs["extensive"]),
    )

    obj_fn = partial(
        _objective,
        model_kw=model_kw,
        train=train,
        test=test,
        param_sampling_fn=param_sampling_fn,
        model_save_fn=model_save_fn,
    )

    study.optimize(
        obj_fn,
        n_trials=1,
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
