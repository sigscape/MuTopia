import optuna
import os
from glob import glob
from functools import partial
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from numpy import isfinite
from .gtensor.interfaces import *
from .utils import logger


def sample_params(study, trial, extensive=0):

    params = {}

    if extensive > 0:
        params = {
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-5, 1000.0, log=True
            ),
            "tree_learning_rate": trial.suggest_float("tree_learning_rate", 0.025, 0.2),
            "init_variance_theta": trial.suggest_float(
                "init_variance_theta", 0.025, 0.1
            ),
            "empirical_bayes": trial.suggest_categorical(
                "empirical_bayes", [True, False]
            ),
        }

    if extensive > 1:
        params["convolution_width"] = trial.suggest_categorical(
            "convolution_width", [0, 1, 2]
        )
        params["max_features"] = 1 / (params["convolution_width"] + 1)

    if extensive > 2:
        params.update(
            {
                "context_reg": trial.suggest_float("context_reg", 1e-5, 5e-2, log=True),
                "context_conditioning": trial.suggest_float(
                    "context_conditioning", 1e-9, 1e-2, log=True
                ),
                "init_variance_context": trial.suggest_float(
                    "init_variance_context", 0.025, 0.15
                ),
            }
        )

    if extensive > 3:
        params.update(
            {
                "batch_subsample": trial.suggest_categorical(
                    "batch_subsample",
                    [
                        None,
                        0.0625,
                        0.125,
                        0.25,
                    ],
                ),
                "locus_subsample": trial.suggest_categorical(
                    "locus_subsample",
                    [
                        None,
                        0.0625,
                        0.125,
                        0.25,
                    ],
                ),
            }
        )

    if extensive > 4:
        params.update(
            {
                "conditioning_alpha": trial.suggest_float(
                    "conditioning_alpha", 1e-10, 1e-7, log=True
                ),
            }
        )

    return params


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


def _model_report_callback(trial, model_state, train_scores, test_scores):

    if isfinite(test_scores[-1]):
        trial.report(test_scores[-1], len(train_scores))
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


def _objective(
    trial,
    model_fn,
    param_sampling_fn,
    model_save_fn,
):
    params = param_sampling_fn(trial)

    callback = partial(_model_report_callback, trial)

    (model, _, test_scores) = model_fn(**params, callback=callback, seed=trial.number)

    model_save_fn(trial, model)

    return test_scores[-1]


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

    if "eval_every" in model_kw:
        model_kw.pop("eval_every")

    model_fn = partial(
        example_corpus.modality().make_model,
        train,
        test,
        eval_every=5,
        **model_kw,
    )

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
        partial(
            example_corpus.modality().sample_params, extensive=study_attrs["extensive"]
        ),
    )

    obj_fn = partial(
        _objective,
        model_fn=model_fn,
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
                    corpus, study_attrs["test_chroms"]
                )
                for corpus in study_attrs["train_corpuses"]
            ]
        )
    )
    example_corpus = train[0]

    model, *_ = example_corpus.modality().make_model(
        train,
        test,
        eval_every=5,
        seed=seed or trial_number,
        **model_kw,
    )

    model.save(save_name)
