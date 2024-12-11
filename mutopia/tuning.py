import optuna
import os
from functools import partial
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(' Mutopia-tuning ')
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from .corpus import load_dataset

def _get_nfs_storage(study_name):
    
    journal = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    f'.journal.{study_name}.db'
                )

    storage = JournalStorage(JournalFileBackend(journal))
    return storage


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


def create_study(
    train_corpuses,
    test_corpuses,
    seed=0,
    storage=None,
    save_model=False,
    output_dir=None,
    *,
    min_components, 
    max_components,
    study_name,
    **model_kw,
):
    
    if storage is None:
        storage = _get_nfs_storage(study_name)

    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        direction='maximize',
        load_if_exists = True,
        pruner = optuna.pruners.NopPruner(),
        sampler = optuna.samplers.RandomSampler(
            seed = seed,
        ),
    )

    study.set_user_attr('min_components', min_components)
    study.set_user_attr('max_components', max_components)
    study.set_user_attr('train_corpuses', train_corpuses)
    study.set_user_attr('test_corpuses', test_corpuses)
    study.set_user_attr('save_model', save_model)
    study.set_user_attr('output_dir', output_dir)

    for key, value in model_kw.items():
        study.set_user_attr(key, value)



def load_study(study_name, storage = None, prune=True):
    
    if storage is None:
        storage = _get_nfs_storage(study_name)    

    study = optuna.load_study(
        study_name=study_name, 
        storage=storage,
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=40,
            max_resource=2000,
            reduction_factor=3,
        ) if prune else optuna.pruners.NopPruner(),
    )

    model_attrs = study.user_attrs
    
    study_attrs = {
        'min_components' : model_attrs.pop('min_components'),
        'max_components' : model_attrs.pop('max_components'),
        'train_corpuses' : model_attrs.pop('train_corpuses'),
        'test_corpuses' : model_attrs.pop('test_corpuses'),
        'save_model' : model_attrs.pop('save_model'),
        'output_dir' : model_attrs.pop('output_dir'),
    }

    return (
        study, 
        study_attrs,
        model_attrs,
    )


def _model_report_callback(
    trial,
    model_state, 
    train_scores, 
    test_scores
):
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
            f'{study_name.replace("/",".")}_{trial.number}.pkl'
        )
        
        trial.set_user_attr('model_path',  model_path)
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
    
    (model, _, test_scores) = model_fn(
        **params, 
        callback=callback, 
        seed=trial.number
    )

    model_save_fn(trial, model)
    
    return test_scores[-1]


def _sample_params(study, extra_param_fn, trial):

    params = {
        'num_components' : trial.suggest_int(
            'num_components', 
            study.user_attrs['min_components'], 
            study.user_attrs['max_components']
        )
    }

    params.update(extra_param_fn(study, trial))

    logger.info(
        f"Running trial {trial.number} with params:\n\t" \
        + "\n\t".join(
            [f"{key}: {value}" for key, value in params.items()]
        )
    )

    return params
    

def run_trial(
    storage = None,
    threads = 1,
    *,
    study_name,
):

    study, study_attrs, model_kw = load_study(study_name, storage)

    train_corpuses = tuple(map(load_dataset, study_attrs['train_corpuses']))
    test_corpuses = tuple(map(load_dataset, study_attrs['test_corpuses']))
    
    example_corpus = train_corpuses[0]

    model_fn = partial(
        example_corpus.modality().make_model,
        train_corpuses,
        test_corpuses,
        threads = threads,
        **model_kw,
    )

    if not study_attrs['save_model']:
        model_save_fn = lambda *args: None
    else:
        model_save_fn = _get_save_model_fn(
            study_name,
            study_attrs['output_dir'],
        )

    param_sampling_fn = partial(
        _sample_params,
        study,
        example_corpus.modality().sample_params,
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
