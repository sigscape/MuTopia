import optuna
import os
from functools import partial
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(' Mutopia-tuning ')
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from .corpus.interfaces import lazy_load

def sample_params(study, trial, extensive=0):

    params = {
        'l2_regularization' : trial.suggest_float('l2_regularization', 1e-5, 1000., log=True),
        'tree_learning_rate' : trial.suggest_float('tree_learning_rate', 0.025, 0.2),
        'init_variance' : (0.1, 0.1, trial.suggest_float('init_variance_theta', 0.01, 0.1)),
    }

    if extensive>0:
        params.update({
            'context_reg' : trial.suggest_float('context_reg', 1e-5, 5e-2, log=True),
            'max_features' : trial.suggest_categorical('max_features', [0.25, 0.33, 0.5, 0.75, 1.]),
        })

    if extensive>1:
        params['init_variance'] = (
            trial.suggest_float('init_variance_mutation', 0.01, 0.1),
            trial.suggest_float('init_variance_context', 0.01, 0.1),
            params['init_variance'][2],
        )

        params.update({
            'context_conditioning' : trial.suggest_float('context_conditioning', 1e-9, 1e-2, log=True),
            'context_encoder' : trial.suggest_categorical('context_encoder', ['diagonal', 'kmer']),
            'kmer_reg' : trial.suggest_float('kmer_reg', 1e-4, 5e-2, log=True),
        })

    if extensive>2:
        params.update({
            'batch_subsample' : trial.suggest_categorical('batch_subsample', [None, 0.0625, 0.125, 0.25, 0.5]),
            'locus_subsample' : trial.suggest_categorical('locus_subsample', [None, 0.0625, 0.125, 0.25, 0.5]),
        })

    if extensive>3:
        params.update({
            'conditioning_alpha' : trial.suggest_float('conditioning_alpha', 1e-10, 1e-7, log=True),
        })
    
    return params



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
    extensive=0,
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
    study.set_user_attr('extensive', extensive)
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
        'extensive' : model_attrs.pop('extensive', 0),
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

    train_corpuses = tuple(map(lazy_load, study_attrs['train_corpuses']))
    test_corpuses = tuple(map(lazy_load, study_attrs['test_corpuses']))
    
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
        partial(example_corpus.modality().sample_params, extensive=study_attrs['extensive']),
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
