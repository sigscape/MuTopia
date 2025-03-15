
import mutopia as mu
from mutopia.corpus.interfaces import *
from mutopia.model.corpus_state import CorpusState as CS
import mutopia.corpus.disk_interface as disk
import netCDF4 as nc
from ..utils import logger

import click
from tabulate import tabulate
from typing import *
import numpy as np
import os
from tqdm import tqdm


def _setup_bootstrap(train_corpuses, test_corpuses, seed):
    
    logger.warning('Bootstrapping training and testing corpuses ...')
    test_corpus_map = {CS.get_name(corpus) : corpus for corpus in test_corpuses}

    train_corpuses = tuple([
        BootstrapCorpus(corpus, np.random.RandomState(seed)) 
        for corpus in train_corpuses
    ])

    test_corpuses = tuple([
        DifferentSamples(test_corpus_map[CS.get_name(corpus)], corpus.list_samples())
        for corpus in train_corpuses
    ])

    return train_corpuses, test_corpuses


@click.group("Model training")
def model():
    pass


@model.command("train")
@click.argument(
    'train_corpuses',
    type=click.Path(exists=True),
    nargs=-1,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True),
    required=True,
    help="Path to save the trained model to.",
)
@click.option(
    '-k',
    '--num-components',
    type=click.IntRange(1, 1000),
    required=True,
)
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option(
    "-init",
    "--init-components",
    type=str,
    default=[],
    multiple=True,
    help="List of components to initialize.",
)
@click.option(
    "-fix",
    "--fix-components",
    type=str,
    default=[],
    multiple=True,
    help="List of components to fix.",
)
@click.option(
    "-creg",
    "--context-reg",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="Regularization for context model",
)
@click.option(
    "-alpha",
    "--conditioning-alpha",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="Stabilizing parameter - increase if you're getting NaNs",
)
@click.option(
    "-pi",
    "--pi-prior",
    type=click.FloatRange(0.0, 1000.0),
    default=None,
    help="Prior for local model, if --empirical-bayes is passed, then this prior is updated using empirical bayes.",
)
@click.option(
    "-model",
    "--locus-model-type",
    type=click.Choice(["gbt", "linear"]),
    default="gbt",
    help="Type of model to use for locus model",
)
@click.option(
    "-lr",
    "--tree-learning-rate",
    type=click.FloatRange(0.0, 100.0),
    default=None,
    help="Learning rate for tree models",
)
@click.option(
    "--max-features",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Maximum number of features to use per tree",
)
@click.option(
    '--convolution-width',
    '-cw',
    type=click.IntRange(0, 5),
    default=None,
    help='Width of the convolutional kernel',
)
@click.option(
    "--use-groups/--no-use-groups",
    type=bool,
    default=False,
    is_flag=True,
    help="Use groups in locus model",
)
@click.option(
    "--add-corpus-intercepts",
    type=bool,
    default=False,
    is_flag=True,
    help="Model interactions for each corpus",
)
@click.option(
    "-l2",
    "--l2-regularization",
    type=click.FloatRange(0.0, 100000.0),
    default=None,
    help="L2 regularization for locus model",
)
@click.option(
    "--empirical-bayes/--no-empirical-bayes",
    type=bool,
    default=True,
    is_flag=True,
    help="Use empirical bayes for prior updates",
)
@click.option(
    "--begin-prior-updates",
    type=click.IntRange(0, 100000),
    default=None,
    help="Number of epochs to wait before updating priors",
)
@click.option(
    "-stop",
    "--stop-condition",
    type=click.IntRange(0, 100000),
    default=50,
    help="Number of epochs elapsed without improvement before stopping training",
)
@click.option(
    "-epochs",
    "--num-epochs",
    type=click.IntRange(0, 100000),
    default=2000,
    help="Number of epochs to train.",
)
@click.option(
    "-lsub",
    "--locus-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Subsample rate for locus model",
)
@click.option(
    "-bsub",
    "--batch-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Subsample rate for locus model",
)
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of threads to use",
)
@click.option(
    "--kappa",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='"Forgetting" parameter for stochastic variational inference',
)
@click.option(
    "--tau",
    type=click.IntRange(1, 1000),
    default=None,
    help='"Offset" parameter for stochastic variational inference',
)
@click.option(
    '--init-variance-theta',
    '-ivt',
    type=click.FloatRange(0.0, 1000.0),
    default=None,
    help='Initial variance for theta',
)
@click.option(
    "-eval",
    "--eval-every",
    type=click.IntRange(1, 1000),
    default=5,
    help="Evaluate every N epochs",
)
@click.option(
    "--time-profile/--no-time-profile",
    type=bool,
    default=False,
    is_flag=True,
    help="Profile the training time",
)
@click.option(
    "--lazy/--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Lazy load the underlying data to reduce memory requirements.\n",
)
@click.option(
    "--time-limit",
    '-t',
    type=int,
    default=None,
    help="Time limit for training, in seconds",
)
@click.option(
    "--bootstrap",
    '-b',
    type=click.IntRange(0, np.iinfo(np.int32).max),
    default=None,
    help="Bootstrap the training and testing corpuses",
)
@click.option(
    "--test-chroms",
    '-test',
    multiple=True,
    type=str,
    default=['chr1'],
    help="Chromosomes to use for testing. If not specified, all chromosomes are used.",
)
def train(
    *,
    output,
    train_corpuses: List[str],
    #test_corpuses: List[str],
    time_profile: bool = False,
    lazy: bool = False,
    bootstrap : Union[int, None] = None,
    test_chroms : List[str] = ['chr1'],
    **model_kw,
):  
    if not len(train_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "train-corpuses",
            "At least one training corpus is required",
        )
    
    model_kw = {k: v for k, v in model_kw.items() if v is not None}
    click.echo(
        f'Training model with parameters: ', 
        file=click.get_text_stream('stderr')
    )
    click.echo(
        tabulate(
            model_kw.items(),
            headers=['Parameter', 'Value'],
            tablefmt='simple',
        ),
        file=click.get_text_stream('stderr')
    )
    
    click.echo(
        'Testing on chromosomes: {}'.format(','.join(test_chroms)),
        file=click.get_text_stream('stderr')
    )
    train, test = list(zip(*[
        (lazy_train_test_load if lazy else eager_train_test_load)(
            corpus, test_chroms
        )
        for corpus in train_corpuses
    ]))

    #if bootstrap:
    #    train_corpuses, test_corpuses = _setup_bootstrap(train_corpuses, test_corpuses, seed=bootstrap)
    
    if time_profile:
        model_kw['eval_every'] = 1
        model_kw['num_epochs'] = 1
        logger.setLevel('DEBUG')

    model, _, test_scores = mu.MutopiaModel(
        train,
        test,
        **model_kw,
    )

    click.echo('Best test score:\t{:.5f}'.format(max(test_scores)))

    if not time_profile:
        model.save(output)


@model.group("study")
def study():
    pass

@study.command("create")
@click.argument("study_name", type=str)#, help="Name of the study")
@click.argument(
    'train_corpuses',
    type=click.Path(exists=True),
    nargs=-1,
)
@click.option(
    "--min-components",
    "-min",
    type=click.IntRange(1, 1000),
    default=3,
    help="Minimum number of components",
)
@click.option(
    "-max",
    "--max-components",
    type=click.IntRange(1, 1000),
    default=20,
    help="Maximum number of components",
)
@click.option(
    "--save-model/--no-save-model",
    type=bool,
    default=False,
    is_flag=True,
    help="Save the model under `<output_dir>/<study_name>_<trial_number>.pkl`",
)
@click.option(
    "-outdir",
    "--output-dir", 
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default='.',
    help="Directory to save model files"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "-init",
    "--init-components",
    type=str,
    default=[],
    multiple=True,
    help="List of components to initialize.",
)
@click.option(
    "-fix",
    "--fix-components",
    type=str,
    default=[],
    multiple=True,
    help="List of components to fix.",
)
@click.option(
    "-creg",
    "--context-reg",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="Regularization for context model",
)
@click.option(
    "-alpha",
    "--conditioning-alpha",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="Stabilizing parameter - increase if you're getting NaNs",
)
@click.option(
    "-pi",
    "--pi-prior",
    type=click.FloatRange(0.0, 1000.0),
    default=1.0,
    help="Prior for local model, if --empirical-bayes is passed, then this prior is updated using empirical bayes.",
)
@click.option(
    "-model",
    "--locus-model-type",
    type=click.Choice(["gbt", "linear"]),
    default="gbt",
    help="Type of model to use for locus model",
)
@click.option(
    "-lr",
    "--tree-learning-rate",
    type=click.FloatRange(0.0, 100.0),
    default=None,
    help="Learning rate for tree models",
)
@click.option(
    '--convolution-width',
    '-cw',
    type=click.IntRange(0, 5),
    default=None,
    help='Width of the convolutional kernel',
)
@click.option(
    "--max-features",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Maximum number of features to use per tree",
)
@click.option(
    "--use-groups/--no-use-groups",
    type=bool,
    default=False,
    is_flag=True,
    help="Use groups in locus model",
)
@click.option(
    "--add-corpus-intercepts",
    type=bool,
    default=False,
    is_flag=True,
    help="Model interactions for each corpus",
)
@click.option(
    "-l2",
    "--l2-regularization",
    type=click.FloatRange(0.0, 100000.0),
    default=None,
    help="L2 regularization for locus model",
)
@click.option(
    "--empirical-bayes/--no-empirical-bayes",
    type=bool,
    default=True,
    is_flag=True,
    help="Use empirical bayes for prior updates",
)
@click.option(
    "--begin-prior-updates",
    type=click.IntRange(0, 100000),
    default=None,
    help="Number of epochs to wait before updating priors",
)
@click.option(
    "-stop",
    "--stop-condition",
    type=click.IntRange(0, 100000),
    default=50,
    help="Number of epochs elapsed without improvement before stopping training",
)
@click.option(
    "-epochs",
    "--num-epochs",
    type=click.IntRange(0, 100000),
    default=2000,
    help="Number of epochs to train.",
)
@click.option(
    "-lsub",
    "--locus-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Subsample rate for locus model",
)
@click.option(
    "-bsub",
    "--batch-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Subsample rate for locus model",
)
@click.option(
    "--kappa",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='"Forgetting" parameter for stochastic variational inference',
)
@click.option(
    "--tau",
    type=click.IntRange(1, 1000),
    default=None,
    help='"Offset" parameter for stochastic variational inference',
)
@click.option(
    '-e',
    '--extensive',
    count=True,
    type=int,
    default=0,
    help='How extensively to tune the hyperparameters of the model, use -e, -ee, -eee, etc. for more tuning.'
)
@click.option(
    '--test-chroms',
    '-test',
    multiple=True,
    type=str,
    default=['chr1'],
    help="Chromosomes to use for testing. If not specified, all chromosomes are used.",
)
def create_study(
    study_name: str,
    *,
    train_corpuses: List[str],
    min_components: int,
    max_components: int,
    seed: int = 0,
    save_model: bool = False,
    output_dir: str = '.',
    extensive: int = 0,
    test_chroms : List[str] = ['chr1'],
    **model_kw,
):
    if not len(train_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "train-corpuses",
            "At least one training corpus is required",
        )
    
    model_kw = {k: v for k, v in model_kw.items() if v is not None}
    click.echo(f'Fixing parameters: ')
    click.echo(tabulate(
        model_kw.items(),
        headers=['Parameter', 'Value'],
        tablefmt='simple',
    ))

    mu.tune.create_study(
        list(map(os.path.abspath, train_corpuses)),
        eval_every=5,
        min_components=min_components,
        max_components=max_components,
        study_name=study_name,
        seed=seed,
        save_model=save_model,
        output_dir=os.path.abspath(output_dir),
        extensive=extensive,
        test_chroms=test_chroms,
        **model_kw,
    )


@study.command("run")
@click.argument("study_name", type=str)
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of threads to use",
)
@click.option(
    "--lazy/--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Lazy load the underlying data to reduce memory requirements.\n",
)
@click.option(
    "--time-limit",
    '-t',
    type=int,
    default=None,
    help="Time limit for training, in minutes",
)
def run_trial(
    study_name: str,
    threads : int = 1,
    time_limit: Union[None, int] = None,
    lazy: bool = False,
):
    mu.tune.run_trial(
        study_name=study_name,
        threads=threads,
        lazy=lazy,
        time_limit=time_limit,
    )


@study.command("dashboard")
@click.argument("study_name", type=str)
def dashboard(
    study_name: str,
):
    mu.tune.dashboard(study_name)


@study.command("summary")
@click.argument("study_name", type=str)
@click.option(
    '--output',
    '-o',
    type=click.Path(writable=True),
    default=None,
    help='Output file to write summary to. If this is specified, the info is saved as a CSV instead of pretty-printed.'
)
def summary(
    study_name: str,
    output=None,
):
    trials = mu.tune.summary(study_name)
    trials = trials.sort_values('value', ascending=False, na_position='last')
    sel_cols = ['number', 'value', 'state'] 
    
    if 'user_attrs_model_path' in trials.columns:
        sel_cols+=['user_attrs_model_path']
    sel_cols += [col for col in trials.columns if col.startswith('params_')]
    
    trials = trials[sel_cols]
    trials.columns = [col.removeprefix('params_') for col in trials.columns]

    if not output is None:
        trials.to_csv(output, index=False)
    else:
        print(tabulate(
            trials,
            headers='keys',
            tablefmt="simple",
        ))


@study.command("retrain")
@click.argument("study_name", type=str)
@click.argument("trial_number", type=int)
@click.argument("output", type=click.Path(writable=True))
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of threads to use",
)
@click.option(
    "--lazy/--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Lazy load the underlying data to reduce memory requirements.\n",
)
@click.option(
    "--time-limit",
    '-t',
    type=int,
    default=None,
    help="Time limit for training, in minutes",
)
@click.option(
    "--seed",
    type=click.IntRange(0, 100000),
    default=None,
    help="Use a different random seed",
)
def retrain(
    output : str,
    study_name: str,
    trial_number: int,
    threads : int = 1,
    time_limit: Union[None, int] = None,
    seed: Union[None, int] = None,
    lazy: bool = False,
    
):
    mu.tune.retrain(
        lazy=lazy,
        study_name=study_name,
        trial_number=trial_number,
        seed=seed,
        threads=threads,
        time_limit=time_limit,
        save_name=output,
    )


@study.command("ls")
def list_studies():
    click.echo('Available studies:')
    studies = mu.tune.list_studies()
    click.echo('\n'.join(studies))


@model.command("report")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(writable=True))
def report(
    model_path: str,
    output: str,
):  
    import matplotlib.backends.backend_pdf
    matplotlib.rcParams['figure.max_open_warning']=100

    model = mu.load_model(model_path)
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(output)
    for i in range(model.n_components):
        fig = model.signature_report(i, show=False)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')

    pdf.close()


@model.command("predict")
@click.argument("model", type=click.Path(exists=True))
@click.argument("dataset", type=click.Path(exists=True))
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of threads to use",
)
def predict(
    model: str,
    dataset: str,
    threads: int = 1,
):
    
    model = mu.load_model(model)
    corpus_path = dataset
    dataset = lazy_load(dataset)

    logger.info('Setting up corpus ...')
    dataset = model.setup_corpus(dataset)

    logger.info('Annotating contributions ...')
    dataset = model.annot_contributions(dataset, threads=threads)
    dataset.contributions.name = 'contributions'
    
    '''dataset.contributions.to_netcdf(
        corpus_path,
        mode='a',
        **disk.WRITE_KW,
    )'''

    disk._write_model_state(
        dataset,
        corpus_path,
    )


@model.group("tools")
def tools():
    pass

@tools.command("simulate")
@click.argument("model", type=click.Path(exists=True))
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("output", type=click.Path(writable=True))
@click.option(
    "--seed",
    type=int,
    default=42,
)
def simulate_from_model(
    model : str,
    dataset : str,
    output : str,
    seed : int = 0,
):
    
    dataset = disk.load_dataset(
        dataset,
        with_samples=False,
    )

    model = mu.load_model(model)

    resampled = mu.tl.simulate_from_model(
        model,
        dataset,
        seed=seed,
    )

    disk.write_dataset(
        resampled,
        output,
    )
