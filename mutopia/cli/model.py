
import mutopia as mu
from mutopia.corpus.interfaces import *
import mutopia.corpus.disk_interface as disk
from ..utils import logger
import click
from tabulate import tabulate
from typing import *
import numpy as np
import os
from tqdm import tqdm

@click.group("Model training")
def model():
    pass

@model.command("train-test-split")
@click.argument("filename", type=click.Path(exists=True))
@click.argument("test_contigs", nargs=-1, type=str)
def train_test_split(
    filename : str, 
    test_contigs : list[str],
):
    outprefix = '.'.join(filename.split(".")[:-1])
    dataset = disk.load_dataset(
        filename,
        with_samples=False
    )

    test_mask = np.isin(dataset.regions.chrom.values, test_contigs)

    disk.write_dataset(
        dataset.isel(locus=~test_mask),
        outprefix + ".train.nc",
    )

    disk.write_dataset(
        dataset.isel(locus=test_mask),
        outprefix + ".test.nc",
    )

    loader = LazySampleLoader(CorpusInterface(dataset))

    for sample_name in tqdm(
        dataset.sample.values,
        desc=f'Writing samples',
        ncols=100,
    ):
        sample = loader.fetch_sample(sample_name)
        disk.write_sample(
            outprefix + ".train.nc",
            sample.isel(locus=~test_mask),
            sample_name=f'X/{sample_name}',
        )
        disk.write_sample(
            outprefix + ".test.nc",
            sample.isel(locus=test_mask),
            sample_name=f'X/{sample_name}',
        )


@model.command("train")
@click.argument("output", type=click.Path(exists=False))
@click.option(
    "-train",
    "--train-corpuses",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Paths to training corpuses. Invoke multiple times for multiple corpuses: `-train <path1> -train <path2>`",
)
@click.option(
    "-test",
    "--test-corpuses",
    required=True,
    type=click.Path(exists=True),
    multiple=True,
    help="Paths to testing corpuses. Invoke multiple times for multiple corpuses: `-test <path1> -test <path2>`",
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
    default=None,
    multiple=True,
    help="List of components to initialize.",
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
    type=click.IntRange(1, 5),
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
    type=click.FloatRange(0.0, 1000.0),
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
    "-eval",
    "--eval-every",
    type=click.IntRange(1, 1000),
    default=1,
    help="Evaluate every N epochs",
)
@click.option('-v', '--verbose', count=True)
def train(
    output,
    *,
    train_corpuses: List[str],
    test_corpuses: List[str],
    init_components : Union[None, List[str]],
    **model_kw,
):  
    if not len(train_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "train-corpuses",
            "At least one training corpus is required",
        )

    if not len(test_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "test-corpuses",
            "At least one testing corpus is required",
        )
    
    model_kw = {k: v for k, v in model_kw.items() if v is not None}
    click.echo(f'Training model with parameters: ')
    click.echo(tabulate(
        model_kw.items(),
        headers=['Parameter', 'Value'],
        tablefmt='simple',
    ))
    
    train_corpuses = tuple(map(lazy_load, train_corpuses))
    test_corpuses = tuple(map(lazy_load, test_corpuses))

    model, *_ = mu.MutopiaModel(
        train_corpuses,
        test_corpuses,
        init_components=init_components if len(init_components) > 0 else None,
        **model_kw,
    )

    model.save(output)


@model.group("study")
def study():
    pass

@study.command("create")
@click.argument("study_name", type=str)#, help="Name of the study")
@click.option(
    "-train",
    "--train-corpuses",
    multiple=True,
    type=click.Path(exists=True),
    help="Paths to training corpuses. Invoke multiple times for multiple corpuses: `-train <path1> -train <path2>`",
)
@click.option(
    "-test",
    "--test-corpuses",
    type=click.Path(exists=True),
    multiple=True,
    help="Paths to testing corpuses. Invoke multiple times for multiple corpuses: `-test <path1> -test <path2>`",
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
    default=None,
    multiple=True,
    help="List of components to initialize.",
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
    type=click.IntRange(1, 5),
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
    type=click.FloatRange(0.0, 1000.0),
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
def create_study(
    study_name: str,
    *,
    train_corpuses: List[str],
    test_corpuses: List[str],
    min_components: int,
    max_components: int,
    seed: int = 0,
    save_model: bool = False,
    output_dir: str = '.',
    extensive: int = 0,
    init_components: Union[List[str], None] = None,
    **model_kw,
):
    if not len(train_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "train-corpuses",
            "At least one training corpus is required",
        )

    if not len(test_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "test-corpuses",
            "At least one testing corpus is required",
        )
    
    model_kw = {k: v for k, v in model_kw.items() if v is not None}
    click.echo(f'Fixing parameters: ')
    click.echo(tabulate(
        model_kw.items(),
        headers=['Parameter', 'Value'],
        tablefmt='simple',
    ))

    mu.create_study(
        list(map(os.path.abspath, train_corpuses)),
        list(map(os.path.abspath, test_corpuses)),
        eval_every=5,
        min_components=min_components,
        max_components=max_components,
        study_name=study_name,
        seed=seed,
        save_model=save_model,
        output_dir=os.path.abspath(output_dir),
        extensive=extensive,
        init_components=init_components if len(init_components) > 0 else None,
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
def run_trial(
    study_name: str,
    threads : int = 1,
):
    mu.run_trial(
        study_name=study_name,
        threads=threads,
    )


@study.command("dashboard")
@click.argument("study_name", type=str)
def dashboard(
    study_name: str,
):
    mu.dashboard(study_name)


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
    trials = mu.tuning.summary(study_name)
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


@study.command("ls")
def list_studies():
    click.echo('Available studies:')
    studies = mu.tuning.list_studies()
    click.echo('\n'.join(studies))
