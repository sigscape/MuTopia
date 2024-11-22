import click
from typing import *
import mutopia as mu

@click.group("Model training")
def model():
    pass

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
    default=0.0001,
    help="Regularization for context model",
)
@click.option(
    "-alpha",
    "--conditioning-alpha",
    type=click.FloatRange(0.0, 10.0),
    default=1e-5,
    help="Stabilizing parameter - increase if you're getting NaNs",
)
@click.option(
    "-mreg",
    "--mutation-reg",
    type=click.FloatRange(0.0, 10.0),
    default=0.0005,
    help="Regularization for mutation model",
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
    default=0.1,
    help="Learning rate for tree models",
)
@click.option(
    "--max-features",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
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
    "--smoothing-size",
    type=click.IntRange(1, 100000),
    default=1000,
    help="Smooth locus features using this window size (in bp).",
)
@click.option(
    "--add-corpus-intercepts",
    type=bool,
    default=True,
    is_flag=True,
    help="Model interactions for each corpus",
)
@click.option(
    "-l2",
    "--l2-regularization",
    type=click.FloatRange(0.0, 1000.0),
    default=1.0,
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
    default=10,
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
    "-sub",
    "--locus-subsample",
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
    default=0.5,
    help='"Forgetting" parameter for stochastic variational inference',
)
@click.option(
    "--tau",
    type=click.IntRange(1, 1000),
    default=1,
    help='"Offset" parameter for stochastic variational inference',
)
@click.option(
    "-eval",
    "--eval-every",
    type=click.IntRange(1, 1000),
    default=10,
    help="Evaluate every N epochs",
)
@click.option(
    "--sparse/--no-sparse",
    type=bool,
    default=True,
    is_flag=True,
    help="Use sparse matrices in the updates.",
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
    
    train_corpuses = tuple(map(mu.corpus.load_dataset, train_corpuses))
    test_corpuses = tuple(map(mu.corpus.load_dataset, test_corpuses))

    model = mu.MutopiaModel(
        train_corpuses,
        test_corpuses,
        init_components=init_components if len(init_components) > 0 else None,
        **model_kw,
    )

    model.save(output)



@model.command("study-create")
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
    help="Save the model under `<model_prefix>/<study_name>_<trial_number>.pkl`",
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
    default=0.0001,
    help="Regularization for context model",
)
@click.option(
    "-alpha",
    "--conditioning-alpha",
    type=click.FloatRange(0.0, 10.0),
    default=5e-5,
    help="Stabilizing parameter - increase if you're getting NaNs",
)
@click.option(
    "-mreg",
    "--mutation-reg",
    type=click.FloatRange(0.0, 10.0),
    default=0.0005,
    help="Regularization for mutation model",
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
    default=0.1,
    help="Learning rate for tree models",
)
@click.option(
    "--max-features",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
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
    "--smoothing-size",
    type=click.IntRange(1, 100000),
    default=1000,
    help="Smooth locus features using this window size (in bp).",
)
@click.option(
    "--add-corpus-intercepts",
    type=bool,
    default=True,
    is_flag=True,
    help="Model interactions for each corpus",
)
@click.option(
    "-l2",
    "--l2-regularization",
    type=click.FloatRange(0.0, 1000.0),
    default=1.0,
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
    default=10,
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
    "-sub",
    "--locus-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Subsample rate for locus model",
)
@click.option(
    "--kappa",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    help='"Forgetting" parameter for stochastic variational inference',
)
@click.option(
    "--tau",
    type=click.IntRange(1, 1000),
    default=1,
    help='"Offset" parameter for stochastic variational inference',
)
@click.option(
    "-eval",
    "--eval-every",
    type=click.IntRange(1, 1000),
    default=10,
    help="Evaluate every N epochs",
)
@click.option(
    "--sparse/--no-sparse",
    type=bool,
    default=True,
    is_flag=True,
    help="Use sparse matrices in the updates.",
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

    mu.create_study(
        train_corpuses,
        test_corpuses,
        min_components=min_components,
        max_components=max_components,
        study_name=study_name,
        seed=seed,
        save_model=save_model,
        output_dir=output_dir,
        init_components=init_components if len(init_components) > 0 else None,
        **model_kw,
    )


@model.command("study-trial")
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


@model.command("study-dashboard")
@click.argument("study_name", type=str)
def dashboard(
    study_name: str,
):
    mu.dashboard(study_name)

