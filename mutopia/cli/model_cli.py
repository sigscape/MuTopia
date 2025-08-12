import click
from tabulate import tabulate
from typing import *
import numpy as np
from mutopia.tuning import create_study
from . import model_core


@click.group("Model training")
def model():
    """
    Train and manage genome topography models for genomic data analysis.

    Tools for training models that learn genomic components from G-Tensor datasets,
    hyperparameter optimization, and model analysis.
    """
    pass


@model.command("train", short_help="Train a genome topography model")
@click.argument(
    "train_corpuses", type=click.Path(exists=True), nargs=-1, metavar="CORPUS..."
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True),
    required=True,
    help="Path to save the trained model file",
)
@click.option(
    "-k",
    "--num-components",
    type=click.IntRange(1, 1000),
    required=True,
    help="Number of components to learn",
)
@click.option(
    "--seed", type=int, default=0, help="Random seed for reproducible training"
)
@click.option(
    "-init",
    "--init-components",
    type=str,
    default=[],
    multiple=True,
    help="Names of components to initialize with known values",
)
@click.option(
    "-fix",
    "--fix-components",
    type=str,
    default=[],
    multiple=True,
    help="Names of components to fix during training (not updated)",
)
@click.option(
    "-creg",
    "--context-reg",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="L2 regularization strength for the context frequency model",
)
@click.option(
    "-alpha",
    "--conditioning-alpha",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="Dirichlet concentration parameter for stabilizing training (increase if getting NaNs)",
)
@click.option(
    "-pi",
    "--pi-prior",
    type=click.FloatRange(0.0, 1000.0),
    default=None,
    help="Prior strength for the local exposure model (updated via empirical Bayes if enabled)",
)
@click.option(
    "-model",
    "--locus-model-type",
    type=click.Choice(["gbt", "linear"]),
    default="gbt",
    help="Type of model for learning locus effects: 'gbt' (gradient boosting) or 'linear'",
)
@click.option(
    "-lr",
    "--tree-learning-rate",
    type=click.FloatRange(0.0, 100.0),
    default=None,
    help="Learning rate for gradient boosting trees (lower values = more conservative training)",
)
@click.option(
    "--max-features",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Fraction of features to consider for each tree split (0.0-1.0)",
)
@click.option(
    "--convolution-width",
    "-cw",
    type=click.IntRange(0, 5),
    default=None,
    help="Width of convolutional kernel for feature processing (0=no convolution)",
)
@click.option(
    "--use-groups/--no-use-groups",
    type=bool,
    default=True,
    is_flag=True,
    help="Whether to use feature groups in the locus model for regularization",
)
@click.option(
    "--add-corpus-intercepts",
    type=bool,
    default=False,
    is_flag=True,
    help="Add corpus-specific intercept terms to model dataset-specific effects",
)
@click.option(
    "-l2",
    "--l2-regularization",
    type=click.FloatRange(0.0, 100000.0),
    default=None,
    help="L2 regularization strength for locus models",
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
    help="Maximum number of training epochs",
)
@click.option(
    "-lsub",
    "--locus-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Fraction of genomic loci to subsample for each training step (0.0-1.0)",
)
@click.option(
    "-bsub",
    "--batch-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Fraction of samples to subsample for each training step (0.0-1.0)",
)
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of parallel threads to use for training",
)
@click.option(
    "--kappa",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Forgetting parameter for stochastic variational inference (0.0=no forgetting, 1.0=full forgetting)",
)
@click.option(
    "--tau",
    type=click.IntRange(1, 1000),
    default=None,
    help="Delay parameter for stochastic variational inference (higher=slower adaptation)",
)
@click.option(
    "--init-variance-theta",
    "-ivt",
    type=click.FloatRange(0.0, 1000.0),
    default=None,
    help="Initial variance for theta parameters in variational inference",
)
@click.option(
    "-eval",
    "--eval-every",
    type=click.IntRange(1, 1000),
    default=5,
    help="Evaluate model performance every N training epochs",
)
@click.option(
    "--time-profile/--no-time-profile",
    type=bool,
    default=False,
    is_flag=True,
    help="Profile training time and performance (runs only 1 epoch for timing)",
)
@click.option(
    "--lazy/--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Use lazy loading to reduce memory usage vs eager loading for faster access",
)
@click.option(
    "--time-limit",
    "-t",
    type=int,
    default=None,
    help="Maximum training time in seconds (training stops when limit reached)",
)
@click.option(
    "--bootstrap",
    "-b",
    type=click.IntRange(0, np.iinfo(np.int32).max),
    default=None,
    help="Random seed for bootstrapping training and test datasets",
)
@click.option(
    "--test-chroms",
    "-test",
    multiple=True,
    type=str,
    default=["chr1"],
    help="Chromosomes to reserve for testing (remaining chroms used for training)",
)
def train(
    *,
    output,
    train_corpuses: List[str],
    time_profile: bool = False,
    lazy: bool = False,
    bootstrap: Union[int, None] = None,
    test_chroms: List[str] = ["chr1"],
    **model_kw,
):
    """
    Train a genome topography model from G-Tensor datasets.
    
    CORPUS... are G-Tensor dataset files (.nc format) containing genomic samples
    with data and features for training the model.
    
    Examples:
        # Basic training with 5 components
        model train dataset1.nc dataset2.nc -k 5 -o model.pkl
        
        # Advanced training with regularization and custom parameters
        model train data.nc -k 10 -o model.pkl --context-reg 0.1 \\
                    --locus-model-type linear --l2-regularization 0.01 \\
                    --test-chroms chr1 chr2 --threads 8
        
        # Training with feature convolution and empirical Bayes
        model train corpus.nc -k 8 -o model.pkl --convolution-width 2 \\
                    --empirical-bayes --pi-prior 1.0 --epochs 1000
    """
    if not len(train_corpuses) > 0:
        raise click.exceptions.BadOptionUsage(
            "train-corpuses",
            "At least one training corpus is required",
        )

    model_kw = {k: v for k, v in model_kw.items() if v is not None}
    click.echo(
        f"Training model with parameters: ", file=click.get_text_stream("stderr")
    )
    click.echo(
        tabulate(
            model_kw.items(),
            headers=["Parameter", "Value"],
            tablefmt="simple",
        ),
        file=click.get_text_stream("stderr"),
    )

    click.echo(
        "Testing on chromosomes: {}".format(",".join(test_chroms)),
        file=click.get_text_stream("stderr"),
    )

    try:
        best_score = model_core.train_model(
            train_corpuses=train_corpuses,
            output=output,
            time_profile=time_profile,
            lazy=lazy,
            bootstrap=bootstrap,
            test_chroms=test_chroms,
            **model_kw,
        )

        click.echo("Best test score:\t{:.5f}".format(best_score))

    except ValueError as e:
        raise click.exceptions.BadOptionUsage("train-corpuses", str(e))


@model.group("study")
def study():
    """
    Hyperparameter optimization studies for model training.

    Tools for systematic hyperparameter tuning using Optuna optimization to find
    optimal model configurations based on validation performance.
    """
    pass


@study.command("create", short_help="Create a new hyperparameter optimization study")
@click.argument("study_name", type=str, metavar="STUDY_NAME")
@click.option(
    "-ds",
    "--dataset",
    type=click.Tuple([click.Path(exists=True), click.Path(exists=True)]),
    multiple=True,
    metavar="TRAIN_DATA TEST_DATA",
    help="Paired training and testing dataset files (.nc format) - can specify multiple pairs",
)
@click.option(
    "--min-components",
    "-min",
    type=click.IntRange(1, 1000),
    default=3,
    help="Minimum number of components to explore in optimization",
)
@click.option(
    "-max",
    "--max-components",
    type=click.IntRange(1, 1000),
    default=20,
    help="Maximum number of components to explore in optimization",
)
@click.option(
    "--save-model/--no-save-model",
    type=bool,
    default=False,
    is_flag=True,
    help="Save trained models for each trial under '<output_dir>/<study_name>_<trial_number>.pkl'",
)
@click.option(
    "-outdir",
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=".",
    help="Directory to save model files and study results",
)
@click.option(
    "--seed", type=int, default=42, help="Random seed for reproducible optimization"
)
@click.option(
    "-init",
    "--init-components",
    type=str,
    default=[],
    multiple=True,
    help="Names of components to initialize with known values",
)
@click.option(
    "-fix",
    "--fix-components",
    type=str,
    default=[],
    multiple=True,
    help="Names of components to fix during training (not optimized)",
)
@click.option(
    "-creg",
    "--context-reg",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="L2 regularization strength for the context frequency model",
)
@click.option(
    "-alpha",
    "--conditioning-alpha",
    type=click.FloatRange(0.0, 10.0),
    default=None,
    help="Dirichlet concentration parameter for stabilizing training (increase if getting NaNs)",
)
@click.option(
    "-pi",
    "--pi-prior",
    type=click.FloatRange(0.0, 1000.0),
    default=1.0,
    help="Prior strength for the local exposure model (updated via empirical Bayes if enabled)",
)
@click.option(
    "-model",
    "--locus-model-type",
    type=click.Choice(["gbt", "linear"]),
    default="gbt",
    help="Type of model for learning locus effects: 'gbt' (gradient boosting) or 'linear'",
)
@click.option(
    "-lr",
    "--tree-learning-rate",
    type=click.FloatRange(0.0, 100.0),
    default=None,
    help="Learning rate for gradient boosting trees (lower values = more conservative training)",
)
@click.option(
    "--convolution-width",
    "-cw",
    type=click.IntRange(0, 5),
    default=None,
    help="Width of convolutional kernel for feature processing (0=no convolution)",
)
@click.option(
    "--max-features",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Fraction of features to consider for each tree split (0.0-1.0)",
)
@click.option(
    "--use-groups/--no-use-groups",
    type=bool,
    default=False,
    is_flag=True,
    help="Whether to use feature groups in the locus model for regularization",
)
@click.option(
    "--add-corpus-intercepts",
    type=bool,
    default=False,
    is_flag=True,
    help="Add corpus-specific intercept terms to model dataset-specific effects",
)
@click.option(
    "-l2",
    "--l2-regularization",
    type=click.FloatRange(0.0, 100000.0),
    default=None,
    help="L2 regularization strength for locus models",
)
@click.option(
    "--empirical-bayes/--no-empirical-bayes",
    type=bool,
    default=True,
    is_flag=True,
    help="Use empirical Bayes for automatic prior parameter updates during training",
)
@click.option(
    "--begin-prior-updates",
    type=click.IntRange(0, 100000),
    default=None,
    help="Number of training epochs to complete before starting empirical Bayes updates",
)
@click.option(
    "-stop",
    "--stop-condition",
    type=click.IntRange(0, 100000),
    default=50,
    help="Number of epochs without improvement before early stopping",
)
@click.option(
    "-epochs",
    "--num-epochs",
    type=click.IntRange(0, 100000),
    default=2000,
    help="Maximum number of training epochs per trial",
)
@click.option(
    "-lsub",
    "--locus-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Fraction of genomic loci to subsample for each training step (reduces memory)",
)
@click.option(
    "-bsub",
    "--batch-subsample",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Fraction of samples to subsample for each training step (reduces memory)",
)
@click.option(
    "--kappa",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Forgetting parameter for stochastic variational inference (0.0=no forgetting, 1.0=full forgetting)",
)
@click.option(
    "--tau",
    type=click.IntRange(1, 1000),
    default=None,
    help="Delay parameter for stochastic variational inference (higher=slower adaptation)",
)
@click.option(
    "-e",
    "--extensive",
    count=True,
    type=int,
    default=0,
    help="How extensively to tune hyperparameters: -e (basic), -ee (moderate), -eee (extensive)",
)
@click.option(
    "--test-chroms",
    "-test",
    multiple=True,
    type=str,
    default=["chr1"],
    help="Chromosomes to reserve for testing (remaining chromosomes used for training)",
)
def _create_study(
    study_name: str,
    *,
    dataset: List[Tuple[str, str]] = [],
    min_components: int,
    max_components: int,
    seed: int = 0,
    save_model: bool = False,
    output_dir: str = ".",
    extensive: int = 0,
    test_chroms: List[str] = ["chr1"],
    **model_kw,
):
    """
    Create a hyperparameter optimization study for genome topography models.
    
    Sets up an Optuna study to explore hyperparameter combinations and find
    optimal model configurations based on validation performance.
    
    Examples:
        # Basic study with component range exploration
        model study create my_study --dataset train.nc test.nc \\
                            --min-components 3 --max-components 15
        
        # Extensive hyperparameter search with model saving
        model study create comprehensive_study \\
                            --dataset data1_train.nc data1_test.nc \\
                            --dataset data2_train.nc data2_test.nc \\
                            --min-components 5 --max-components 20 \\
                            --extensive 3 --save-model --output-dir ./models/
        
        # Study with fixed parameters
        model study create focused_study --dataset train.nc test.nc \\
                            --min-components 8 --max-components 12 \\
                            --locus-model-type linear --empirical-bayes
    """
    if not len(dataset) > 0:
        raise click.exceptions.BadOptionUsage(
            "train-corpuses",
            "At least one training corpus is required",
        )

    model_kw = {k: v for k, v in model_kw.items() if v is not None}

    click.echo(f"Fixing parameters: ")
    click.echo(
        tabulate(
            model_kw.items(),
            headers=["Parameter", "Value"],
            tablefmt="simple",
        )
    )

    train, test = list(zip(*dataset))

    try:
        create_study(
            train=train,
            test=test,
            eval_every=5,
            min_components=min_components,
            max_components=max_components,
            study_name=study_name,
            seed=seed,
            save_model=save_model,
            output_dir=output_dir,
            extensive=extensive,
            **model_kw,
        )
        click.echo(f"Successfully created optimization study: {study_name}")
    except Exception as e:
        raise click.ClickException(f"Failed to create study: {str(e)}")


@study.command("run", short_help="Run optimization trials for an existing study")
@click.argument("study_name", type=str, metavar="STUDY_NAME")
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of parallel threads for training (more threads = faster but more memory)",
)
@click.option(
    "--lazy/--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Use lazy loading to reduce memory usage vs eager loading for faster training",
)
@click.option(
    "--time-limit",
    "-t",
    type=int,
    default=None,
    help="Maximum time limit for each trial in minutes (trials stop when exceeded)",
)
def run_trial(**kw):
    """
    Execute optimization trials for a previously created study.

    Runs hyperparameter optimization using Optuna's suggestion engine to test
    different parameter combinations and report performance.

    Examples:
        # Run trials with default settings
        model study run my_study

        # Run with multiple threads and time limits
        model study run my_study --threads 8 --time-limit 30

        # Run with lazy loading for large datasets
        model study run my_study --lazy --threads 4
    """
    try:
        model_core.run_optimization_trial(**kw)
        click.echo("Trial completed successfully")
    except Exception as e:
        raise click.ClickException(f"Trial execution failed: {str(e)}")


@study.command("dashboard", short_help="Launch interactive optimization dashboard")
@click.argument("study_name", type=str, metavar="STUDY_NAME")
def dashboard(
    study_name: str,
):
    """
    Launch an interactive web dashboard for monitoring study progress.

    Opens a browser interface showing real-time optimization progress,
    parameter importance, trial history, and convergence analysis.

    Examples:
        # Launch dashboard for study monitoring
        model study dashboard my_study
    """
    try:
        model_core.launch_optimization_dashboard(study_name)
        click.echo(f"Dashboard launched for study: {study_name}")
    except Exception as e:
        raise click.ClickException(f"Failed to launch dashboard: {str(e)}")


@study.command("summary", short_help="Show study results and best trials")
@click.argument("study_name", type=str, metavar="STUDY_NAME")
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    default=None,
    help="Output CSV file path for saving results (if not specified, prints to console)",
)
def summary(
    study_name: str,
    output=None,
):
    """
    Display a summary of optimization study results.

    Shows trial performance rankings, best parameter combinations, and
    optimization progress. Can export results to CSV for analysis.

    Examples:
        # Display results in console
        model study summary my_study

        # Export results to CSV file
        model study summary my_study --output results.csv
    """
    try:
        trials = model_core.get_study_summary(study_name)
        trials = trials.sort_values("value", ascending=False, na_position="last")
        sel_cols = ["number", "value", "state"]

        if "user_attrs_model_path" in trials.columns:
            sel_cols += ["user_attrs_model_path"]
        sel_cols += [col for col in trials.columns if col.startswith("params_")]

        trials = trials[sel_cols]
        trials.columns = [col.removeprefix("params_") for col in trials.columns]

        if not output is None:
            trials.to_csv(output, index=False)
            click.echo(f"Results saved to: {output}")
        else:
            print(
                tabulate(
                    trials,
                    headers="keys",
                    tablefmt="simple",
                )
            )
    except Exception as e:
        raise click.ClickException(f"Failed to get study summary: {str(e)}")


@study.command(
    "retrain", short_help="Retrain a specific trial with optional modifications"
)
@click.argument("study_name", type=str, metavar="STUDY_NAME")
@click.argument("trial_number", type=int, metavar="TRIAL_NUM")
@click.argument("output", type=click.Path(writable=True), metavar="OUTPUT_PATH")
@click.option(
    "-@",
    "--threads",
    type=click.IntRange(1, 1000),
    default=1,
    help="Number of parallel threads for retraining",
)
@click.option(
    "--lazy/--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Use lazy loading to reduce memory usage during retraining",
)
@click.option(
    "--time-limit",
    "-t",
    type=int,
    default=None,
    help="Maximum retraining time in minutes",
)
@click.option(
    "--seed",
    type=click.IntRange(0, 100000),
    default=None,
    help="Use different random seed for retraining (default: use original trial seed)",
)
def retrain(
    output: str,
    study_name: str,
    trial_number: int,
    threads: int = 1,
    time_limit: Union[None, int] = None,
    seed: Union[None, int] = None,
    lazy: bool = False,
):
    """
    Retrain a specific trial from an optimization study.

    Takes the best parameters from a completed trial and retrains the model,
    optionally with different computational settings or random seed.

    Examples:
        # Retrain best trial from study
        model study retrain my_study 42 final_model.pkl

        # Retrain with more computational resources
        model study retrain my_study 42 model.pkl --threads 16 --time-limit 120

        # Retrain with different random seed for ensemble
        model study retrain my_study 42 model_v2.pkl --seed 999 --lazy
    """
    try:
        model_core.retrain_trial(
            study_name=study_name,
            trial_number=trial_number,
            output=output,
            lazy=lazy,
            seed=seed,
            threads=threads,
            time_limit=time_limit,
        )
        click.echo(
            f"Successfully retrained trial {trial_number} from study {study_name}"
        )
        click.echo(f"Model saved to: {output}")
    except Exception as e:
        raise click.ClickException(f"Retraining failed: {str(e)}")


@study.command("ls", short_help="List all available optimization studies")
def list_studies():
    """
    List all available hyperparameter optimization studies.

    Shows the names of all studies that have been created and are available
    for running trials, viewing results, or retraining models.

    Examples:
        # List all studies
        model study ls
    """
    try:
        click.echo("Available studies:")
        studies = model_core.list_optimization_studies()
        if studies:
            click.echo("\n".join(studies))
        else:
            click.echo("No studies found.")
    except Exception as e:
        raise click.ClickException(f"Failed to list studies: {str(e)}")
    

@model.command("annot")
@click.argument("model", type=click.Path(exists=True), metavar="MODEL_FILE")
@click.argument("dataset", type=click.Path(exists=True), metavar="DATASET_FILE")
@click.argument("output", type=click.Path(writable=True), metavar="OUTPUT_FILE")
def annot(
    model : str,
    dataset : str,
    output : str,
):
    model_core.annot(model, dataset, output)


@model.group("tools")
def tools():
    """
    Utility tools for model analysis and data simulation.

    This command group provides specialized tools for working with trained
    models, including data simulation and advanced analysis utilities.
    """
    pass


@tools.command("simulate", short_help="Simulate genomic data from trained model")
@click.argument("model", type=click.Path(exists=True), metavar="MODEL_FILE")
@click.argument("dataset", type=click.Path(exists=True), metavar="TEMPLATE_DATASET")
@click.argument("output", type=click.Path(writable=True), metavar="OUTPUT_DATASET")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible simulation",
)
@click.option(
    "-scale",
    "--scale-num-mutations",
    type=float,
    default=1.0,
    help="Scaling factor for mutation counts (1.0=same as template, 2.0=double mutations)",
)
def simulate_from_model(
    model: str, dataset: str, output: str, seed: int = 0, scale_num_mutations=1.0
):
    """
    Simulate genomic mutation data from a trained genome topography model.

    Uses learned component patterns and activity models to generate synthetic
    genomic datasets following the same patterns as the original training data.
    Useful for model validation, power analysis, and method development.

    Simulation Process:
    - Uses template dataset structure and genomic features
    - Samples mutation counts based on learned activity patterns
    - Generates mutations according to component probabilities
    - Preserves spatial patterns and feature dependencies

    Applications:
    - Model validation and testing
    - Power analysis for study design
    - Benchmarking component detection methods
    - Generating training data for new algorithms

    Examples:
        # Basic simulation with same mutation burden
        model tools simulate trained_model.pkl template.nc simulated.nc

        # Simulate with doubled mutation burden
        model tools simulate model.pkl template.nc high_burden.nc --scale 2.0

        # Reproducible simulation with specific seed
        model tools simulate model.pkl template.nc sim_data.nc --seed 12345
    """
    try:
        model_core.simulate_from_model(
            model_path=model,
            dataset_path=dataset,
            output_path=output,
            seed=seed,
            scale_num_mutations=scale_num_mutations,
        )
        click.echo(f"Simulated data saved to: {output}")
    except Exception as e:
        raise click.ClickException(f"Simulation failed: {str(e)}")
