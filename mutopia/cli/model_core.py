"""
Core business logic for model training and analysis operations.

This module contains the core business logic separated from CLI interfaces,
making the functionality reusable across different interfaces (CLI, API, notebooks).
"""

import mutopia as mu
from mutopia.gtensor import *
from mutopia.dtypes import get_mode
from mutopia.tuning import _run_trial_cli
from mutopia.model import GtensorInterface as CS
import mutopia.gtensor.disk_interface as disk
from ..utils import logger

import numpy as np
import os
from typing import List, Tuple, Union, Optional
import matplotlib
import matplotlib.backends.backend_pdf


'''def setup_bootstrap(train_corpuses, test_corpuses, seed):
    """
    Set up bootstrapped training and testing corpuses.
    
    Parameters
    ----------
    train_corpuses : list
        List of training corpus objects
    test_corpuses : list
        List of testing corpus objects  
    seed : int
        Random seed for bootstrap sampling
        
    Returns
    -------
    tuple
        Bootstrapped (train_corpuses, test_corpuses)
    """
    logger.warning("Bootstrapping training and testing corpuses ...")
    test_corpus_map = {CS.get_name(corpus): corpus for corpus in test_corpuses}

    train_corpuses = tuple(
        [
            BootstrapCorpus(corpus, np.random.RandomState(seed))
            for corpus in train_corpuses
        ]
    )

    test_corpuses = tuple(
        [
            DifferentSamples(
                test_corpus_map[CS.get_name(corpus)], corpus.list_samples()
            )
            for corpus in train_corpuses
        ]
    )

    return train_corpuses, test_corpuses'''


def train_model(
    train_corpuses: List[str],
    output: str,
    time_profile: bool = False,
    lazy: bool = False,
    bootstrap: Optional[int] = None,
    test_chroms: List[str] = ["chr1"],
    **model_kw,
):
    if not len(train_corpuses) > 0:
        raise ValueError("At least one training corpus is required")

    # Filter out None values from model parameters
    model_kw = {k: v for k, v in model_kw.items() if v is not None}

    # Load train/test data splits
    train, test = list(
        zip(
            *[
                (lazy_train_test_load if lazy else eager_train_test_load)(
                    corpus, *test_chroms
                )
                for corpus in train_corpuses
            ]
        )
    )

    # Configure profiling mode
    if time_profile:
        model_kw["eval_every"] = 1
        model_kw["num_epochs"] = 1
        logger.setLevel("DEBUG")

    # Train the model
    model = get_mode(train[0]).TopographyModel(**model_kw)
    model = model.fit(*train, test_datasets=test)

    best_score = max(model.test_scores_)

    # Save model if not profiling
    if not time_profile:
        model.save(output)

    return best_score
    


def run_optimization_trial(**kw):
    """
    Execute optimization trials for an existing study.

    Parameters
    ----------
    **kw
        Trial execution parameters passed to the tuning engine
    """
    _run_trial_cli(**kw)


def launch_optimization_dashboard(study_name: str):
    """
    Launch interactive optimization dashboard.

    Parameters
    ----------
    study_name : str
        Name of the study to visualize
    """
    mu.tune.dashboard(study_name)


def get_study_summary(study_name: str):
    trials = mu.tune.summary(study_name)
    trials = trials.sort_values("value", ascending=False, na_position="last")
    sel_cols = ["number", "value", "state"]

    if "user_attrs_model_path" in trials.columns:
        sel_cols += ["user_attrs_model_path"]
    sel_cols += [col for col in trials.columns if col.startswith("params_")]

    trials = trials[sel_cols]
    trials.columns = [col.removeprefix("params_") for col in trials.columns]

    return trials


def retrain_trial(
    study_name: str,
    trial_number: int,
    output: str,
    threads: int = 1,
    time_limit: Optional[int] = None,
    seed: Optional[int] = None,
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


def list_optimization_studies():
    return mu.tune.list_studies()


def generate_model_report(model_path: str, output: Optional[str] = None):
    matplotlib.rcParams["figure.max_open_warning"] = 100

    model: mu.model.Model = mu.load_model(model_path)

    if output is None:
        output = os.path.splitext(model_path)[0] + ".report.pdf"

    pdf = matplotlib.backends.backend_pdf.PdfPages(output)

    # Generate signature panel
    fig = model.signature_panel(ncols=3, show=False)
    pdf.savefig(fig, dpi=300, bbox_inches="tight")

    # Generate individual component reports
    for i in range(model.n_components):
        fig = model.signature_report(i, show=False)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")

    pdf.close()


def predict_model(
    model_path: str,
    dataset_path: str,
    threads: int = 1,
    contributions: bool = True,
    output: Optional[str] = None,
):
    """
    Apply trained model to predict component contributions and activities.

    Parameters
    ----------
    model_path : str
        Path to trained model file
    dataset_path : str
        Path to dataset for prediction
    threads : int, default 1
        Number of parallel threads
    contributions : bool, default True
        Whether to calculate detailed contributions
    output : Optional[str], default None
        Output file path (None for in-place modification)
    """
    model = mu.load_model(model_path)
    corpus_path = dataset_path
    dataset = lazy_load(dataset_path)

    inplace = not ("component" in dataset.dims) and output is None
    if not inplace and output is None:
        raise ValueError(
            'Output file must be specified since the dataset already has a "state".'
        )

    if inplace:
        logger.info("Modifying dataset in place ...")

    dataset = CorpusInterface(dataset)

    logger.info("Setting up corpus ...")
    dataset = model.setup_corpus(dataset)

    if contributions:
        logger.info("Annotating contributions ...")
        dataset = model.annot_contributions(dataset, threads=threads)
        dataset.contributions.name = "contributions"

    if inplace:
        disk._write_model_state(dataset, corpus_path)
        dataset.contributions.to_netcdf(corpus_path, mode="a", **disk.WRITE_KW)
    else:
        disk.write_dataset(dataset, output, bar=True)


def calculate_shap_values(
    model_path: str,
    dataset_path: str,
    components: List[str] = [],
    threads: int = 1,
    n_samples: int = 2000,
    scan: bool = False,
):
    corpus_path = dataset_path
    model = mu.load_model(model_path)
    dataset = lazy_load(dataset_path)

    logger.info("Calculating SHAP values ...")
    dataset = model.annot_SHAP_values(
        dataset,
        *components,
        threads=threads,
        n_samples=n_samples,
        scan=scan,
    )

    dataset.SHAP_values.name = "SHAP_values"
    dataset.SHAP_values.to_netcdf(corpus_path, mode="a", group="varm", **disk.WRITE_KW)


def export_to_excel(
    model_path: str,
    dataset_path: str,
    output_path: str,
):
    model = mu.load_model(model_path)
    dataset = mu.gt.load_dataset(dataset_path, with_samples=False)

    model.execel_report(dataset, output_path)


def simulate_from_model(
    model_path: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    scale_num_mutations: float = 1.0,
):

    dataset = disk.load_dataset(dataset_path, with_samples=False)
    model = mu.load_model(model_path)

    resampled = mu.tl.simulate_from_model(
        model,
        dataset,
        seed=seed,
        scale_num_mutations=scale_num_mutations,
    )

    disk.write_dataset(resampled, output_path)
