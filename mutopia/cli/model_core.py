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


def annot(model: str, dataset: str, output: str):
    model = mu.load_model(model)
    ds = mu.gt.load_dataset(dataset, with_samples=False, with_state=False)
    annotated = model.annot_data(ds)
    disk.write_dataset(annotated, output)


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
