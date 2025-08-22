"""
Core business logic for model training and analysis operations.

This module contains the core business logic separated from CLI interfaces,
making the functionality reusable across different interfaces (CLI, API, notebooks).
"""

from typing import List, Optional
import mutopia as mu
import mutopia.gtensor as gt
import mutopia.gtensor.disk_interface as disk
from mutopia.utils import logger


def train_model(
    train_corpuses: List[str],
    output: str,
    time_profile: bool = False,
    lazy: bool = False,
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
                (gt.lazy_train_test_load if lazy else gt.eager_train_test_load)(
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
    model = gt.get_mode(train[0]).TopographyModel(**model_kw)
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
    from mutopia.tuning import _run_trial_cli

    _run_trial_cli(**kw)


def launch_optimization_dashboard(study_name: str):
    """
    Launch interactive optimization dashboard.

    Parameters
    ----------
    study_name : str
        Name of the study to visualize
    """
    from mutopia.model import tuning

    tuning.dashboard(study_name)


def get_study_summary(study_name: str):
    from mutopia.model import tuning

    trials = tuning.summary(study_name)
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
    from mutopia.model import tuning

    tuning.retrain(
        lazy=lazy,
        study_name=study_name,
        trial_number=trial_number,
        seed=seed,
        threads=threads,
        time_limit=time_limit,
        save_name=output,
    )


def list_optimization_studies():
    from mutopia.model import tuning

    return tuning.list_studies()


def annot(model: str, dataset: str, output: str):
    model = mu.load_model(model)
    ds = gt.load_dataset(dataset, with_samples=False, with_state=False)
    annotated = model.annot_data(ds)
    disk.write_dataset(annotated, output)


def simulate_from_model(
    model_path: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    scale_num_mutations: float = 1.0,
):
    from mutopia.simulate import simulate_from_model

    dataset = disk.load_dataset(dataset_path, with_samples=False)
    model = mu.load_model(model_path)

    resampled = simulate_from_model(
        model,
        dataset,
        seed=seed,
        scale_num_mutations=scale_num_mutations,
    )

    disk.write_dataset(resampled, output_path)
