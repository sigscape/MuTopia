"""
Core business logic for model training and analysis operations.

This module contains the core business logic separated from CLI interfaces,
making the functionality reusable across different interfaces (CLI, API, notebooks).
"""
from typing import List, Optional
import mutopia.gtensor as gt
import mutopia.gtensor.disk_interface as disk
from mutopia.utils import logger

def train_model(
    train: List[str],
    test: List[str],
    output: str,
    time_profile: bool = False,
    lazy: bool = False,
    **model_kw,
):
    if not len(train) > 0:
        raise ValueError("At least one training dataset is required")
    
    if len(train) != len(test):
        raise ValueError("Number of training and testing datasets must match")

    # Filter out None values from model parameters
    model_kw = {k: v for k, v in model_kw.items() if v is not None}

    logger.info(f"Training with {len(train)} dataset pairs")

    # Load train/test data splits
    if lazy:
        train_data = [gt.lazy_load(path) for path in train]
        test_data = [gt.lazy_load(path) for path in test]
    else:
        train_data = [gt.eager_load(path) for path in train]
        test_data = [gt.eager_load(path) for path in test]

    # Configure profiling mode
    if time_profile:
        model_kw["eval_every"] = 1
        model_kw["num_epochs"] = 1
        logger.setLevel("DEBUG")

    # Train the model
    model = gt.get_mode(train_data[0]).TopographyModel(**model_kw)
    model = model.fit(*train_data, test_datasets=test_data)

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


def get_study_summary(study_name: str):
    from mutopia.tuning import summary

    trials = summary(study_name)
    trials = trials.sort_values("value", ascending=False, na_position="last")
    sel_cols = ["number", "value", "state"]

    if "user_attrs_model_path" in trials.columns:
        sel_cols += ["user_attrs_model_path"]
    sel_cols += [col for col in trials.columns if col.startswith("params_")]

    trials = trials[sel_cols]
    trials.columns = [col.removeprefix("params_") for col in trials.columns]

    return trials


def annot(
    model: str, 
    dataset: str, 
    output: str, 
    region: Optional[str] = None, 
    threads: int =1, 
    calc_shap: bool = False,
    celltype: Optional[str] = None,
):
    import mutopia.analysis as mu
    model = mu.load_model(model)
    ds = gt.load_dataset(dataset, with_samples=False, with_state=False)
    annotated = model.annot_data(
        ds,
        subset_region=region,
        threads=threads,
        calc_shap=calc_shap,
        source=celltype,
    )
    disk.write_dataset(annotated, output, write_samples=False)


def simulate_from_model(
    model_path: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    scale_num_mutations: float = 1.0,
):
    from mutopia.simulate import simulate_from_model
    import mutopia.analysis as mu

    dataset = disk.load_dataset(dataset_path, with_samples=False)
    model = mu.load_model(model_path)

    resampled = simulate_from_model(
        model,
        dataset,
        seed=seed,
        scale_num_mutations=scale_num_mutations,
    )

    disk.write_dataset(resampled, output_path)
