"""
Training-path tests against the full-scale Liver-HCC gtensor from Zenodo.

These are slow (each fits a real model end-to-end) and gated behind
--runslow so they don't bloat the default suite or CI. To run:

    pytest tests/test_training.py --runslow

The Zenodo asset (~hundreds of MB) is downloaded once into
tests/fixtures/zenodo/ and cached. Use these to sanity-check optimization
changes before tagging a release.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _make_model(train_dataset, **overrides):
    import mutopia.analysis as mu

    ModelCls = mu.make_model_cls(train_dataset)
    kwargs = dict(
        num_components=5,
        seed=0,
        threads=1,
        eval_every=1,
    )
    kwargs.update(overrides)
    return ModelCls(**kwargs)


def test_batch_training_completes_without_nan(zenodo_liver_split):
    """Full-batch training (no subsampling) on the production-scale gtensor."""
    train, test = zenodo_liver_split

    model = _make_model(
        train,
        locus_subsample=None,
        batch_subsample=None,
        num_epochs=20,
    )
    model.fit(train, test)

    assert model.test_scores_, "test_scores_ should be populated"
    assert all(math.isfinite(s) for s in model.test_scores_), (
        f"non-finite test scores: {model.test_scores_}"
    )


def test_batch_training_test_scores_nondecreasing(zenodo_liver_split):
    """In full-batch mode on real data, the test score should be non-decreasing
    modulo small numerical noise. A real decrease flags an optimizer regression."""
    train, test = zenodo_liver_split

    model = _make_model(
        train,
        locus_subsample=None,
        batch_subsample=None,
        num_epochs=20,
    )
    model.fit(train, test)

    scores = np.asarray(model.test_scores_)
    diffs = np.diff(scores)
    # Allow tiny float noise; any meaningful decrease is a regression.
    assert (diffs >= -1e-3).all(), (
        f"test scores decreased between epochs: {scores.tolist()}"
    )


def test_svi_training_completes_without_nan(zenodo_liver_split):
    """SVI training (with subsampling) at production scale."""
    train, test = zenodo_liver_split

    model = _make_model(
        train,
        locus_subsample=1 / 8,
        num_epochs=20,
    )
    model.fit(train, test)

    assert model.test_scores_
    assert all(math.isfinite(s) for s in model.test_scores_), (
        f"non-finite test scores: {model.test_scores_}"
    )


def test_trained_model_save_load_roundtrip(zenodo_liver_split, tmp_path):
    """A freshly trained model must save and reload to bit-equivalent predictions."""
    import mutopia.analysis as mu

    train, test = zenodo_liver_split

    model = _make_model(
        train,
        locus_subsample=1 / 8,
        num_epochs=5,
    )
    model.fit(train, test)

    out = tmp_path / "fresh.pkl"
    model.save(str(out))
    reloaded = mu.load_model(str(out))

    a = model.annot_data(train, threads=1, calc_shap=False)
    b = reloaded.annot_data(train, threads=1, calc_shap=False)
    np.testing.assert_allclose(
        a["contributions"].values, b["contributions"].values, rtol=1e-5, atol=1e-7
    )


def test_init_components_with_cosmic_names(zenodo_liver_split):
    """The tutorial-recommended init_components path should fit without crashing."""
    train, test = zenodo_liver_split

    model = _make_model(
        train,
        num_components=2,
        init_components=["SBS1", "SBS3"],
        locus_subsample=1 / 8,
        num_epochs=5,
    )
    model.fit(train, test)
    assert model.test_scores_
