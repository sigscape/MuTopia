"""
Inference-path regression tests.

These pin the behavior of a frozen model + frozen fixture: outputs that change
in shape, dtype, sign, or magnitude flag a regression. Numerical tolerances are
loose so legitimate BLAS/threading drift doesn't flake the suite.
"""

from __future__ import annotations

import numpy as np
import pytest


EXPECTED_ANNOT_VARS = {
    "contributions",
    "Spectra/spectra",
    "Spectra/interactions",
    "Spectra/shared_effects",
    "component_distributions",
    "component_distributions_locus",
    "predicted_marginal",
    "predicted_marginal_locus",
    "empirical_marginal",
    "empirical_marginal_locus",
}


def test_model_loads(trained_model):
    assert type(trained_model).__name__ == "SBSModel"
    assert trained_model.num_components == 15


def test_train_fixture_shape(train_dataset):
    sizes = dict(train_dataset.sizes)
    assert sizes["sample"] == 10
    assert sizes["context"] == 96
    assert sizes["configuration"] == 2
    assert sizes["locus"] > 0


def test_test_fixture_shape(test_dataset):
    sizes = dict(test_dataset.sizes)
    assert sizes["sample"] == 10
    assert sizes["locus"] > 0


def test_annot_data_produces_expected_vars(trained_model, train_dataset):
    annotated = trained_model.annot_data(train_dataset, threads=1, calc_shap=False)
    missing = EXPECTED_ANNOT_VARS - set(annotated.data_vars)
    assert not missing, f"annot_data missing vars: {missing}"


def test_contributions_are_finite(trained_model, train_dataset):
    annotated = trained_model.annot_data(train_dataset, threads=1, calc_shap=False)
    contrib = annotated["contributions"].values
    assert np.all(np.isfinite(contrib)), "non-finite values in contributions"
    assert (contrib >= 0).all(), "contributions should be non-negative"


def test_contributions_shape_matches_components(trained_model, train_dataset):
    annotated = trained_model.annot_data(train_dataset, threads=1, calc_shap=False)
    contrib = annotated["contributions"]
    assert "sample" in contrib.dims
    assert "component" in contrib.dims
    assert contrib.sizes["component"] == trained_model.num_components


def test_predicted_marginal_finite(trained_model, train_dataset):
    annotated = trained_model.annot_data(train_dataset, threads=1, calc_shap=False)
    pred = annotated["predicted_marginal"].values
    assert np.all(np.isfinite(pred))


def test_component_distributions_finite_and_nonneg(trained_model, train_dataset):
    annotated = trained_model.annot_data(train_dataset, threads=1, calc_shap=False)
    cd = annotated["component_distributions"].values
    finite = cd[np.isfinite(cd)]
    assert finite.size > 0
    assert (finite >= 0).all()


def test_model_save_load_roundtrip(trained_model, tmp_path):
    """Saving then loading should produce a model that pickles to the same state."""
    import mutopia.analysis as mu

    out = tmp_path / "roundtrip.pkl"
    trained_model.save(str(out))
    reloaded = mu.load_model(str(out))
    assert reloaded.num_components == trained_model.num_components
    assert type(reloaded).__name__ == type(trained_model).__name__


def test_save_load_predictions_match(trained_model, train_dataset, tmp_path):
    """A reloaded model must produce numerically equivalent predictions."""
    import mutopia.analysis as mu

    out = tmp_path / "rt.pkl"
    trained_model.save(str(out))
    reloaded = mu.load_model(str(out))

    a = trained_model.annot_data(train_dataset, threads=1, calc_shap=False)
    b = reloaded.annot_data(train_dataset, threads=1, calc_shap=False)
    np.testing.assert_allclose(
        a["contributions"].values, b["contributions"].values, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        a["predicted_marginal"].values,
        b["predicted_marginal"].values,
        rtol=1e-5,
        atol=1e-7,
    )


def test_callback_dropped_from_pickle(trained_model, tmp_path):
    """__getstate__ must drop the training callback so closures over training data
    don't get pickled (regression for c73656e)."""
    import mutopia.analysis as mu

    trained_model.callback = lambda *a, **kw: None  # simulate a trained-with-callback model
    out = tmp_path / "no_callback.pkl"
    trained_model.save(str(out))
    reloaded = mu.load_model(str(out))
    assert reloaded.callback is None
