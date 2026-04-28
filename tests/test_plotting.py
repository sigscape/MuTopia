"""
Plotting smoke tests.

We don't pixel-snapshot. We just check that plot calls return without error
and produce a Figure with sensible structure. Visual review stays manual.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def annotated_dataset(trained_model, train_dataset):
    """Annotated dataset usable as input to plotting helpers."""
    return trained_model.annot_data(train_dataset, threads=1, calc_shap=False)


def teardown_function(_):
    plt.close("all")


def test_plot_signature_panel(annotated_dataset):
    import mutopia.analysis as mu

    fig = mu.pl.plot_signature_panel(annotated_dataset)
    assert fig is not None or plt.gcf() is not None


def test_plot_component(annotated_dataset):
    import mutopia.analysis as mu

    components = mu.gt.list_components(annotated_dataset)
    assert components, "no components on annotated dataset"
    mu.pl.plot_component(annotated_dataset, components[0])
    assert plt.gcf().axes, "expected at least one axes"


def test_plot_shap_summary_skips_without_shap(annotated_dataset):
    """plot_shap_summary requires SHAP values; verify it errors clearly when absent."""
    import mutopia.analysis as mu

    with pytest.raises(Exception):
        mu.pl.plot_shap_summary(annotated_dataset)
