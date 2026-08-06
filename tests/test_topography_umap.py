"""Tests for the topography UMAP projector.

Split three ways by what each test needs:

* the coordinates fixture — pandas only, no umap-learn, no large artifact;
* the estimator mechanics — umap-learn, but a synthetic reference, so these run
  without the bundled 11 MB artifact;
* ``annot_component_rates`` — the shared liver chr22 model/gTensor fixtures, and
  no umap at all.
"""

from __future__ import annotations

import os
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from mutopia.analysis.topography_umap import (
    REFERENCE_ARTIFACT,
    REFERENCE_COORDINATES,
    TopographyUMAP,
    annot_component_rates,
    load_reference_coordinates,
    load_reference_umap,
)

needs_umap = pytest.mark.skipif(
    find_spec("umap") is None, reason="umap-learn not installed"
)
needs_artifact = pytest.mark.skipif(
    not os.path.exists(REFERENCE_ARTIFACT),
    reason="bundled reference artifact not built",
)


# --------------------------------------------------------------- coordinates


def test_coordinates_fixture_loads_without_umap():
    """The point of the fixture: a plain TSV read, no heavy deps."""
    if not os.path.exists(REFERENCE_COORDINATES):
        pytest.skip("coordinates fixture not built")

    ref = load_reference_coordinates()
    assert ref.index.name == "component"
    assert {"tumor_type", "class", "cluster_id", "cluster_name",
            "UMAP1", "UMAP2"} <= set(ref.columns)
    assert len(ref) > 0
    # cluster_id must stay a string so it keys into TOPO_NAME_MAP
    assert ref["cluster_id"].map(type).eq(str).all()
    assert np.isfinite(ref[["UMAP1", "UMAP2"]].values).all()


def test_coordinates_fixture_is_small():
    if not os.path.exists(REFERENCE_COORDINATES):
        pytest.skip("coordinates fixture not built")
    assert os.path.getsize(REFERENCE_COORDINATES) < 1_000_000


# ---------------------------------------------------------------- estimator


@pytest.fixture(scope="module")
def toy_reference():
    """A small synthetic reference: 3 well-separated topography 'clusters'."""
    rng = np.random.default_rng(0)
    n_bins, per = 200, 12
    peaks = [slice(0, 60), slice(70, 130), slice(140, 200)]
    rows, meta = [], []
    for c, sl in enumerate(peaks):
        for i in range(per):
            row = rng.random(n_bins) * 0.1
            row[sl] += 1.0 + rng.random(sl.stop - sl.start) * 0.2
            rows.append(row)
            meta.append((f"T{c}", f"SBS{c}", str(c), f"Topo-{c}"))
    X = np.array(rows)
    X = X / X.sum(1, keepdims=True) * 100_000
    names = np.array([f"T{m[0]}_c{i}" for i, m in enumerate(meta)])
    metadata = pd.DataFrame(
        meta, columns=["tumor_type", "class", "cluster_id", "cluster_name"],
        index=pd.Index(names, name="component"),
    )
    bins = (np.array(["chr1"] * n_bins), np.arange(n_bins) * 10_000,
            np.full(n_bins, 10_000.0))
    return X.astype(np.float32), names, metadata, bins


@needs_umap
def test_fit_transform_recovers_clusters(toy_reference):
    X, names, metadata, bins = toy_reference
    est = TopographyUMAP(n_neighbors=4).fit((X, names), metadata=metadata, bins=bins)

    assert est.embedding_.shape == (len(names), 2)
    assert est.X_ref_.shape == X.shape

    # Perturb so umap's input-hash short-circuit does not just return embedding_.
    probe = (X.astype(np.float64) * (1 + 1e-6)).astype(np.float32)
    out = est.transform((probe, names))
    assert list(out.index) == list(names)
    assert (out["cluster_id"].values == metadata["cluster_id"].values).mean() > 0.9
    assert not out["is_outlier"].all()


@needs_umap
def test_pinned_embedding_is_used_verbatim(toy_reference):
    """A pinned layout must survive fit -- this is how published coords are kept."""
    X, names, metadata, bins = toy_reference
    pinned = np.arange(2 * len(names), dtype=np.float32).reshape(len(names), 2)
    est = TopographyUMAP(n_neighbors=4).fit(
        (X, names), metadata=metadata, bins=bins, embedding=pinned
    )
    np.testing.assert_allclose(est.embedding_, pinned)
    np.testing.assert_allclose(est.reducer_.embedding_, pinned)


@needs_umap
def test_save_load_round_trip(toy_reference, tmp_path):
    X, names, metadata, bins = toy_reference
    est = TopographyUMAP(n_neighbors=4).fit((X, names), metadata=metadata, bins=bins)
    path = est.save(tmp_path / "ref.npz", provenance={"test": True})

    # Every label column must land as fixed-width unicode, not object.  An
    # object array is stored as a pickle, which load() rejects outright with
    # allow_pickle=False -- and whether astype(str) produces one depends on the
    # installed pandas, so the round trip alone would not catch a regression.
    with np.load(path, allow_pickle=False) as art:
        for field in ("component", "tumor_type", "class", "cluster_id"):
            assert art[field].dtype.kind == "U", f"{field} is {art[field].dtype}"

    back = TopographyUMAP.load(path, verify=False)
    assert list(back.components_) == list(names)
    np.testing.assert_allclose(back.embedding_, est.embedding_)
    np.testing.assert_array_equal(back.bin_start_, bins[1])
    assert back.provenance_ == {"test": True}

    coords = back.write_coordinates(tmp_path / "coords.tsv")
    frame = load_reference_coordinates(coords)
    np.testing.assert_allclose(frame[["UMAP1", "UMAP2"]].values, est.embedding_)


@needs_umap
def test_axis_mismatch_raises_then_aggregates(toy_reference):
    X, names, metadata, bins = toy_reference
    est = TopographyUMAP(n_neighbors=4).fit((X, names), metadata=metadata, bins=bins)

    chrom, start, length = bins
    assert est._axis_matches(chrom, start, length)
    assert not est._axis_matches(chrom[:-1], start[:-1], length[:-1])

    with pytest.raises(ValueError, match="does not match the reference axis"):
        est._regrid(X[:, :-1], chrom[:-1], start[:-1], length[:-1])

    est.set_params(on_grid_mismatch="aggregate")
    with pytest.warns(UserWarning, match="lossy"):
        out, space = est._regrid(X[:, :-1], chrom[:-1], start[:-1], length[:-1])
    assert space == "aggregate"


@needs_umap
def test_degenerate_set_is_flagged(toy_reference):
    """A set of near-identical components must be called out, not silently placed."""
    X, names, metadata, bins = toy_reference
    est = TopographyUMAP(n_neighbors=4).fit((X, names), metadata=metadata, bins=bins)

    rng = np.random.default_rng(1)
    base = X[0].astype(np.float64)
    clones = np.array([base * (1 + rng.normal(0, 1e-4, len(base))) for _ in range(5)])
    clones = est._normalize(clones)

    with pytest.warns(UserWarning, match="mutually more similar"):
        out = est.transform((clones, np.array([f"clone{i}" for i in range(5)])))
    assert out["set_is_degenerate"].all()

    spread_real = TopographyUMAP.set_spread(X)
    spread_clone = TopographyUMAP.set_spread(clones)
    assert spread_clone < spread_real


@needs_umap
def test_kneighbors_shape(toy_reference):
    X, names, metadata, bins = toy_reference
    est = TopographyUMAP(n_neighbors=4).fit((X, names), metadata=metadata, bins=bins)
    kn = est.kneighbors((X[:3], names[:3]), k=4)
    assert len(kn) == 12
    assert {"component", "rank", "ref_component", "cosine_distance"} <= set(kn.columns)
    assert kn.groupby("component")["cosine_distance"].apply(
        lambda s: s.is_monotonic_increasing
    ).all()


@needs_umap
@needs_artifact
def test_bundled_reference_self_consistent():
    ref = load_reference_umap()
    assert len(ref.components_) == len(ref.embedding_)
    assert ref.X_ref_.shape[0] == len(ref.components_)
    coords = load_reference_coordinates()
    np.testing.assert_allclose(
        coords.loc[list(ref.components_), ["UMAP1", "UMAP2"]].values,
        ref.embedding_, atol=1e-5,
    )


# ------------------------------------------------- annot_component_rates


def test_annot_component_rates_on_training_grid(trained_model, train_dataset):
    """Evaluate a trained model on a gTensor and get per-component locus rates.

    Uses the same grid the model was trained on, so this checks the mechanics --
    state rebuild, normalizer lookup, rate computation -- rather than any
    cross-cohort claim.
    """
    out = annot_component_rates(trained_model, train_dataset)

    rates = out["component_distributions_locus"]
    assert "component" in rates.dims and "locus" in rates.dims
    assert rates.sizes["component"] == trained_model.n_components
    assert rates.sizes["locus"] == train_dataset.sizes["locus"]
    assert list(rates["component"].values) == list(trained_model.component_names)

    values = np.asarray(rates.values)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert values.sum() > 0


def test_annot_component_rates_is_deterministic(trained_model, train_dataset):
    a = np.asarray(annot_component_rates(trained_model, train_dataset)
                   ["component_distributions_locus"].values)
    b = np.asarray(annot_component_rates(trained_model, train_dataset)
                   ["component_distributions_locus"].values)
    np.testing.assert_allclose(a, b)


def test_annot_component_rates_components_differ(trained_model, train_dataset):
    """Distinct components should have distinct topographies."""
    if trained_model.n_components < 2:
        pytest.skip("model has a single component")
    from sklearn.metrics.pairwise import cosine_distances

    X = np.asarray(annot_component_rates(trained_model, train_dataset)
                   ["component_distributions_locus"].values, dtype=np.float64)
    X = X / X.sum(1, keepdims=True)
    D = cosine_distances(X)
    assert D[np.triu_indices(len(X), k=1)].max() > 1e-3


# ------------------------------------------------- COSMIC signature matching


def test_match_components_against_database(trained_model):
    """Component spectra match the bundled signature database."""
    import mutopia.analysis as mu

    SBS = mu.modalities.SBS
    matches = SBS.match_components(trained_model, top_k=2)

    assert {"component", "rank", "reference", "cosine_similarity"} <= set(matches.columns)
    assert len(matches) == trained_model.n_components * 2
    assert set(matches["reference"]) <= set(SBS.available_components)
    # cosine similarity of two non-negative spectra is in [0, 1]
    assert matches["cosine_similarity"].between(0, 1).all()
    # ranks must be ordered best-first within each component
    assert matches.groupby("component")["cosine_similarity"].apply(
        lambda s: s.is_monotonic_decreasing
    ).all()


def test_match_components_restricted_to_named_references(trained_model):
    import mutopia.analysis as mu

    names = ["SBS1", "SBS4", "SBS5"]
    matches = mu.modalities.SBS.match_components(trained_model, *names, top_k=1)
    assert set(matches["reference"]) <= set(names)
    assert len(matches) == trained_model.n_components


def test_component_spectra_are_normalized_rates(trained_model):
    """Spectra are probability vectors over the 96 contexts."""
    import numpy as np
    import mutopia.analysis as mu

    spectra = mu.modalities.SBS.component_spectra(trained_model)
    assert spectra.dims == ("component", "context")
    assert spectra.sizes["context"] == 96
    values = np.asarray(spectra.values)
    assert (values >= 0).all()
    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=1e-5)


def test_database_stores_context_normalized_rates():
    """The bundled database has the trinucleotide composition divided out.

    SBS1 is C>T at methylated CpG.  In COSMIC's published convention -- which
    bakes in the genome's trinucleotide composition, which is why those profiles
    are genome-specific -- roughly 87% of SBS1's mass sits on N[C>T]G.  The
    bundled database puts ~98% there, i.e. it stores per-context rates.  Matching
    must therefore compare rate against rate; see `ModeConfig.match_components`.
    """
    import numpy as np
    import mutopia.analysis as mu

    SBS = mu.modalities.SBS
    contexts = SBS.coords["context"][1]
    sbs1 = SBS.load_components("SBS1").sel(component="SBS1")
    values = np.asarray(sbs1.reindex(context=contexts).values, dtype=float)
    values = values / values.sum()

    cpg = [i for i, c in enumerate(contexts) if c[2:5] == "C>T" and c[-1] == "G"]
    assert values[cpg].sum() > 0.95, (
        "SBS1 is not concentrated on N[C>T]G as expected for a rate-convention "
        "database; the matching normalization may need revisiting"
    )
