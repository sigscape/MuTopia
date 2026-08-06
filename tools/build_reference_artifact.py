#!/usr/bin/env python
"""Build the bundled topography UMAP reference artifact.

Freezes a published component UMAP into the pair of files that
``mutopia.analysis.topography_umap`` loads:

* ``pancan_topography_umap.npz``       — feature matrix + layout + metadata (~11 MB)
* ``pancan_topography_umap.coords.tsv`` — layout only (~15 kB, committed)

The npz is distributed as a GitHub release asset rather than committed; see
DEVELOPING.md.

Inputs
------
--collected-data
    Pickled ``xarray.Dataset`` with ``normalized_mutation_rate``
    (component x locus) and ``Regions/{chrom,start,length}``.
--annotations
    TSV indexed by ``component`` with ``cluster_id``, ``class``, ``tumor_type``
    and the canonical ``UMAP1``/``UMAP2`` columns.

The published coordinates are preserved verbatim rather than re-derived.  They
are not bit-reproducible under current umap-learn, but that does not matter:
``umap.transform`` places new points by optimizing against ``embedding_`` held
fixed, so any fixed layout of the reference points defines a valid projection,
and preserving it keeps the published figures authoritative.

Requires the ``umap`` extra::

    pip install -e '.[umap]'
    python tools/build_reference_artifact.py \
        --collected-data data/pancan/collected_data.4.pkl \
        --annotations   data/pancan/02.04.final_annotations.tsv
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from mutopia.analysis.topography_umap import (
    REFERENCE_ARTIFACT,
    REFERENCE_COORDINATES,
    TopographyUMAP,
    load_reference_coordinates,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Display names for the curated locus clusters.  Kept here rather than imported
#: from the analysis tree so the artifact is regenerable from a plain checkout.
TOPO_NAMES = {
    "4": "Topo-5mc",
    "7": "Topo-Rep",
    "1": "Topo-RepAPOBEC",
    "3": "Topo-Tr",
    "2": "Topo-CanAPOBEC",
    "6": "Topo-StressD",
    "0": "Topo-CanD II",
    "8": "Topo-CanD I",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collected-data", required=True,
                   help="pickled Dataset with normalized_mutation_rate")
    p.add_argument("--annotations", required=True,
                   help="TSV with component, cluster_id, class, tumor_type, UMAP1, UMAP2")
    p.add_argument("--artifact", default=REFERENCE_ARTIFACT,
                   help=f"output npz (default: {REFERENCE_ARTIFACT})")
    p.add_argument("--coordinates", default=REFERENCE_COORDINATES,
                   help=f"output coordinates TSV (default: {REFERENCE_COORDINATES})")
    p.add_argument("--dtype", choices=["float16", "float32"], default=None,
                   help="storage dtype for the feature matrix (default: float16 "
                        "if it is validated to leave projections unchanged)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import numba
    import sklearn
    import umap

    with open(args.collected_data, "rb") as f:
        d = pickle.load(f)
    ann = pd.read_csv(args.annotations, sep="\t", dtype={"cluster_id": str})
    ann = ann.set_index("component")

    missing = [c for c in ann.index if c not in set(d["component"].values)]
    if missing:
        raise SystemExit(
            f"{len(missing)} annotated components absent from the source data: {missing[:5]}"
        )

    order = ann.index.tolist()
    rates = d["normalized_mutation_rate"].sel(component=order)
    chrom = np.asarray(d["Regions/chrom"].values, dtype=str)
    start = np.asarray(d["Regions/start"].values, dtype=np.int64)
    length = np.asarray(d["Regions/length"].values, dtype=np.float64)

    n_unique = len({*zip(chrom, start)})
    print(f"reference axis: {len(chrom)} loci on {sorted(set(chrom))}, "
          f"{n_unique} unique (chrom, start)")
    if n_unique != len(chrom):
        print(f"  {len(chrom) - n_unique} loci are mesoscale splits sharing coordinates; "
              "the axis is matched positionally, not by coordinate")

    X = TopographyUMAP._normalize(np.asarray(rates.values, dtype=np.float64))
    published = ann[["UMAP1", "UMAP2"]].values.astype(np.float32)
    print(f"X: {X.shape}; published layout: {published.shape}")

    metadata = pd.DataFrame(index=ann.index)
    metadata["tumor_type"] = ann["tumor_type"].astype(str)
    metadata["class"] = ann["class"].astype(str)
    metadata["cluster_id"] = ann["cluster_id"].astype(str)
    metadata["cluster_name"] = metadata["cluster_id"].map(TOPO_NAMES).fillna("")
    unnamed = sorted(set(metadata.loc[metadata.cluster_name == "", "cluster_id"]))
    if unnamed:
        print(f"  note: cluster ids with no display name: {unnamed}")

    est = TopographyUMAP().fit(
        (X, np.array(order)), metadata=metadata,
        bins=(chrom, start, length), embedding=published,
    )
    print(f"fitted; LOF flag threshold {est.outlier_threshold_:.3f}, "
          f"lowest cohort spread {min(est.cohort_spread_.values()):.4f}")

    provenance = dict(
        built=datetime.datetime.now().isoformat(timespec="seconds"),
        collected_data=os.path.basename(args.collected_data),
        collected_data_sha256=sha256(args.collected_data),
        annotations=os.path.basename(args.annotations),
        annotations_sha256=sha256(args.annotations),
        n_components=len(order),
        n_loci=len(chrom),
        n_unique_bins=n_unique,
        chroms=sorted(set(chrom)),
        axis_matching="positional (mesoscale splits share genomic coordinates)",
        embedding="published coordinates, preserved verbatim (not re-derived)",
        umap_learn=umap.__version__, numba=numba.__version__,
        sklearn=sklearn.__version__, numpy=np.__version__,
        python=sys.version.split()[0], git_rev=git_rev(),
    )

    dtype = np.dtype(args.dtype) if args.dtype else validate_float16(
        est, X, published, metadata, (chrom, start, length), order
    )

    path = est.save(args.artifact, provenance=provenance, dtype=dtype)
    print(f"\nwrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, dtype={dtype})")
    coords = est.write_coordinates(args.coordinates)
    print(f"wrote {coords} ({os.path.getsize(coords) / 1e3:.1f} kB)")

    back = TopographyUMAP.load(path, verify=False)
    assert np.array_equal(back.components_, est.components_)
    assert np.allclose(back.embedding_, published)
    assert np.array_equal(back.bin_start_, start)
    fixture = load_reference_coordinates(coords)
    assert list(fixture.index) == order
    assert np.allclose(fixture[["UMAP1", "UMAP2"]].values, published)
    print(f"reload OK: {len(back.components_)} components, "
          f"{back.metadata_['cluster_id'].nunique()} clusters, "
          f"{back.metadata_['tumor_type'].nunique()} tumor types")
    print("\nNext: upload the npz as a release asset and bump "
          "ARTIFACT_RELEASE_TAG if needed (see DEVELOPING.md).")
    return 0


def validate_float16(est, X, published, metadata, bins, order):
    """Pick a storage dtype, checking float16 does not move any projection."""
    from sklearn.metrics.pairwise import cosine_distances

    print("\nvalidating float16 storage:")
    X16 = X.astype(np.float16).astype(np.float32)
    rel = np.abs(X16 - X) / np.maximum(np.abs(X), 1e-12)
    subnormal = int((np.abs(X.astype(np.float16)) < 6.1e-5).sum())
    print(f"  round-trip relative error: median={np.median(rel):.3g} max={rel.max():.3g}")
    print(f"  float16 subnormals: {subnormal} of {X.size}")
    print(f"  cosine-distance shift: max="
          f"{np.abs(cosine_distances(X) - cosine_distances(X16)).max():.3g}")

    est16 = TopographyUMAP().fit(
        (X16, np.array(order)), metadata=metadata, bins=bins, embedding=published
    )
    probe = (X.astype(np.float64) * (1 + 1e-6)).astype(np.float32)
    p32 = est.transform((probe, np.array(order)))[["UMAP1", "UMAP2"]].values
    p16 = est16.transform((probe.astype(np.float16).astype(np.float32),
                           np.array(order)))[["UMAP1", "UMAP2"]].values
    shift = np.linalg.norm(p32 - p16, axis=1)
    span = float(np.linalg.norm(published.max(0) - published.min(0)))
    print(f"  projection shift: median={np.median(shift):.4f} max={shift.max():.4f} "
          f"({100 * np.median(shift) / span:.2f}% of layout diagonal)")

    if shift.max() > 0.05 * span:
        print("  -> float16 moves projections too much; storing float32")
        return np.dtype(np.float32)
    return np.dtype(np.float16)


if __name__ == "__main__":
    raise SystemExit(main())
