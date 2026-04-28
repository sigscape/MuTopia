"""
One-shot script to produce the small fixture artifacts used by the test suite.

Slices the full Liver tutorial gtensor down to chr22 (train) + chr1 (test),
keeps a small sample subset, and copies the existing tutorial-trained model
as the inference baseline. Outputs land in tests/fixtures/ and are intended
to be uploaded to a GitHub release; CI downloads them via tests/conftest.py.

Run from repo root:

    python tests/build_chr22_fixture.py

Re-run only when the input gtensors or the schema changes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mutopia.analysis as mu

REPO = Path(__file__).resolve().parents[1]
TUTORIAL = REPO / "tutorials" / "tutorial_data"
OUT = REPO / "tests" / "fixtures"

SOURCE_TRAIN = TUTORIAL / "Liver.train.nc"
SOURCE_TEST = TUTORIAL / "Liver.test.nc"
SOURCE_MODEL = TUTORIAL / "trained_model.pkl"

TRAIN_REGION = "chr22"
# A 50 Mb slice of chr1 to keep the held-out fixture small (~5 MB instead of ~35 MB).
TEST_REGION = "chr1:0-50000000"
TRAIN_LABEL = "chr22"
TEST_LABEL = "chr1_50mb"
N_SAMPLES = 10


def build(n_samples: int = N_SAMPLES) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for src in (SOURCE_TRAIN, SOURCE_TEST, SOURCE_MODEL):
        if not src.exists():
            raise FileNotFoundError(
                f"Missing source: {src}. Fetch tutorial_data first (see tutorials/)."
            )

    print(f"[build] loading {SOURCE_TRAIN.name}")
    train_full = mu.gt.eager_load(str(SOURCE_TRAIN))

    sample_names = list(train_full.list_samples())[:n_samples]
    print(f"[build] keeping {len(sample_names)} of {len(list(train_full.list_samples()))} samples")

    print(f"[build] slicing train to {TRAIN_REGION}")
    train_chr = mu.gt.slice_regions(train_full, TRAIN_REGION)
    train_chr = mu.gt.slice_samples(train_chr, sample_names)

    train_out = OUT / f"liver.{TRAIN_LABEL}.train.nc"
    print(f"[build] writing {train_out}")
    mu.gt.write_dataset(train_chr, str(train_out), write_samples=True)

    print(f"[build] loading {SOURCE_TEST.name}")
    test_full = mu.gt.eager_load(str(SOURCE_TEST))
    print(f"[build] slicing test to {TEST_REGION}")
    test_chr = mu.gt.slice_regions(test_full, TEST_REGION)
    test_chr = mu.gt.slice_samples(test_chr, sample_names)

    test_out = OUT / f"liver.{TEST_LABEL}.test.nc"
    print(f"[build] writing {test_out}")
    mu.gt.write_dataset(test_chr, str(test_out), write_samples=True)

    model_out = OUT / "liver.trained_model.pkl"
    print(f"[build] copying {SOURCE_MODEL.name} -> {model_out.name}")
    shutil.copy(SOURCE_MODEL, model_out)

    print()
    print("[build] done. fixtures:")
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size / 1e6:.2f} MB)")
    print()
    print("[build] upload these to a release on sigscape/MuTopia tagged")
    print("        e.g. test-fixtures-v1, then update FIXTURE_RELEASE_TAG in")
    print("        tests/conftest.py and the cache key in .github/workflows/tests.yml.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help=f"Number of samples to retain (default: {N_SAMPLES})",
    )
    args = parser.parse_args()
    build(n_samples=args.n_samples)
