"""
Shared pytest fixtures.

Test data lives outside the repo (in a GitHub release) so the tree stays small.
On first use, the conftest downloads each fixture into tests/fixtures/ and caches
it. To regenerate the fixtures from scratch, see tests/build_chr22_fixture.py.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (training, convergence).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests that take >30s (deselect with default pytest run)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Update both the tag and the asset list when fixtures are rebuilt.
FIXTURE_RELEASE_TAG = "test-fixtures"
FIXTURE_BASE_URL = (
    f"https://github.com/sigscape/MuTopia/releases/download/{FIXTURE_RELEASE_TAG}"
)
FIXTURE_FILES = (
    "liver.chr22.train.nc",
    "liver.chr1_50mb.test.nc",
    "liver.trained_model.pkl",
)


def _ensure_fixture(name: str) -> Path:
    path = FIXTURE_DIR / name
    if path.exists():
        return path

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{FIXTURE_BASE_URL}/{name}"
    print(f"[fixtures] downloading {name} from {url}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        if path.exists():
            path.unlink()
        pytest.skip(
            f"Could not download fixture {name} from {url} ({e}). "
            f"Either run tests/build_chr22_fixture.py locally or upload "
            f"the fixtures to the {FIXTURE_RELEASE_TAG} release."
        )
    return path


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def train_nc_path() -> Path:
    return _ensure_fixture("liver.chr22.train.nc")


@pytest.fixture(scope="session")
def test_nc_path() -> Path:
    return _ensure_fixture("liver.chr1_50mb.test.nc")


@pytest.fixture(scope="session")
def model_pkl_path() -> Path:
    return _ensure_fixture("liver.trained_model.pkl")


@pytest.fixture(scope="session")
def trained_model(model_pkl_path: Path):
    import mutopia.analysis as mu

    return mu.load_model(str(model_pkl_path))


@pytest.fixture(scope="session")
def train_dataset(train_nc_path: Path):
    import mutopia.analysis as mu

    return mu.gt.eager_load(str(train_nc_path))


@pytest.fixture(scope="session")
def test_dataset(test_nc_path: Path):
    import mutopia.analysis as mu

    return mu.gt.eager_load(str(test_nc_path))


# ---------------------------------------------------------------------------
# Full-scale (Zenodo) fixtures — only requested by slow-tier tests.
# ---------------------------------------------------------------------------

ZENODO_RECORD = "18803136"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
ZENODO_CACHE_DIR = FIXTURE_DIR / "zenodo"
ZENODO_TUMOR_TYPE = "Liver-HCC"


def _ensure_zenodo_asset(name: str) -> Path:
    path = ZENODO_CACHE_DIR / name
    if path.exists():
        return path

    ZENODO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{ZENODO_BASE_URL}/{name}"
    print(f"[fixtures] downloading {name} from Zenodo (large; first run only)")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        if path.exists():
            path.unlink()
        pytest.skip(
            f"Could not download Zenodo asset {name} from {url} ({e}). "
            f"Slow-tier tests require network access to zenodo.org."
        )
    return path


@pytest.fixture(scope="session")
def zenodo_liver_path() -> Path:
    return _ensure_zenodo_asset(f"{ZENODO_TUMOR_TYPE}.nc")


@pytest.fixture(scope="session")
def zenodo_liver_split(zenodo_liver_path: Path):
    """Train/test split of the full Liver-HCC gtensor by holding out chr1."""
    from mutopia.gtensor import lazy_train_test_load

    return lazy_train_test_load(str(zenodo_liver_path), "chr1")
