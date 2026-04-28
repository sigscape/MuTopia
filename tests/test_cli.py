"""
CLI smoke tests.

These shell out to the installed entry points and check exit codes.
They verify packaging / command registration, not numerical behavior.
"""

from __future__ import annotations

import subprocess


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def test_gtensor_help():
    r = _run(["gtensor", "--help"])
    assert r.returncode == 0, r.stderr
    assert "compose" in r.stdout


def test_topo_model_help():
    r = _run(["topo-model", "--help"])
    assert r.returncode == 0, r.stderr
    assert "train" in r.stdout
    assert "study" in r.stdout


def test_topo_model_score_help():
    r = _run(["topo-model", "score", "--help"])
    assert r.returncode == 0, r.stderr


def test_mutopia_sbs_help():
    r = _run(["mutopia-sbs", "--help"])
    assert r.returncode == 0, r.stderr


def test_gtensor_info(train_nc_path):
    r = _run(["gtensor", "info", str(train_nc_path)])
    assert r.returncode == 0, r.stderr


def test_gtensor_feature_ls(train_nc_path):
    r = _run(["gtensor", "feature", "ls", str(train_nc_path)])
    assert r.returncode == 0, r.stderr
    assert "H3K27ac" in r.stdout
