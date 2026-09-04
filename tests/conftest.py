"""Shared pytest configuration.

Simulator packages are deliberately NOT stubbed: the absence of a simulator is a code path the
harness must handle, and tests marked ``sim``/``gpu`` are skipped unless the resources exist.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Isolate tests from the user's machine-local cache (Hub API responses, datasets, checkpoints).
os.environ.setdefault("ICIL_EVAL_CACHE", tempfile.mkdtemp(prefix="icil-eval-test-cache-"))


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def pytest_collection_modifyitems(config, items):
    skip_sim = pytest.mark.skip(reason="simulator (libero/robosuite) not installed")
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    have_sim = _has_module("libero") and _has_module("robosuite")
    have_gpu = os.environ.get("ICIL_EVAL_TEST_GPU") == "1"
    for item in items:
        if "sim" in item.keywords and not have_sim:
            item.add_marker(skip_sim)
        if "gpu" in item.keywords and not have_gpu:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
