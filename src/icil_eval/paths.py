"""Filesystem locations used across the harness.

Committed data (registry, schemas) lives inside the package so it is available from an installed
wheel and from an editable checkout alike. Large, machine-local artefacts (raw demonstrations,
converted datasets, checkpoints, results) live under a cache root that defaults to
``~/.cache/icil-eval`` and can be relocated with ``ICIL_EVAL_CACHE``.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data"
REGISTRY_ROOT = DATA_ROOT / "registry"
SCHEMAS_ROOT = DATA_ROOT / "schemas"


def cache_root() -> Path:
    """Return the machine-local cache root (created on demand)."""
    root = Path(os.environ.get("ICIL_EVAL_CACHE", "~/.cache/icil-eval")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def raw_demos_root(provider: str) -> Path:
    """Where a provider's raw source demonstrations are downloaded."""
    return cache_root() / "raw" / provider


def datasets_root() -> Path:
    """Where converted LeRobotDataset v3 demonstration datasets are stored."""
    return cache_root() / "datasets"


def checkpoints_root() -> Path:
    """Where policy checkpoints are downloaded."""
    return cache_root() / "checkpoints"
