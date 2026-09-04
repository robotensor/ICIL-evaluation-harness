"""Container-side modules must import and behave under Python 3.8 with only numpy + PyYAML.

Run under 3.8 in CI via ``uv run --python 3.8 ... pytest -m container``.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.container

CONTAINER_SIDE_MODULES = [
    "icil_eval",
    "icil_eval.paths",
    "icil_eval.cli",
    "icil_eval.registry.schema",
    "icil_eval.registry.hashing",
    "icil_eval.registry.relations",
    "icil_eval.registry.io",
    "icil_eval.registry.validate",
    "icil_eval.context.types",
    "icil_eval.context.sampler",
    "icil_eval.context.transforms",
    "icil_eval.providers.libero.bddl",
]


@pytest.mark.parametrize("module", CONTAINER_SIDE_MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_python_version_floor() -> None:
    assert sys.version_info >= (3, 8)


def test_cli_help_runs() -> None:
    from icil_eval.cli import main

    assert main([]) == 0
