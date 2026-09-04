"""The ``TaskProvider`` contract: how an existing benchmark joins the global registry."""

from __future__ import annotations

import abc
import importlib
from pathlib import Path
from typing import Any, Dict, List

from icil_eval.registry.schema import ProviderInfo, Task

# Lazily imported provider classes, keyed by provider name.
PROVIDER_CLASSES: Dict[str, str] = {
    "libero": "icil_eval.providers.libero.provider:LiberoProvider",
}


class TaskProvider(abc.ABC):
    """Normalise one source benchmark into unified tasks, demonstrations and a backend binding.

    Implementations must not import simulators at module level; do so inside methods.
    """

    name: str = ""

    @abc.abstractmethod
    def version(self) -> Dict[str, str]:
        """Pinned versions of the source benchmark (commit, simulator versions, ...)."""

    @abc.abstractmethod
    def enumerate_tasks(self) -> List[Task]:
        """All tasks of this provider in the unified schema (sorted by task_id)."""

    @abc.abstractmethod
    def info(self, tasks: List[Task]) -> ProviderInfo:
        """Provider metadata written to ``registry/providers/<name>.yaml``."""

    @abc.abstractmethod
    def backend_binding(self, suite: str) -> Dict[str, Any]:
        """How to roll out tasks of ``suite``: ``{backend, inner, params}``."""

    def export_demos(self, task: Task, dest: Path) -> Path:
        """Export the task's demonstration pool to a LeRobotDataset v3 at ``dest``."""
        raise NotImplementedError(f"{self.name} does not implement demonstration export")


def get_provider(name: str, **options: Any) -> TaskProvider:
    if name not in PROVIDER_CLASSES:
        raise KeyError(f"unknown provider '{name}'; known: {sorted(PROVIDER_CLASSES)}")
    module_name, class_name = PROVIDER_CLASSES[name].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**options)
