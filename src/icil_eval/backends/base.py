"""The environment/rollout backend contract.

The ICIL standard (registry, context, policy protocol, metrics) is backend-agnostic. A backend
supplies four things so that a track can be executed and scored:

1. a **benchmark adapter** that expands registry tasks into ``task x condition`` episodes and
   exposes the flat context-by-reference fields at episode start;
2. a **policy bridge** that hosts an :class:`icil_eval.policy.ICILPolicy` behind the backend's
   policy interface (recomputing the context spec from those fields);
3. a **run configuration** generator for the backend's orchestrator;
4. a **results reader** that yields :class:`icil_eval.scoring.ingest.EpisodeRow` objects.

``vla_eval`` is the first implementation. Others (lerobot-eval, XPolicyLab env clients, an
in-process gym loop) register here without touching the standard.
"""

from __future__ import annotations

import abc
import importlib
from pathlib import Path
from typing import Any, Dict, List

BACKENDS: Dict[str, str] = {
    "vla_eval": "icil_eval.backends.vla_eval:VlaEvalBackend",
}


class EnvBackend(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def benchmark_import_string(self, provider: str) -> str:
        """Import string of the ICIL benchmark adapter for this backend (``module:Class``)."""

    @abc.abstractmethod
    def build_run_config(self, **kwargs: Any) -> Dict[str, Any]:
        """Backend-native run configuration for one track (see S4/S5 for the fields recorded)."""

    @abc.abstractmethod
    def ingest(self, results_dir: Path, registry: Any, server_logs: List[Path]) -> Dict[str, Any]:
        """Read backend results (+ policy-server logs) into ``{"rows": [EpisodeRow], ...}``."""


def get_backend(name: str) -> EnvBackend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend '{name}'; known: {sorted(BACKENDS)}")
    module_name, class_name = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)()
