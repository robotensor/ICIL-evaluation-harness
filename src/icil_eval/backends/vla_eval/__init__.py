"""vla-eval backend: ICIL benchmark wrapper (runs inside simulator containers, Python 3.8-safe),
ICIL model-server base (runs in the policy environment), run-config generation and ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from icil_eval.backends.base import EnvBackend


class VlaEvalBackend(EnvBackend):
    name = "vla_eval"

    def benchmark_import_string(self, provider: str) -> str:
        return "icil_eval.backends.vla_eval.benchmark:ICILBenchmark"

    def build_run_config(self, **kwargs: Any) -> Dict[str, Any]:
        from icil_eval.backends.vla_eval.config import build_run_config

        return build_run_config(**kwargs)

    def ingest(self, results_dir: Path, registry: Any, server_logs: List[Path]) -> Dict[str, Any]:
        from icil_eval.scoring.ingest import ingest

        return ingest(results_dir, registry, server_logs=server_logs)
