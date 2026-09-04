"""Per-task supported K for a policy, from its context budget and the demonstration pool lengths.

Writing this to a JSON file lets the benchmark wrapper skip conditions the policy cannot run
(``unsupported``) instead of spending simulator time on hold actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from icil_eval.policy.protocol import PolicySpec, UnsupportedContext, resolve_k_max
from icil_eval.registry.io import Registry


def k_max_by_task(
    spec: PolicySpec,
    registry: Registry,
    lengths_for_task,
    k_sweep: Sequence[int] = (0, 1, 2, 4, 8),
    task_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    for task_id in task_ids or registry.tasks:
        task = registry.tasks[task_id]
        try:
            lengths = lengths_for_task(task)
        except Exception:  # noqa: BLE001 - demonstrations not available locally
            continue
        try:
            out[task_id] = resolve_k_max(spec, k_sweep, lengths)
        except UnsupportedContext:
            out[task_id] = None
    return out


def write_capabilities(
    path: Path,
    spec: PolicySpec,
    kmax: Dict[str, Optional[int]],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = {
        "policy": spec.name,
        "context_budget": spec.context_budget,
        "k_min": spec.k_min,
        "k_max": spec.k_max,
        "k_max_by_task": {k: v for k, v in sorted(kmax.items()) if v is not None},
        "unsupported_tasks": sorted(k for k, v in kmax.items() if v is None),
    }
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path
