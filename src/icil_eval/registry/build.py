"""Build registry entries (tasks, provider info, materialised pools) for one provider."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from icil_eval.paths import REGISTRY_ROOT
from icil_eval.registry.io import (
    Registry,
    load_registry,
    save_pool,
    save_provider,
    save_tasks,
)
from icil_eval.registry.relations import compute_capabilities, relation_pool, wrong_task_pool
from icil_eval.registry.schema import PoolEntry, Task, Track


def build_pools(track: Track, tasks: List[Task]) -> List[PoolEntry]:
    """Materialise context and wrong-task pools of one track for one provider's tasks."""
    entries: List[PoolEntry] = []
    for q in sorted(tasks, key=lambda t: t.task_id):
        if track.relation not in q.capabilities.relations:
            continue
        context = [t.task_id for t in relation_pool(q, track.relation, tasks)]
        if not context:
            continue
        wrong = [list(pair) for pair in wrong_task_pool(q, tasks, track.wrong_task_rule)]
        flags: Dict[str, Any] = {}
        if not wrong:
            flags["no_wrong_task"] = True
        entries.append(PoolEntry(task_id=q.task_id, context=context, wrong=wrong, flags=flags))
    return entries


def build_provider(
    name: str, root: Optional[Path] = None, provider_options: Optional[Dict[str, Any]] = None
) -> int:
    from icil_eval.providers import get_provider

    root = Path(root) if root is not None else REGISTRY_ROOT
    provider = get_provider(name, **(provider_options or {}))
    tasks = provider.enumerate_tasks()
    if not tasks:
        print(f"provider '{name}' produced no tasks", file=sys.stderr)
        return 1
    compute_capabilities(tasks)

    suites = sorted({t.suite for t in tasks})
    for suite in suites:
        path = save_tasks(root, name, suite, [t for t in tasks if t.suite == suite])
        print(f"wrote {path.relative_to(root)}")

    registry: Registry = load_registry(root)
    capability_matrix: Dict[str, Dict[str, int]] = {}
    for track in registry.tracks.values():
        entries = build_pools(track, tasks)
        if entries:
            path = save_pool(root, track.name, name, entries)
            print(f"wrote {path.relative_to(root)} ({len(entries)} query tasks)")
            capability_matrix[track.name] = {track.relation: len(entries)}

    info = provider.info(tasks)
    info.capability_matrix = capability_matrix
    path = save_provider(root, info)
    print(f"wrote {path.relative_to(root)}")
    print(f"registry hash: {load_registry(root).registry_hash()}")
    return 0
