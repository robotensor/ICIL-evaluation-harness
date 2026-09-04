"""Load and save the committed registry (YAML) and compute its version hash."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from icil_eval.paths import REGISTRY_ROOT
from icil_eval.registry.hashing import hash_files, sha256_file
from icil_eval.registry.schema import (
    POOL_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    PoolEntry,
    ProviderInfo,
    Task,
    Track,
)


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=110)


def _load(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tasks_path(root: Path, provider: str, suite: str) -> Path:
    return root / "tasks" / provider / f"{suite}.yaml"


def provider_path(root: Path, provider: str) -> Path:
    return root / "providers" / f"{provider}.yaml"


def track_path(root: Path, track: str) -> Path:
    return root / "tracks" / f"{track}.yaml"


def pool_path(root: Path, track: str, provider: str) -> Path:
    return root / "pools" / track / f"{provider}.yaml"


def save_tasks(root: Path, provider: str, suite: str, tasks: List[Task]) -> Path:
    tasks = sorted(tasks, key=lambda t: t.task_id)
    payload = {
        "schema_version": TASK_SCHEMA_VERSION,
        "provider": provider,
        "suite": suite,
        "n_tasks": len(tasks),
        "tasks": [t.to_dict() for t in tasks],
    }
    path = tasks_path(root, provider, suite)
    _dump(payload, path)
    return path


def save_provider(root: Path, info: ProviderInfo) -> Path:
    path = provider_path(root, info.name)
    _dump(info.to_dict(), path)
    return path


def save_track(root: Path, track: Track) -> Path:
    path = track_path(root, track.name)
    _dump(track.to_dict(), path)
    return path


def save_pool(root: Path, track: str, provider: str, entries: List[PoolEntry]) -> Path:
    entries = sorted(entries, key=lambda e: e.task_id)
    payload = {
        "schema_version": POOL_SCHEMA_VERSION,
        "track": track,
        "provider": provider,
        "n_query_tasks": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
    path = pool_path(root, track, provider)
    _dump(payload, path)
    return path


@dataclass
class Registry:
    root: Path
    tasks: Dict[str, Task] = field(default_factory=dict)
    providers: Dict[str, ProviderInfo] = field(default_factory=dict)
    tracks: Dict[str, Track] = field(default_factory=dict)
    pools: Dict[Tuple[str, str], Dict[str, PoolEntry]] = field(default_factory=dict)

    def pool(self, track: str, task_id: str) -> Optional[PoolEntry]:
        provider = self.tasks[task_id].provider
        return self.pools.get((track, provider), {}).get(task_id)

    def tasks_of(self, provider: str, suite: Optional[str] = None) -> List[Task]:
        return sorted(
            (
                t
                for t in self.tasks.values()
                if t.provider == provider and (suite is None or t.suite == suite)
            ),
            key=lambda t: t.task_id,
        )

    def registry_hash(self) -> str:
        """Content hash over every YAML file under the registry root."""
        entries = []
        for path in sorted(self.root.rglob("*.yaml")):
            entries.append((path.relative_to(self.root).as_posix(), sha256_file(path)))
        return hash_files(entries)


def load_registry(root: Optional[Path] = None) -> Registry:
    root = Path(root) if root is not None else REGISTRY_ROOT
    reg = Registry(root=root)
    for path in sorted((root / "tasks").rglob("*.yaml")) if (root / "tasks").exists() else []:
        payload = _load(path)
        for d in payload.get("tasks", []):
            task = Task.from_dict(d)
            if task.task_id in reg.tasks:
                raise ValueError(f"duplicate task id {task.task_id} in {path}")
            reg.tasks[task.task_id] = task
    for path in (
        sorted((root / "providers").glob("*.yaml")) if (root / "providers").exists() else []
    ):
        info = ProviderInfo.from_dict(_load(path))
        reg.providers[info.name] = info
    for path in sorted((root / "tracks").glob("*.yaml")) if (root / "tracks").exists() else []:
        track = Track.from_dict(_load(path))
        reg.tracks[track.name] = track
    for path in sorted((root / "pools").rglob("*.yaml")) if (root / "pools").exists() else []:
        payload = _load(path)
        key = (payload["track"], payload["provider"])
        reg.pools[key] = {e["task_id"]: PoolEntry.from_dict(e) for e in payload.get("entries", [])}
    return reg
