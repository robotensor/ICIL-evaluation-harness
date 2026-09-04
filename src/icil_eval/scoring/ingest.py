"""Ingest vla-eval results (aggregate JSON) and the ICIL server log into episode rows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from icil_eval.context.types import Condition
from icil_eval.registry.io import Registry
from icil_eval.registry.relations import normalize_language

STATUSES = ("ok", "unsupported", "not_applicable", "error", "missing")


@dataclass
class EpisodeRow:
    task_id: str
    provider: str
    suite: str
    track: str
    condition: str
    k: Optional[int]
    control: str
    episode_idx: int
    status: str = "ok"
    success: Optional[bool] = None
    progress: Optional[float] = None
    steps: Optional[int] = None
    elapsed_s: Optional[float] = None
    eid: Optional[str] = None
    benchmark: Optional[str] = None
    relation: Optional[str] = None
    transform: Optional[str] = None
    language: Optional[str] = None
    context_refs: List[List[Any]] = field(default_factory=list)
    context_hash: Optional[str] = None
    context_task_id: Optional[str] = None
    k_max: Optional[int] = None
    adaptation_latency_s: Optional[float] = None
    control_latency_s: Optional[float] = None
    min_context_init_l2: Optional[float] = None
    query_exposure: str = "undisclosed"
    context_exposure: str = "undisclosed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def load_server_log(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    """Server JSONL rows keyed by vla-eval episode id (``eid``)."""
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("episode_id"):
                    rows[row["episode_id"]] = row
    return rows


def load_server_meta(paths: Iterable[Path]) -> Optional[Dict[str, Any]]:
    for path in paths:
        meta = Path(path).with_suffix(".meta.json")
        if meta.exists():
            with open(meta, encoding="utf-8") as f:
                return json.load(f)
    return None


def find_aggregates(results_dir: Path) -> List[Path]:
    return sorted(Path(results_dir).glob("*_aggregate.json"))


def _resolve_task_id(
    rec: Dict[str, Any], registry: Registry, provider_hint: Optional[str]
) -> Optional[str]:
    tid = rec.get("icil_task_id")
    if tid and tid in registry.tasks:
        return tid
    suite, lang = (
        rec.get("suite"),
        normalize_language(str(rec.get("name", "")).split("|")[0].replace("_", " ")),
    )
    matches = [
        t.task_id
        for t in registry.tasks.values()
        if (suite is None or t.suite == suite)
        and (provider_hint is None or t.provider == provider_hint)
        and normalize_language(t.language) == lang
    ]
    return matches[0] if len(matches) == 1 else None


def ingest(
    results_dir: Path,
    registry: Registry,
    server_logs: Optional[Iterable[Path]] = None,
    default_track: str = "configuration",
    default_condition: str = "k1",
) -> Dict[str, Any]:
    """Return ``{"rows": [EpisodeRow], "configs": [...], "server_meta": {...} | None}``."""
    server_rows = load_server_log(server_logs or [])
    server_meta = load_server_meta(server_logs or [])
    rows: List[EpisodeRow] = []
    configs: List[Dict[str, Any]] = []
    for path in find_aggregates(results_dir):
        with open(path, encoding="utf-8") as f:
            agg = json.load(f)
        configs.append(
            {
                "file": path.name,
                "config": agg.get("config"),
                "server_info": agg.get("server_info"),
                "seed": agg.get("seed"),
            }
        )
        provider_hint = (agg.get("config") or {}).get("params", {}).get("provider")
        for task_block in agg.get("tasks", []):
            for rec in task_block.get("episodes", []):
                task_id = _resolve_task_id(rec, registry, provider_hint)
                if task_id is None:
                    continue
                task = registry.tasks[task_id]
                cond_name = rec.get("icil_condition") or default_condition
                cond = Condition.parse(cond_name)
                metrics = rec.get("metrics") or {}
                srv = server_rows.get(rec.get("eid") or "")
                ctx = (srv or {}).get("context") or {}
                row = EpisodeRow(
                    task_id=task_id,
                    provider=task.provider,
                    suite=task.suite,
                    track=rec.get("icil_track") or default_track,
                    condition=cond_name,
                    k=ctx.get("k", cond.k),
                    control=cond.control,
                    episode_idx=int(rec.get("episode_idx", rec.get("episode_id", 0))),
                    success=None
                    if metrics.get("success") is None
                    else bool(metrics.get("success")),
                    progress=metrics.get("progress"),
                    steps=rec.get("steps"),
                    elapsed_s=rec.get("elapsed_sec"),
                    eid=rec.get("eid"),
                    benchmark=agg.get("benchmark"),
                )
                if rec.get("failure_reason"):
                    row.status, row.error = "error", str(rec.get("failure_reason"))
                if srv:
                    row.status = (
                        srv.get("status", row.status) if srv.get("status") != "ok" else row.status
                    )
                    row.error = srv.get("error") or row.error
                    row.relation = ctx.get("relation")
                    row.transform = ctx.get("transform")
                    row.language = ctx.get("language")
                    row.context_refs = ctx.get("refs") or []
                    row.context_hash = ctx.get("context_hash")
                    row.context_task_id = ctx.get("context_task_id")
                    row.k_max = srv.get("k_max")
                    row.adaptation_latency_s = srv.get("adaptation_latency_s")
                    row.control_latency_s = (srv.get("control_latency_s") or {}).get("median")
                    row.min_context_init_l2 = srv.get("min_context_init_l2")
                if row.status in ("unsupported", "error"):
                    row.success = None  # hold actions are not a policy outcome
                    row.progress = None
                rows.append(row)
    return {"rows": rows, "configs": configs, "server_meta": server_meta}


def tag_exposure(
    rows: List[EpisodeRow], registry: Registry, card: Optional[Dict[str, Any]]
) -> None:
    """Fill ``query_exposure`` / ``context_exposure`` from a policy card."""
    if not card:
        return
    training_tasks = card.get("training_tasks")
    training_eps = card.get("training_episodes") or {}
    if training_tasks is None:
        return
    seen_langs = {normalize_language(t) for t in training_tasks}
    seen_ids = set(training_tasks)

    def task_seen(task_id: str) -> bool:
        t = registry.tasks[task_id]
        return task_id in seen_ids or normalize_language(t.language) in seen_langs

    for row in rows:
        row.query_exposure = "seen" if task_seen(row.task_id) else "unseen"
        if not row.context_refs:
            row.context_exposure = "none"
            continue
        flags = []
        for ref in row.context_refs:
            ctask, idx = ref[0], int(ref[1])
            lang = registry.tasks[ctask].language if ctask in registry.tasks else ctask
            eps = training_eps.get(lang) or training_eps.get(ctask)
            if eps is not None:
                flags.append(idx in set(eps))
            else:
                flags.append(task_seen(ctask) if ctask in registry.tasks else False)
        row.context_exposure = "seen" if all(flags) else "unseen" if not any(flags) else "partial"
