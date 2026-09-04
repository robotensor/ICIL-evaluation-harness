"""Build the model profile (standard S5) from episode rows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from icil_eval.context.types import Condition
from icil_eval.registry.io import Registry
from icil_eval.scoring.ingest import EpisodeRow
from icil_eval.scoring.stats import (
    cluster_bootstrap_mean,
    group_by,
    mean_or_none,
    median_or_none,
    paired_difference,
    rate,
)

K_SWEEP = (0, 1, 2, 4, 8)


def _ok(rows: Sequence[EpisodeRow]) -> List[EpisodeRow]:
    return [r for r in rows if r.status == "ok" and r.success is not None]


def _condition_summary(rows: Sequence[EpisodeRow]) -> Dict[str, Any]:
    ok = _ok(rows)
    by_task = group_by(ok, lambda r: r.task_id)
    macro = cluster_bootstrap_mean({t: [float(r.success) for r in rs] for t, rs in by_task.items()})
    statuses = group_by(rows, lambda r: r.status)
    return {
        "success": rate([r.success for r in ok]),
        "success_macro": macro,
        "progress": mean_or_none([r.progress for r in ok]),
        "n_episodes": len(rows),
        "n_ok": len(ok),
        "n_tasks": len(by_task),
        "status_counts": {s: len(v) for s, v in sorted(statuses.items())},
        "adaptation_latency_s_median": median_or_none([r.adaptation_latency_s for r in ok]),
        "control_latency_s_median": median_or_none([r.control_latency_s for r in ok]),
        "steps_mean": mean_or_none([r.steps for r in ok]),
    }


def _paired(rows_a: Sequence[EpisodeRow], rows_b: Sequence[EpisodeRow]) -> Dict[str, Any]:
    """Pair by (task_id, episode_idx): same evaluation initial state under two conditions."""
    idx_b = {(r.task_id, r.episode_idx): r.success for r in _ok(rows_b)}
    a_vals, b_vals = [], []
    for r in _ok(rows_a):
        key = (r.task_id, r.episode_idx)
        if key in idx_b:
            a_vals.append(r.success)
            b_vals.append(idx_b[key])
    return paired_difference(a_vals, b_vals)


def _k_name(k: int) -> str:
    return f"k{k}"


def _resolve_kmax_rows(rows: Sequence[EpisodeRow]) -> Dict[str, List[EpisodeRow]]:
    """Group rows by condition, resolving ``kmax.*`` names with the server-reported k."""
    out: Dict[str, List[EpisodeRow]] = {}
    for r in rows:
        name = r.condition
        if name.startswith("kmax") and r.k is not None:
            name = Condition(k=r.k, control=r.control).name
        out.setdefault(name, []).append(r)
    return out


def _derived(
    by_cond: Dict[str, List[EpisodeRow]], rows: Sequence[EpisodeRow], registry: Registry
) -> Dict[str, Any]:
    sr_at_k: Dict[str, Optional[float]] = {}
    k_supported: List[int] = []
    for k in K_SWEEP:
        cond_rows = by_cond.get(_k_name(k), [])
        ok = _ok(cond_rows)
        sr_at_k[str(k)] = rate([r.success for r in ok])["value"] if ok else None
        if ok:
            k_supported.append(k)
    auc = (
        None
        if any(sr_at_k[str(k)] is None for k in K_SWEEP)
        else sum(sr_at_k[str(k)] for k in K_SWEEP) / len(K_SWEEP)
    )

    def wrong_for(k: int) -> List[EpisodeRow]:
        return by_cond.get(Condition(k=k, control="wrong_task").name, [])

    kmax_values = sorted(
        {
            r.k
            for r in rows
            if r.control == "wrong_task" and r.k is not None and r.condition.startswith("kmax")
        }
    )
    k_max = max(kmax_values) if kmax_values else (max(k_supported) if k_supported else None)
    out: Dict[str, Any] = {
        "sr_at_k": sr_at_k,
        "k_supported": k_supported,
        "context_auc": auc,
        "k_max_observed": k_max,
        "delta_context@1": _paired(by_cond.get("k1", []), wrong_for(1)),
        "delta_context@kmax": _paired(by_cond.get(_k_name(k_max), []), wrong_for(k_max))
        if k_max
        else None,
        "order_sensitivity@1": _paired(
            by_cond.get("k1", []), by_cond.get("k1.shuffled_chunks", [])
        ),
        "context_gain": {
            str(k): _paired(by_cond.get(_k_name(k), []), by_cond.get("k0", []))
            for k in K_SWEEP
            if k > 0
        },
    }
    # chance-adjusted wrong-context gap when the registry knows chance levels
    ok1 = _ok(by_cond.get("k1", []))
    chance = mean_or_none(
        [
            registry.tasks[r.task_id].bounds.chance_success
            for r in ok1
            if r.task_id in registry.tasks
        ]
    )
    wrong1 = _ok(wrong_for(1))
    if ok1 and (wrong1 or chance is not None):
        sr1 = rate([r.success for r in ok1])["value"]
        srw = rate([r.success for r in wrong1])["value"] if wrong1 else None
        floor = (
            max([v for v in (srw, chance) if v is not None])
            if (srw is not None or chance is not None)
            else None
        )
        out["delta_context_above_chance@1"] = None if floor is None else sr1 - floor
        out["chance_success_mean"] = chance
    return out


def build_profile(
    rows: List[EpisodeRow], registry: Registry, slice_key=lambda r: r.suite
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"tracks": {}, "coverage": {}}
    by_track = group_by(rows, lambda r: r.track)
    for track, trows in sorted(by_track.items()):
        by_cond = _resolve_kmax_rows(trows)
        entry: Dict[str, Any] = {
            "conditions": {c: _condition_summary(rs) for c, rs in sorted(by_cond.items())},
            "derived": _derived(by_cond, trows, registry),
            "providers": {},
            "slices": {},
        }
        for provider, prows in sorted(group_by(trows, lambda r: r.provider).items()):
            pcond = _resolve_kmax_rows(prows)
            entry["providers"][provider] = {
                "conditions": {c: _condition_summary(rs) for c, rs in sorted(pcond.items())},
                "derived": _derived(pcond, prows, registry),
            }
        for sl, srows in sorted(group_by(trows, slice_key).items()):
            scond = _resolve_kmax_rows(srows)
            entry["slices"][sl] = {
                "conditions": {c: _condition_summary(rs) for c, rs in sorted(scond.items())},
                "derived": _derived(scond, srows, registry),
            }
        profile["tracks"][track] = entry

    statuses = group_by(rows, lambda r: r.status)
    profile["coverage"] = {
        "tasks_run": len({r.task_id for r in rows if r.status == "ok"}),
        "tasks_total_in_results": len({r.task_id for r in rows}),
        "episodes": len(rows),
        "status_counts": {s: len(v) for s, v in sorted(statuses.items())},
        "providers": sorted({r.provider for r in rows}),
        "tracks": sorted(by_track),
        "query_exposure": {
            e: len(v) for e, v in sorted(group_by(rows, lambda r: r.query_exposure).items())
        },
        "context_exposure": {
            e: len(v) for e, v in sorted(group_by(rows, lambda r: r.context_exposure).items())
        },
    }
    return profile
