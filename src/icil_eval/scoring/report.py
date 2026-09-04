"""Assemble ``icil_results.json`` and a markdown summary from a vla-eval run directory."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from icil_eval import __version__
from icil_eval.registry.io import Registry, load_registry
from icil_eval.scoring.ingest import ingest, tag_exposure
from icil_eval.scoring.profile import build_profile

RESULTS_SCHEMA_VERSION = 1


def _git_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "–"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def _fmt_ci(d: Optional[Dict[str, Any]]) -> str:
    if not d or d.get("value") is None:
        return "–"
    lo, hi = d.get("ci95") or [None, None]
    core = _fmt(d["value"])
    return f"{core} [{_fmt(lo)}, {_fmt(hi)}]" if lo is not None else core


def _fmt_delta(d: Optional[Dict[str, Any]]) -> str:
    if not d or d.get("value") is None:
        return "–"
    lo, hi = d.get("ci95") or [None, None]
    s = f"{100 * d['value']:+.1f} pp (n={d['n']}, discordant={d['n_discordant']}"
    if lo is not None:
        s += f", CI [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    if d.get("mcnemar_p") is not None:
        s += f", McNemar p={d['mcnemar_p']:.3g}"
    return s + ")"


def markdown_table(entry: Dict[str, Any], title: str) -> str:
    header = (
        "| condition | success [Wilson 95%] | macro (task-bootstrap 95%) | n | tasks | status "
        "| progress | adapt s | act s |"
    )
    lines = [f"#### {title}", "", header, "|---|---|---|---|---|---|---|---|---|"]
    for cond, s in entry["conditions"].items():
        macro = s["success_macro"]
        macro_s = _fmt(macro["value"])
        if macro["ci95"][0] is not None:
            macro_s += f" [{_fmt(macro['ci95'][0])}, {_fmt(macro['ci95'][1])}]"
        status = ", ".join(f"{k}={v}" for k, v in s["status_counts"].items())
        adapt = _fmt(s["adaptation_latency_s_median"], pct=False)
        act = _fmt(s["control_latency_s_median"], pct=False)
        lines.append(
            f"| `{cond}` | {_fmt_ci(s['success'])} | {macro_s} | {s['n_ok']}/{s['n_episodes']} "
            f"| {s['n_tasks']} | {status} | {_fmt(s['progress'])} | {adapt} | {act} |"
        )
    d = entry["derived"]
    sr = ", ".join(f"K={k}: {_fmt(v)}" for k, v in d["sr_at_k"].items())
    lines += [
        "",
        f"- Δ_context@1 (k1 − k1.wrong_task, paired): {_fmt_delta(d.get('delta_context@1'))}",
        f"- Δ_context@kmax (k{d.get('k_max_observed')}): {_fmt_delta(d.get('delta_context@kmax'))}",
        f"- order sensitivity@1 (k1 − k1.shuffled_chunks): "
        f"{_fmt_delta(d.get('order_sensitivity@1'))}",
        f"- SR@K: {sr}; supported K: {d['k_supported']}; context AUC: {_fmt(d['context_auc'])}",
    ]
    if d.get("delta_context_above_chance@1") is not None:
        lines.append(
            f"- Δ_context above chance@1: {100 * d['delta_context_above_chance@1']:+.1f} pp "
            f"(chance {_fmt(d.get('chance_success_mean'))})"
        )
    return "\n".join(lines) + "\n"


def render_markdown(results: Dict[str, Any]) -> str:
    out = [f"# ICIL results — {results['policy'].get('name') or 'policy'}", ""]
    cov = results["profile"]["coverage"]
    out.append(
        f"Coverage: {cov['tasks_run']} tasks run, {cov['episodes']} episodes; "
        f"status {cov['status_counts']}; query exposure {cov['query_exposure']}; "
        f"context exposure {cov['context_exposure']}."
    )
    out.append(
        f"Registry hash `{results['config']['registry_hash']}`; "
        f"harness {results['harness_version']}; date {results['date']}."
    )
    out.append("")
    for track, entry in results["profile"]["tracks"].items():
        out.append(f"## Track `{track}`")
        out.append("")
        out.append(markdown_table(entry, "all providers"))
        for sl, sentry in entry["slices"].items():
            out.append(markdown_table(sentry, f"slice `{sl}`"))
    return "\n".join(out)


def build_results(
    run_dir: Path,
    registry: Optional[Registry] = None,
    server_logs: Optional[Iterable[Path]] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    registry = registry or load_registry()
    results_dir = Path(results_dir) if results_dir else Path(run_dir) / "results"
    ingested = ingest(results_dir, registry, server_logs=server_logs)
    rows = ingested["rows"]
    meta = ingested["server_meta"] or {}
    spec = meta.get("policy_spec") or {}
    card = spec.get("card") or meta.get("card")
    tag_exposure(rows, registry, card)
    profile = build_profile(rows, registry)
    run_config: Dict[str, Any] = {}
    cfg_path = Path(run_dir) / "config.yaml"
    if cfg_path.exists():
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            run_config = yaml.safe_load(f) or {}
    backend_version = None
    if ingested["configs"]:
        backend_version = (ingested["configs"][0].get("server_info") or {}).get("harness_version")
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "harness_version": __version__,
        "backend": {"name": "vla_eval", "version": backend_version},
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": _git_hash(),
        "policy": {"name": spec.get("name"), "spec": spec or None, "card": card},
        "config": {
            "registry_hash": registry.registry_hash(),
            "run": run_config.get("icil"),
            "backend_configs": ingested["configs"],
        },
        "profile": profile,
        "episodes": [r.to_dict() for r in rows],
    }


def write_results(results: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "icil_results.json"
    md_path = out_dir / "icil_results.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    md_path.write_text(render_markdown(results), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def validate_results(results: Dict[str, Any]) -> List[str]:
    try:
        import jsonschema
    except ImportError:
        return []
    from icil_eval.paths import SCHEMAS_ROOT

    schema_path = SCHEMAS_ROOT / "results.schema.json"
    if not schema_path.exists():
        return []
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    payload = json.loads(json.dumps(results, default=str))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in validator.iter_errors(payload)
    ]
