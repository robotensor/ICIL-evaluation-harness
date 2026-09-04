"""Generate vla-eval run configurations for ICIL tracks (presets ``smoke``, ``quick``, ``full``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from icil_eval.registry.io import Registry, load_registry

VLA_EVAL_IMAGES = {"libero": "ghcr.io/allenai/vla-evaluation-harness/libero:0.5.0"}
DEFAULT_INNER = {"libero": "icil_eval.backends.vla_eval.libero:ICILLIBEROBenchmark"}

PRESETS: Dict[str, Dict[str, Any]] = {
    # 2 query tasks x 5 init states x {k1, k1.wrong_task}
    "smoke": {"episodes_per_task": 5, "max_query_tasks": 2, "conditions": ["k1", "k1.wrong_task"]},
    # all query tasks of the selected suites x 10 init states x every track condition
    "quick": {"episodes_per_task": 10, "max_query_tasks": None, "conditions": None},
    # the full protocol: 50 init states per condition
    "full": {"episodes_per_task": 50, "max_query_tasks": None, "conditions": None},
}


def build_run_config(
    *,
    provider: str,
    track: str,
    suites: Sequence[str],
    preset: str,
    output_dir: Path,
    server_url: str = "ws://localhost:8000",
    server_timeout: int = 300,
    seed: int = 1000,
    resolution: int = 128,
    num_steps_wait: int = 10,
    capabilities: Optional[str] = None,
    query_tasks: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    episodes_per_task: Optional[int] = None,
    docker_image: Optional[str] = None,
    docker_volumes: Optional[Sequence[str]] = None,
    docker_env: Optional[Sequence[str]] = None,
    registry: Optional[Registry] = None,
) -> Dict[str, Any]:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset '{preset}'; known: {sorted(PRESETS)}")
    registry = registry or load_registry()
    if track not in registry.tracks:
        raise KeyError(f"unknown track '{track}'")
    p = PRESETS[preset]
    n_episodes = int(episodes_per_task or p["episodes_per_task"])
    if n_episodes > 50:
        raise ValueError(
            "episodes_per_task must be <= 50 (LIBERO has 50 evaluation initial states)"
        )
    conds = list(conditions or p["conditions"] or registry.tracks[track].conditions)

    benchmarks: List[Dict[str, Any]] = []
    for suite in suites:
        max_steps = max(
            (t.horizon.max_steps for t in registry.tasks_of(provider, suite)), default=None
        )
        params: Dict[str, Any] = {
            "inner": DEFAULT_INNER.get(provider, DEFAULT_INNER["libero"]),
            "track": track,
            "provider": provider,
            "suite": suite,
            "seed": seed,
            "num_steps_wait": num_steps_wait,
            "resolution": resolution,
            "conditions": list(conds),
        }
        if max_steps is not None:
            params["max_steps"] = int(max_steps)
        if p["max_query_tasks"] is not None:
            params["max_query_tasks"] = int(p["max_query_tasks"])
        if query_tasks:
            params["query_tasks"] = list(query_tasks)
        if capabilities:
            params["capabilities"] = str(capabilities)
        benchmarks.append(
            {
                "benchmark": "icil_eval.backends.vla_eval.benchmark:ICILBenchmark",
                "subname": f"{provider}_{suite}_{track}_{preset}",
                "episodes_per_task": n_episodes,
                "params": params,
            }
        )

    cfg: Dict[str, Any] = {
        "server": {"url": server_url, "timeout": int(server_timeout)},
        "output_dir": str(output_dir),
        "benchmarks": benchmarks,
        "icil": {
            "provider": provider,
            "track": track,
            "preset": preset,
            "registry_hash": registry.registry_hash(),
            "conditions": list(conds),
            "episodes_per_task": n_episodes,
        },
    }
    image = docker_image or VLA_EVAL_IMAGES.get(provider)
    if image:
        docker: Dict[str, Any] = {"image": image}
        if docker_volumes:
            docker["volumes"] = list(docker_volumes)
        if docker_env:
            docker["env"] = list(docker_env)
        cfg["docker"] = docker
    return cfg


def write_run_config(cfg: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
