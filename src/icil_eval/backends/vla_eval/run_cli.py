"""``icil-eval run``: generate a vla-eval configuration for an ICIL track and execute it."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from icil_eval.paths import cache_root


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="libero")
    parser.add_argument("--track", default="configuration")
    parser.add_argument("--suites", default="libero_goal", help="comma-separated provider suites")
    parser.add_argument("--preset", default="smoke", choices=["smoke", "quick", "full"])
    parser.add_argument("--conditions", help="comma-separated conditions (override the preset)")
    parser.add_argument("--episodes-per-task", type=int, help="override the preset (<= 50)")
    parser.add_argument("--query-tasks", help="comma-separated registry task ids to restrict to")
    parser.add_argument(
        "--capabilities",
        help="policy capabilities JSON ({k_max_by_task}) to skip unsupported conditions",
    )
    parser.add_argument("--server-url", default="ws://localhost:8000")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--resolution", type=int, default=128, help="LIBERO render size (128 = BPP native)"
    )
    parser.add_argument(
        "--run-dir", help="directory for config, log and results (default: cache/runs/<name>)"
    )
    parser.add_argument("--name", help="run name (default: <provider>_<track>_<preset>)")
    parser.add_argument(
        "--no-docker", action="store_true", help="run the benchmark in this environment"
    )
    parser.add_argument(
        "--docker-image", help="ICIL image to run the benchmark in (default: vla-eval base image)"
    )
    parser.add_argument("--shards", type=int, default=1, help="number of parallel vla-eval shards")
    parser.add_argument(
        "--docker-volume", action="append", default=[], help="extra docker -v mounts"
    )
    parser.add_argument("--docker-env", action="append", default=[], help="extra docker -e vars")
    parser.add_argument(
        "--gpus", help="GPUs for the benchmark container (docker --gpus), e.g. all or 0"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write the config and print the command only"
    )


def _csv(text: Optional[str]) -> Optional[List[str]]:
    return [x for x in text.split(",") if x] if text else None


def run_run(args: argparse.Namespace) -> int:
    from icil_eval.backends.vla_eval.config import build_run_config, write_run_config

    name = args.name or f"{args.provider}_{args.track}_{args.preset}"
    run_dir = Path(args.run_dir) if args.run_dir else cache_root() / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_run_config(
        provider=args.provider,
        track=args.track,
        suites=_csv(args.suites) or [],
        preset=args.preset,
        output_dir=run_dir / "results",
        server_url=args.server_url,
        seed=args.seed,
        resolution=args.resolution,
        capabilities=args.capabilities,
        query_tasks=_csv(args.query_tasks),
        conditions=_csv(args.conditions),
        episodes_per_task=args.episodes_per_task,
        docker_image=args.docker_image,
        docker_volumes=args.docker_volume or None,
        docker_env=args.docker_env or None,
        docker_gpus=args.gpus,
    )
    cfg_path = write_run_config(cfg, run_dir / "config.yaml")
    exe = Path(sys.executable).parent / "vla-eval"  # prefer the interpreter's own environment
    vla_eval = str(exe) if exe.exists() else "vla-eval"
    base = [vla_eval, "run", "-c", str(cfg_path), "-y"]
    if args.no_docker:
        base.append("--no-docker")
    commands = []
    if args.shards > 1:
        for i in range(args.shards):
            commands.append(base + ["--shard-id", str(i), "--num-shards", str(args.shards)])
    else:
        commands.append(base)
    print(f"config: {cfg_path}")
    for cmd in commands:
        print("$", " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    procs = [subprocess.Popen(cmd, cwd=str(run_dir), env=env) for cmd in commands]
    rc = 0
    for p in procs:
        rc = max(rc, p.wait())
    if args.shards > 1 and rc == 0:
        merge = [vla_eval, "merge", "-c", str(cfg_path)]
        print("$", " ".join(merge))
        rc = subprocess.call(merge, cwd=str(run_dir), env=env)
    print(f"results: {run_dir / 'results'}", file=sys.stderr)
    return rc
