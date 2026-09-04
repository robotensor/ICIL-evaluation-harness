"""``icil-eval capabilities``: compute per-task supported K for a context budget."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from icil_eval.paths import cache_root


def add_capabilities_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", default="bpp", help="policy name recorded in the file")
    parser.add_argument(
        "--budget-chunks",
        type=int,
        default=50,
        help="prompt chunk budget (BPP: prompt_pos_emb length)",
    )
    parser.add_argument("--chunk-len", type=int, default=20, help="steps per prompt chunk")
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-sweep", default="0,1,2,4,8")
    parser.add_argument("--provider", default="libero")
    parser.add_argument(
        "--demo-root", help="raw demonstration root (default: cache raw/<provider>)"
    )
    parser.add_argument("--registry-root")
    parser.add_argument(
        "--out", help="output JSON (default: cache/capabilities/<policy>_<provider>.json)"
    )


def run_capabilities(args: argparse.Namespace) -> int:
    from icil_eval.context.loaders.hdf5 import LiberoHdf5Store
    from icil_eval.policy.capabilities import k_max_by_task, write_capabilities
    from icil_eval.policy.protocol import PolicyCard, PolicySpec
    from icil_eval.registry.io import load_registry

    registry = load_registry(Path(args.registry_root) if args.registry_root else None)
    store = LiberoHdf5Store(
        Path(args.demo_root) if args.demo_root else cache_root() / "raw" / args.provider
    )
    spec = PolicySpec(
        name=args.policy,
        context_mode="sequence",
        k_min=args.k_min,
        k_max=None,
        context_budget={"chunks": args.budget_chunks, "chunk_len": args.chunk_len},
        needs_every_observation=True,
        order_invariant=False,
        requires_language=False,
        k0_semantics="blank_prompt" if args.k_min == 0 else "none",
        cameras=["image", "image2"],
        action_space="osc_pose_delta_7d",
        action_horizon=16,
        exec_horizon=12,
        control_hz=20.0,
        card=PolicyCard(name=args.policy, version="capabilities", exposure_source="undisclosed"),
    )
    tasks = [
        t.task_id
        for t in registry.tasks.values()
        if t.provider == args.provider and store.available(t)
    ]
    kmax = k_max_by_task(
        spec, registry, store.episode_lengths, [int(k) for k in args.k_sweep.split(",")], tasks
    )
    out = (
        Path(args.out)
        if args.out
        else cache_root() / "capabilities" / f"{args.policy}_{args.provider}.json"
    )
    write_capabilities(out, spec, kmax, {"provider": args.provider, "n_tasks": len(kmax)})
    from collections import Counter

    dist = dict(sorted(Counter(kmax.values()).items(), key=lambda kv: str(kv[0])))
    print(f"wrote {out}: {len(kmax)} tasks; k_max distribution {dist}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="icil-eval capabilities")
    add_capabilities_arguments(parser)
    return run_capabilities(parser.parse_args(argv))
