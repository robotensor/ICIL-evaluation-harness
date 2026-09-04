"""``icil-eval serve``: run an ICIL policy as a vla-eval model server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from icil_eval.paths import cache_root


def add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("policy", choices=["bpp"], help="policy wrapper to serve")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument(
        "--demo-root", help="root of raw provider demonstrations (default: cache raw/libero)"
    )
    parser.add_argument("--registry-root", help="registry root (default: packaged registry)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed", type=int, default=1000, help="context seed when the task carries none"
    )
    parser.add_argument(
        "--track", default="configuration", help="default track for stock vla-eval configs"
    )
    parser.add_argument(
        "--condition", default="k1", help="default condition for stock vla-eval configs"
    )
    parser.add_argument(
        "--log-dir", help="directory for the per-episode JSONL log (default: cache/server_logs)"
    )
    parser.add_argument(
        "--emulate-native",
        type=int,
        default=128,
        help="downsample incoming frames to this size before the policy's own resize (0 = off)",
    )
    parser.add_argument(
        "--camera-map", default="agentview=image,wrist=image2", help="benchmark=policy camera names"
    )
    parser.add_argument("--allow-network", action="store_true", help="do not force HF offline mode")


def _parse_camera_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in text.split(","):
        if item:
            src, dst = item.split("=", 1)
            out[src] = dst
    return out


def run_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from vla_eval.model_servers.serve import serve

    from icil_eval.backends.vla_eval.server import ICILModelServer
    from icil_eval.context.loaders.hdf5 import LiberoHdf5Store
    from icil_eval.registry.io import load_registry

    registry = load_registry(Path(args.registry_root) if args.registry_root else None)
    demo_root = Path(args.demo_root) if args.demo_root else cache_root() / "raw" / "libero"
    store = LiberoHdf5Store(demo_root)
    log_dir = Path(args.log_dir) if args.log_dir else cache_root() / "server_logs"
    log_path = log_dir / f"{args.policy}_{args.port}.jsonl"

    if args.policy == "bpp":
        from icil_eval.policies.bpp.model import BPPModel
        from icil_eval.policies.bpp.policy import BPPPolicy

        model = BPPModel(Path(args.ckpt), device=args.device, allow_network=args.allow_network)
        logging.getLogger("icil_eval").info(
            "loaded %s in %.1fs (budget %d chunks, sink=%s, exec %d/%d, %d training tasks)",
            args.ckpt,
            model.load_seconds,
            model.prompt_budget_chunks,
            model.attention_sink,
            model.exec_horizon,
            model.action_horizon,
            len(model.training_tasks),
        )

        def factory() -> BPPPolicy:
            return BPPPolicy(model, emulate_native=args.emulate_native)

    else:  # pragma: no cover
        raise SystemExit(f"unknown policy {args.policy}")

    server = ICILModelServer(
        factory,
        registry,
        store,
        seed=args.seed,
        default_track=args.track,
        default_condition=args.condition,
        camera_map=_parse_camera_map(args.camera_map),
        log_path=log_path,
    )
    logging.getLogger("icil_eval").info(
        "serving %s on ws://%s:%d; log %s", args.policy, args.host, args.port, log_path
    )
    serve(server, host=args.host, port=args.port)
    return 0


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    parser = argparse.ArgumentParser(prog="icil-eval serve")
    add_serve_arguments(parser)
    return run_serve(parser.parse_args(argv))
