"""``icil-eval report``: score a run directory into icil_results.json + markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from icil_eval.paths import cache_root


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", help="run directory (contains config.yaml and results/)")
    parser.add_argument("--results-dir", help="override the results directory")
    parser.add_argument(
        "--server-log",
        action="append",
        default=[],
        help="ICIL server JSONL log(s); default: all in cache/server_logs",
    )
    parser.add_argument("--registry-root")
    parser.add_argument("--out", help="output directory (default: <run_dir>/icil)")


def run_report(args: argparse.Namespace) -> int:
    from icil_eval.registry.io import load_registry
    from icil_eval.scoring.report import build_results, validate_results, write_results

    run_dir = Path(args.run_dir)
    logs: List[Path] = [Path(p) for p in args.server_log] or sorted(
        (cache_root() / "server_logs").glob("*.jsonl")
    )
    registry = load_registry(Path(args.registry_root) if args.registry_root else None)
    results = build_results(
        run_dir,
        registry,
        server_logs=logs,
        results_dir=Path(args.results_dir) if args.results_dir else None,
    )
    errors = validate_results(results)
    paths = write_results(results, Path(args.out) if args.out else run_dir / "icil")
    print(paths["markdown"].read_text())
    print(f"wrote {paths['json']} and {paths['markdown']}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"SCHEMA ERROR {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="icil-eval report")
    add_report_arguments(parser)
    return run_report(parser.parse_args(argv))
