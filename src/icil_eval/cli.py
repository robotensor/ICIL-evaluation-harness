"""``icil-eval`` command line interface.

Kept dependency-free (argparse only) so the entry point works in the minimal core install used
inside simulator containers. Sub-commands import their implementation lazily.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from icil_eval import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icil-eval",
        description=(
            "ICIL-evaluation-harness: a global benchmark for in-context imitation learning."
        ),
    )
    parser.add_argument("--version", action="version", version=f"icil-eval {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    registry = sub.add_parser("registry", help="build, validate and inspect the task registry")
    registry_sub = registry.add_subparsers(dest="registry_command", metavar="<subcommand>")
    build = registry_sub.add_parser("build", help="build registry entries for a provider")
    build.add_argument("provider", help="provider name, e.g. libero")
    registry_sub.add_parser("validate", help="validate the committed registry against the schemas")

    sub.add_parser("serve", help="serve an ICIL policy to a rollout backend")
    sub.add_parser("run", help="run an evaluation through a rollout backend")
    sub.add_parser("report", help="score backend results into an ICIL results file and tables")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "registry":
        if args.registry_command == "build":
            from icil_eval.registry.build import build_provider

            return build_provider(args.provider)
        if args.registry_command == "validate":
            from icil_eval.registry.validate import validate_registry

            return validate_registry()
        parser.parse_args([args.command, "--help"])
        return 2
    print(f"'{args.command}' is not implemented yet in this pre-release.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
