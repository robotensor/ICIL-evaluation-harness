"""Validate the committed registry: structural invariants plus JSON-Schema when available."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from icil_eval.paths import REGISTRY_ROOT, SCHEMAS_ROOT
from icil_eval.registry.io import _load, load_registry
from icil_eval.registry.relations import RELATIONS
from icil_eval.registry.schema import LANGUAGE_MODES

_TASK_ID = re.compile(r"^[a-z0-9_]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _schema_validate(payload, schema_name: str, errors: List[str], where: str) -> None:
    try:
        import jsonschema
    except ImportError:  # schema extra not installed: structural checks only
        return
    schema_path = SCHEMAS_ROOT / f"{schema_name}.schema.json"
    if not schema_path.exists():
        return
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path)
        errors.append(f"{where}: {loc}: {err.message}")


def validate_registry(root: Optional[Path] = None) -> int:
    root = Path(root) if root is not None else REGISTRY_ROOT
    errors: List[str] = []
    reg = load_registry(root)

    ids = list(reg.tasks)
    if ids != sorted(ids):
        pass  # load order is by file; per-file order is checked below
    for path in sorted((root / "tasks").rglob("*.yaml")) if (root / "tasks").exists() else []:
        payload = _load(path)
        _schema_validate(payload, "tasks_file", errors, str(path.relative_to(root)))
        file_ids = [t["task_id"] for t in payload.get("tasks", [])]
        if file_ids != sorted(file_ids):
            errors.append(f"{path.relative_to(root)}: tasks not sorted by task_id")
        for t in payload.get("tasks", []):
            if not _TASK_ID.match(t["task_id"]):
                errors.append(f"{t['task_id']}: malformed task id")
            if not t["task_id"].isascii():
                errors.append(f"{t['task_id']}: non-ASCII task id")
            expected_prefix = f"{payload['provider']}/{payload['suite']}/"
            if not t["task_id"].startswith(expected_prefix):
                errors.append(f"{t['task_id']}: does not match file {expected_prefix}")
            for rel in t.get("capabilities", {}).get("relations", []):
                if rel not in RELATIONS:
                    errors.append(f"{t['task_id']}: unknown relation '{rel}'")

    for path in (
        sorted((root / "providers").glob("*.yaml")) if (root / "providers").exists() else []
    ):
        _schema_validate(_load(path), "provider", errors, str(path.relative_to(root)))
    for name, track in reg.tracks.items():
        _schema_validate(track.to_dict(), "track", errors, f"tracks/{name}.yaml")
        if track.relation not in RELATIONS:
            errors.append(f"tracks/{name}.yaml: unknown relation '{track.relation}'")
        if track.language not in LANGUAGE_MODES:
            errors.append(f"tracks/{name}.yaml: unknown language mode '{track.language}'")
        if track.k_ref not in track.k_sweep:
            errors.append(f"tracks/{name}.yaml: k_ref {track.k_ref} not in k_sweep")
        for rel in [track.wrong_task_rule.get("relation"), track.wrong_task_rule.get("fallback")]:
            if rel and rel not in RELATIONS:
                errors.append(f"tracks/{name}.yaml: unknown wrong-task relation '{rel}'")

    for (track, provider), entries in reg.pools.items():
        if track not in reg.tracks:
            errors.append(f"pools/{track}/{provider}.yaml: unknown track")
        for e in entries.values():
            for tid in [e.task_id] + e.context + [w[0] for w in e.wrong]:
                if tid not in reg.tasks:
                    errors.append(f"pools/{track}/{provider}.yaml: unknown task {tid}")

    # No two tracks may expand to identical episode sets for the same provider.
    seen = {}
    for (track, provider), entries in reg.pools.items():
        key = (provider, tuple(sorted((e.task_id, tuple(e.context)) for e in entries.values())))
        if key in seen and reg.tracks[track].language == reg.tracks[seen[key]].language:
            errors.append(
                f"tracks '{track}' and '{seen[key]}' expand to identical pools for {provider}"
            )
        seen.setdefault(key, track)

    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        print(f"{len(errors)} error(s)", file=sys.stderr)
        return 1
    print(
        f"registry OK: {len(reg.tasks)} tasks, {len(reg.providers)} providers, "
        f"{len(reg.tracks)} tracks, {len(reg.pools)} pool files; hash {reg.registry_hash()}"
    )
    return 0
