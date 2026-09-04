"""Canonical hashing used for task ids, layout ids, context seeds and registry versions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Tuple


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any, length: int = 64) -> str:
    return sha256_hex(canonical_json(obj))[:length]


def sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_files(entries: Iterable[Tuple[str, str]]) -> str:
    """Hash a sorted list of (relative path, file sha256) pairs."""
    return sha256_hex(canonical_json(sorted(entries)))
