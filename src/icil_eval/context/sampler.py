"""Deterministic, hash-seeded context sampling (standard S2).

Both the benchmark side (inside a simulator container) and the policy side (model server)
recompute the identical :class:`ContextSpec` from the same registry and the same flat inputs, so
context travels by reference and never needs to be transmitted.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional

from icil_eval.context.types import Condition, ContextRef, ContextSpec, UnsupportedContext
from icil_eval.registry.hashing import hash_obj
from icil_eval.registry.io import Registry


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from arbitrary parts (never depends on Python's hash randomisation)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def sample_context(
    registry: Registry,
    registry_hash: str,
    track: str,
    query_task_id: str,
    condition: Condition,
    episode_idx: int,
    seed: int,
    n_demos: Optional[Dict[str, int]] = None,
    modality: str = "sensorimotor",
) -> ContextSpec:
    """Build the context spec for one episode.

    ``condition.k`` must already be resolved (no ``kmax``). ``n_demos`` optionally overrides the
    registry's per-task demonstration count (e.g. when a local pool is smaller).
    """
    if condition.k is None:
        raise ValueError("resolve 'kmax' against the policy before sampling")
    trk = registry.tracks[track]
    if query_task_id not in registry.tasks:
        raise KeyError(f"unknown task {query_task_id}")
    entry = registry.pool(track, query_task_id)
    if entry is None:
        raise KeyError(f"task {query_task_id} is not in the pool of track '{track}'")

    rng = random.Random(
        derive_seed(
            registry_hash,
            seed,
            track,
            query_task_id,
            condition.k,
            trk.relation,
            condition.transform,
            trk.language,
            condition.control,
            episode_idx,
        )
    )

    refs: List[ContextRef] = []
    relation = trk.relation
    context_task_id: Optional[str] = None
    if condition.k > 0:
        if condition.wrong_task:
            if not entry.wrong:
                raise UnsupportedContext(
                    f"no wrong-task pool for {query_task_id} in track '{track}'"
                )
            context_task_id, relation = rng.choice(entry.wrong)
        else:
            context_task_id = rng.choice(entry.context)
        pool_size = (n_demos or {}).get(
            context_task_id, registry.tasks[context_task_id].demo_pool.n
        )
        if condition.k > pool_size:
            raise UnsupportedContext(
                f"k={condition.k} exceeds the {pool_size} demonstrations of {context_task_id}"
            )
        indices = rng.sample(range(pool_size), condition.k)
        refs = [ContextRef(task_id=context_task_id, episode_index=i) for i in indices]

    permutation_seed = rng.getrandbits(32) if condition.transform != "identity" else None
    spec = ContextSpec(
        track=track,
        query_task_id=query_task_id,
        condition=condition.name,
        k=condition.k,
        refs=refs,
        relation=relation if condition.k > 0 else "none",
        context_task_id=context_task_id,
        transform=condition.transform,
        language=trk.language,
        modality=modality,
        seed=seed,
        episode_idx=episode_idx,
        registry_hash=registry_hash,
        permutation_seed=permutation_seed,
    )
    spec.context_hash = hash_obj(
        {
            "refs": [r.to_list() for r in refs],
            "transform": spec.transform,
            "language": spec.language,
            "permutation_seed": permutation_seed,
            "registry_hash": registry_hash,
        },
        16,
    )
    return spec
