"""Context transforms used by the controls (S2): chunk shuffling and action reversal."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import List, Tuple

import numpy as np

from icil_eval.context.types import Demonstration
from icil_eval.registry.hashing import hash_obj


def chunk_slices(length: int, chunk_len: int) -> List[slice]:
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    return [slice(s, min(s + chunk_len, length)) for s in range(0, length, chunk_len)]


def shuffled_chunks(
    demo: Demonstration, chunk_len: int, seed: int
) -> Tuple[Demonstration, List[int], str]:
    """Permute contiguous chunks of (images, state, action, rewards) jointly.

    Returns the transformed demonstration, the permutation and its hash. A one-chunk
    demonstration is returned unchanged (identity permutation).
    """
    length = demo.length
    slices = chunk_slices(length, chunk_len)
    perm = list(range(len(slices)))
    rng = random.Random(seed)
    if len(perm) > 1:
        while True:
            rng.shuffle(perm)
            if perm != list(range(len(slices))):
                break
    index = np.concatenate([np.arange(length)[slices[p]] for p in perm])

    def take(arr):
        return None if arr is None else np.ascontiguousarray(arr[index])

    out = replace(
        demo,
        images={k: take(v) for k, v in demo.images.items()},
        state=take(demo.state),
        action=take(demo.action),
        rewards=take(demo.rewards),
        extra={
            **demo.extra,
            "transform": "shuffled_chunks",
            "chunk_len": chunk_len,
            "permutation": perm,
        },
    )
    return out, perm, hash_obj(perm, 16)


def reversed_actions(demo: Demonstration) -> Demonstration:
    """Reverse the action sequence in time while keeping observations in order.

    A policy that only matches visuals is unaffected; a policy that reads actions is misled.
    """
    if demo.action is None:
        raise ValueError("reversed_actions requires actions")
    return replace(
        demo,
        action=np.ascontiguousarray(demo.action[::-1]),
        extra={**demo.extra, "transform": "reversed_actions"},
    )


def apply_transform(
    demo: Demonstration, transform: str, chunk_len: int, seed: int
) -> Demonstration:
    if transform == "identity":
        return demo
    if transform == "shuffled_chunks":
        return shuffled_chunks(demo, chunk_len, seed)[0]
    if transform == "reversed_actions":
        return reversed_actions(demo)
    raise ValueError(f"unknown transform '{transform}'")
