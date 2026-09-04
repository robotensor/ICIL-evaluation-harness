from __future__ import annotations

import numpy as np

from icil_eval.context.transforms import chunk_slices, reversed_actions, shuffled_chunks
from icil_eval.context.types import Demonstration


def _demo(T=25):
    return Demonstration(
        images={
            "image": np.arange(T)[:, None, None, None]
            .repeat(2, 1)
            .repeat(2, 2)
            .repeat(3, 3)
            .astype(np.uint8)
        },
        state=np.arange(T, dtype=np.float32)[:, None].repeat(8, 1),
        action=np.arange(T, dtype=np.float32)[:, None].repeat(7, 1),
        fps=20.0,
        rewards=np.zeros(T, dtype=np.float32),
    )


def test_chunk_slices_cover_everything():
    s = chunk_slices(25, 10)
    assert [(x.start, x.stop) for x in s] == [(0, 10), (10, 20), (20, 25)]


def test_shuffled_chunks_is_a_joint_permutation():
    demo = _demo()
    out, perm, h = shuffled_chunks(demo, chunk_len=10, seed=7)
    assert sorted(perm) == [0, 1, 2] and perm != [0, 1, 2]
    assert sorted(out.action[:, 0].tolist()) == list(range(25))
    # images, state and action stay aligned frame by frame
    assert np.array_equal(out.images["image"][:, 0, 0, 0].astype(np.float32), out.action[:, 0])
    assert np.array_equal(out.state[:, 0], out.action[:, 0])
    assert len(h) == 16
    out2, perm2, _ = shuffled_chunks(demo, chunk_len=10, seed=7)
    assert perm2 == perm


def test_reversed_actions_keeps_observations():
    demo = _demo()
    out = reversed_actions(demo)
    assert np.array_equal(out.action[:, 0], np.arange(25)[::-1])
    assert np.array_equal(out.state, demo.state)
