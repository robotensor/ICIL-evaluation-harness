from __future__ import annotations

import numpy as np

from icil_eval.policies.bpp.prompt import chunk_actions, chunk_plan, to_policy_image


def test_chunk_plan_matches_bpp_rule():
    # L = ceil(T / 20), observations strided by 20, trailing chunk zero padded
    assert chunk_plan(194)[0] == 10 and chunk_plan(194)[2] == 200
    assert chunk_plan(200)[0] == 10
    assert chunk_plan(201)[0] == 11
    assert chunk_plan(1)[0] == 1
    n, idx, padded = chunk_plan(45)
    assert n == 3 and idx.tolist() == [0, 20, 40] and padded == 60


def test_chunk_actions_pads_with_zeros():
    a = np.ones((45, 10), dtype=np.float32)
    c = chunk_actions(a)
    assert c.shape == (3, 20, 10)
    assert c[2, :5].sum() == 50 and c[2, 5:].sum() == 0


def test_to_policy_image_undoes_horizontal_flip_only():
    raw = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)  # a raw (upside-down) render
    rotate180 = raw[::-1, ::-1]  # standard stored convention
    assert np.array_equal(to_policy_image(rotate180), raw[::-1])  # BPP expects vertical flip only
