from __future__ import annotations

import pytest

from icil_eval.policy import (
    PolicyCard,
    PolicySpec,
    UnsupportedContext,
    chunks_needed,
    resolve_k_max,
    supports_k,
)


def spec(chunks=50, chunk_len=20, k_min=0):
    return PolicySpec(
        name="fake",
        context_mode="sequence",
        k_min=k_min,
        k_max=None,
        context_budget={"chunks": chunks, "chunk_len": chunk_len},
        needs_every_observation=True,
        order_invariant=False,
        requires_language=False,
        k0_semantics="blank_prompt",
        cameras=["image", "image2"],
        action_space="osc_pose_delta_7d",
        action_horizon=16,
        exec_horizon=12,
        control_hz=20.0,
        card=PolicyCard(name="fake", version="0", exposure_source="undisclosed"),
    )


def test_chunks_needed():
    assert chunks_needed([1, 20, 21, 200], 20) == 1 + 1 + 2 + 10
    assert chunks_needed([], 20) == 0


def test_resolve_k_max_uses_worst_case_lengths():
    lengths = [140] * 45 + [196] * 5  # 7 chunks each, longest 10 chunks
    s = spec(chunks=50)
    # 5 x 10 = 50 fits, so K=4 (40) fits; K=8 = 5x10 + 3x7 = 71 > 50
    assert resolve_k_max(s, [0, 1, 2, 4, 8], lengths) == 4
    assert supports_k(s, 8, lengths) is False
    assert supports_k(s, 0, lengths) is True


def test_resolve_k_max_respects_k_min_and_no_support():
    s = spec(chunks=5, k_min=1)
    assert resolve_k_max(s, [0, 1, 2], [60, 60]) == 1  # 3 chunks fits, 6 does not
    with pytest.raises(UnsupportedContext):
        resolve_k_max(s, [1, 2], [200, 200])


def test_spec_round_trips_to_dict():
    d = spec().to_dict()
    assert d["context_budget"] == {"chunks": 50, "chunk_len": 20}
    assert d["card"]["exposure_source"] == "undisclosed"
