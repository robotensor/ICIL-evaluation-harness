"""Build Behavior Prompting Policy prompts from :class:`Demonstration` objects.

Reproduces ``PromptActionChunker.chunk_prompt(obs_predownsampled=True)`` with
``prompt_chunk_n_actions = 20`` and ``pad_end_prompt_actions = zeros``: one observation per 20
actions, actions grouped in chunks of 20, the trailing partial chunk zero-padded.

Chunk-planning math is pure numpy so it can be unit-tested without torch; tensor assembly imports
torch lazily (the wrapper runs inside BPP's environment).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from icil_eval.context.types import Demonstration

CHUNK_LEN = 20
ACTION_DIM = 10  # [pos(3), rot6d(6), gripper(1)]
IMAGE_SIZE = 224


def chunk_plan(length: int, chunk_len: int = CHUNK_LEN) -> Tuple[int, np.ndarray, int]:
    """Return ``(L, obs_indices, padded_length)`` for a demonstration of ``length`` steps."""
    n_chunks = max(1, math.ceil(length / chunk_len))
    obs_indices = np.arange(0, length, chunk_len)[:n_chunks]
    if len(obs_indices) < n_chunks:  # length == 0 edge case
        obs_indices = np.zeros(n_chunks, dtype=int)
    return n_chunks, obs_indices, n_chunks * chunk_len


def chunk_actions(actions10: np.ndarray, chunk_len: int = CHUNK_LEN) -> np.ndarray:
    """(T, 10) -> (L, chunk_len, 10) with zero padding of the last chunk."""
    n_chunks, _, padded = chunk_plan(actions10.shape[0], chunk_len)
    out = np.zeros((padded, actions10.shape[1]), dtype=np.float32)
    out[: actions10.shape[0]] = actions10
    return out.reshape(n_chunks, chunk_len, actions10.shape[1])


def to_policy_image(frame_rotate180: np.ndarray) -> np.ndarray:
    """Standard rotate180 frame -> BPP orientation (vertical flip of the raw render), HWC uint8."""
    return np.ascontiguousarray(frame_rotate180[:, ::-1])


class BppPromptBuilder:
    """Turns demonstrations into the prompt dict ``DiffusionUnetPolicy.prompt`` expects."""

    def __init__(
        self,
        axis_angle_to_rot6d: Callable[[np.ndarray], np.ndarray],
        image_size: int = IMAGE_SIZE,
        chunk_len: int = CHUNK_LEN,
        emulate_native: int = 0,
    ) -> None:
        self.axis_angle_to_rot6d = axis_angle_to_rot6d
        self.image_size = image_size
        self.chunk_len = chunk_len
        self.emulate_native = emulate_native

    # ------------------------------------------------------------------ per-demo numpy parts
    def demo_parts(self, demo: Demonstration) -> Dict[str, np.ndarray]:
        if demo.action is None or demo.state is None:
            raise ValueError("BPP prompts need sensorimotor demonstrations (state + action)")
        length = demo.length
        n_chunks, idx, _ = chunk_plan(length, self.chunk_len)
        idx = idx[idx < length]
        state = demo.state[idx].astype(np.float32)
        ee_pos = state[:, 0:3]
        ee_ori = self.axis_angle_to_rot6d(state[:, 3:6]).astype(np.float32)
        gripper = state[:, 6:8]
        actions = demo.action.astype(np.float32)
        actions10 = np.concatenate(
            [
                actions[:, 0:3],
                self.axis_angle_to_rot6d(actions[:, 3:6]).astype(np.float32),
                actions[:, 6:7],
            ],
            axis=1,
        )
        parts = {
            "agentview": np.stack([to_policy_image(f) for f in demo.images["image"][idx]]),
            "eye_in_hand": np.stack([to_policy_image(f) for f in demo.images["image2"][idx]]),
            "ee_pos": ee_pos,
            "ee_ori": ee_ori,
            "gripper_states": gripper,
            "action": chunk_actions(actions10, self.chunk_len),
        }
        assert parts["action"].shape[0] == n_chunks == len(idx)
        return parts

    def n_chunks(self, demos: Sequence[Demonstration]) -> int:
        return sum(chunk_plan(d.length, self.chunk_len)[0] for d in demos)

    # ------------------------------------------------------------------ torch assembly
    def images_to_tensor(self, frames_hwc: np.ndarray):
        """(L, H, W, 3) uint8 (BPP orientation) -> (1, L, 3, S, S) float32 in [0, 1]."""
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(frames_hwc)).permute(0, 3, 1, 2).float() / 255.0
        if self.emulate_native and x.shape[-1] != self.emulate_native:
            # emulate BPP's native 128 px render before its bilinear upsample to 224
            x = F.interpolate(x, size=(self.emulate_native, self.emulate_native), mode="area")
        if x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        return x.unsqueeze(0)

    def build(self, demos: Sequence[Demonstration], device: Any = "cpu") -> Dict[str, Any]:
        """Concatenate K demonstrations along the chunk axis into one prompt dict (B = 1)."""
        import torch

        if not demos:
            return self.blank(device)
        parts = [self.demo_parts(d) for d in demos]
        cat = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
        n_chunks = cat["action"].shape[0]
        obs = {
            "agentview_rgb": self.images_to_tensor(cat["agentview"]),
            "eye_in_hand_rgb": self.images_to_tensor(cat["eye_in_hand"]),
            "ee_pos": torch.from_numpy(cat["ee_pos"]).unsqueeze(0),
            "ee_ori": torch.from_numpy(cat["ee_ori"]).unsqueeze(0),
            "gripper_states": torch.from_numpy(cat["gripper_states"]).unsqueeze(0),
        }
        prompt = {
            "obs": {k: v.to(device) for k, v in obs.items()},
            "action": torch.from_numpy(cat["action"]).unsqueeze(0).to(device),
            "metadata": {"mask": torch.zeros((1, n_chunks), dtype=torch.bool, device=device)},
        }
        return prompt

    def blank(self, device: Any = "cpu") -> Dict[str, Any]:
        """K = 0: one all-zero, fully masked chunk (receding tokens attend only to sink tokens)."""
        import torch

        s = self.image_size
        obs = {
            "agentview_rgb": torch.zeros((1, 1, 3, s, s)),
            "eye_in_hand_rgb": torch.zeros((1, 1, 3, s, s)),
            "ee_pos": torch.zeros((1, 1, 3)),
            "ee_ori": torch.zeros((1, 1, 6)),
            "gripper_states": torch.zeros((1, 1, 2)),
        }
        return {
            "obs": {k: v.to(device) for k, v in obs.items()},
            "action": torch.zeros((1, 1, self.chunk_len, ACTION_DIM), device=device),
            "metadata": {"mask": torch.ones((1, 1), dtype=torch.bool, device=device)},
        }


def demo_lengths(demos: Sequence[Demonstration]) -> List[int]:
    return [d.length for d in demos]
