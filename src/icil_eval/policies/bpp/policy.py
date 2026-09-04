"""Per-session Behavior Prompting Policy view implementing the ICIL policy protocol."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from icil_eval.context.types import Demonstration, UnsupportedContext
from icil_eval.policies.bpp.model import BPPModel
from icil_eval.policies.bpp.prompt import BppPromptBuilder, to_policy_image
from icil_eval.policy.protocol import ActionChunk, ContextInfo, ICILPolicy, Observation, TaskInfo


class BPPPolicy(ICILPolicy):
    """One evaluation session: holds the encoded context and a 2-frame observation history."""

    def __init__(self, model: BPPModel, emulate_native: int = 128) -> None:
        self.model = model
        self.spec = model.spec()
        self.builder = BppPromptBuilder(
            axis_angle_to_rot6d=model.axis_angle_to_rot6d,
            image_size=model.image_size,
            chunk_len=model.chunk_len,
            emulate_native=emulate_native,
        )
        self._tokens: Any = None
        self._mask: Any = None
        self._history: Deque[Dict[str, np.ndarray]] = deque(maxlen=model.obs_history)
        self._steps = 0

    # ------------------------------------------------------------------ protocol
    def reset(self) -> None:
        self._tokens = None
        self._mask = None
        self._history.clear()
        self._steps = 0

    def set_context(self, demos: List[Demonstration], task: Optional[TaskInfo]) -> ContextInfo:
        t0 = time.monotonic()
        if demos:
            n_chunks = self.builder.n_chunks(demos)
            if n_chunks > self.model.prompt_budget_chunks:
                raise UnsupportedContext(
                    f"{len(demos)} demonstrations need {n_chunks} prompt chunks; "
                    f"checkpoint budget is {self.model.prompt_budget_chunks}"
                )
            prompt = self.builder.build(demos, device=self.model.device)
        else:
            if not self.model.attention_sink:
                raise UnsupportedContext("K=0 requires attention-sink tokens in the checkpoint")
            n_chunks = 1
            prompt = self.builder.blank(device=self.model.device)
        self._tokens, self._mask = self.model.encode_prompt(prompt)
        return ContextInfo(
            n_demos=len(demos),
            n_frames=sum(d.length for d in demos),
            adaptation_latency_s=time.monotonic() - t0,
            chunks_used=n_chunks,
            extra={"k0_semantics": "blank_prompt" if not demos else None},
        )

    def observe(self, obs: Observation, executed_actions: Optional[np.ndarray]) -> None:
        if obs.state is None:
            raise ValueError("BPP needs proprioceptive state (send_state=True)")
        frame = {
            "agentview": to_policy_image(obs.images["image"]),
            "eye_in_hand": to_policy_image(obs.images["image2"]),
            "ee_pos": np.asarray(obs.state[0:3], dtype=np.float32),
            "ee_ori": self.model.axis_angle_to_rot6d(np.asarray(obs.state[3:6])[None])[0],
            "gripper_states": np.asarray(obs.state[6:8], dtype=np.float32),
        }
        self._history.append(frame)
        self._steps += 1

    def act(self) -> ActionChunk:
        if self._tokens is None:
            raise RuntimeError("set_context() must be called before act()")
        if not self._history:
            raise RuntimeError("observe() must be called before act()")
        import torch

        frames = list(self._history)
        while len(frames) < self._history.maxlen:  # first step: duplicate the only frame
            frames.insert(0, frames[0])
        obs = {
            "agentview_rgb": self.builder.images_to_tensor(
                np.stack([f["agentview"] for f in frames])
            ),
            "eye_in_hand_rgb": self.builder.images_to_tensor(
                np.stack([f["eye_in_hand"] for f in frames])
            ),
            "ee_pos": torch.from_numpy(np.stack([f["ee_pos"] for f in frames])).unsqueeze(0),
            "ee_ori": torch.from_numpy(np.stack([f["ee_ori"] for f in frames])).unsqueeze(0),
            "gripper_states": torch.from_numpy(
                np.stack([f["gripper_states"] for f in frames])
            ).unsqueeze(0),
        }
        obs = {k: v.to(self.model.device) for k, v in obs.items()}
        actions7 = self.model.predict(obs, self._tokens, self._mask)
        return ActionChunk(actions=actions7, exec_horizon=self.model.exec_horizon)
