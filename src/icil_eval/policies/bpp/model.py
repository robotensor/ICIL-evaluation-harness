"""Shared Behavior Prompting Policy model: one checkpoint in memory, many concurrent sessions.

Runs inside BPP's own environment (``behavior_prompting`` importable). The prompt encoder keeps a
single cached prompt; sessions therefore encode their prompt once and swap the cached tokens in
under a lock before every action prediction.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from icil_eval.policies.bpp.prompt import ACTION_DIM, CHUNK_LEN, IMAGE_SIZE
from icil_eval.policy.protocol import PolicyCard, PolicySpec

TIMM_BACKBONE = "vit_base_patch16_clip_224.openai"


class BPPModel:
    """Loads a BPP checkpoint and exposes prompt encoding + action prediction."""

    def __init__(self, checkpoint: Path, device: str = "cuda", allow_network: bool = False) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        if not allow_network:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.lock = threading.Lock()
        t0 = time.monotonic()
        self._load()
        self.load_seconds = time.monotonic() - t0

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        import dill
        import hydra
        import torch
        from behavior_prompting.train_network.model.common.rotation_transformer import (
            RotationTransformer,
        )

        payload = torch.load(
            self.checkpoint, pickle_module=dill, map_location="cpu", weights_only=False, mmap=True
        )
        self.cfg = payload["cfg"]
        state = payload["state_dicts"]["model"]
        self.training_split_info: Dict[str, bool] = dict(
            state.get("_extra_training_split_info") or {}
        )
        self.epoch = None
        try:
            self.epoch = dill.loads(payload["pickles"]["epoch"])
        except Exception:
            pass
        self._apply_checkpoint_compat(state)
        self.policy = hydra.utils.instantiate(self.cfg.model)
        self.policy.load_state_dict(state)
        self.policy.eval().to(self.device)
        del payload
        self.encoder = self.policy.obs_encoder
        self.rot_axis_angle_to_6d = RotationTransformer("axis_angle", "rotation_6d")
        sm = (
            self.cfg.model.shape_meta
            if "shape_meta" in self.cfg.model
            else self.cfg.task.shape_meta
        )
        self.shape_meta = sm
        self.chunk_len = int(sm.get("prompt_chunk_n_actions", CHUNK_LEN))
        self.image_size = int(sm.get("image_resolution", IMAGE_SIZE))
        self.action_horizon = int(sm["action"]["horizon"])
        self.exec_horizon = int(self.cfg.task.env_runner.get("exec_action_horizon", 12))
        self.prompt_budget_chunks = int(self.encoder.prompt_pos_emb.shape[1])
        self.attention_sink = bool(getattr(self.encoder, "attention_sink_enabled", False))
        self.obs_history = int(sm["obs"]["agentview_rgb"]["horizon"])

    def _apply_checkpoint_compat(self, state: Dict[str, Any]) -> None:
        """Align encoder flags whose defaults changed after the checkpoint was trained.

        ``PairPromptObsEncoder.use_pool_modality_pos_embed`` defaults to True in current BPP code
        but the released checkpoints predate the parameter; instantiating with the default would
        create ``obs_encoder.obs_encoder.pool_modality_pos_embed`` and fail a strict load.
        """
        from omegaconf import OmegaConf

        has_pool = any(k.endswith("pool_modality_pos_embed") for k in state)
        enc = self.cfg.model.obs_encoder
        if "use_pool_modality_pos_embed" not in enc and not has_pool:
            OmegaConf.set_struct(self.cfg, False)
            enc["use_pool_modality_pos_embed"] = False
            self.compat_overrides = {"obs_encoder.use_pool_modality_pos_embed": False}
        else:
            self.compat_overrides = {}

    # ------------------------------------------------------------------ capability
    @property
    def training_tasks(self) -> List[str]:
        return sorted({k.split(":", 1)[1] for k in self.training_split_info})

    def training_episodes(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for key, used in self.training_split_info.items():
            episode, task = key.split(":", 1)
            try:
                idx = int(episode.rsplit("demo_", 1)[1])
            except (IndexError, ValueError):
                continue
            if used:
                out.setdefault(task, []).append(idx)
        return {k: sorted(v) for k, v in out.items()}

    def spec(self) -> PolicySpec:
        card = PolicyCard(
            name="behavior_prompting_policy",
            version=f"{self.checkpoint.name}@epoch{self.epoch}",
            exposure_source="checkpoint_metadata",
            training_data=[
                {
                    "repo_id": "yifengzhu-hf/LIBERO-datasets",
                    "description": "LIBERO demonstrations; per-demo membership in ckpt metadata",
                    "episodes": "see training_episodes",
                }
            ],
            training_tasks=self.training_tasks,
            training_episodes=self.training_episodes(),
            url="https://github.com/real-stanford/behavior_prompting",
            license="MIT",
            notes=(
                "Task identity in the checkpoint is the instruction string "
                "(ambiguous across LIBERO-90 scenes)."
            ),
        )
        return PolicySpec(
            name="bpp",
            context_mode="sequence",
            k_min=0 if self.attention_sink else 1,
            k_max=None,
            context_budget={"chunks": self.prompt_budget_chunks, "chunk_len": self.chunk_len},
            needs_every_observation=True,
            order_invariant=False,
            requires_language=False,
            k0_semantics="blank_prompt" if self.attention_sink else "none",
            cameras=["image", "image2"],
            image_size=self.image_size,
            state_layout="eef_pos3_axisangle3_gripper2",
            action_space="osc_pose_delta_7d",
            action_horizon=self.action_horizon,
            exec_horizon=self.exec_horizon,
            control_hz=20.0,
            card=card,
            extra={
                "checkpoint": str(self.checkpoint),
                "trained_with_k": int(self.shape_meta.get("max_prompt_full_demos", 1)),
                "prompt_sample_mode": str(self.shape_meta.get("prompt_sample_mode", "pair")),
                "obs_history": self.obs_history,
                "device": self.device,
            },
        )

    # ------------------------------------------------------------------ inference
    def encode_prompt(self, prompt: Dict[str, Any]) -> Tuple[Any, Any]:
        """Run the prompt encoder once; return (tokens, mask) detached from the shared cache."""
        import torch

        with self.lock, torch.inference_mode():
            self.encoder.reset()
            self.policy.prompt(prompt)
            tokens = self.encoder.prompt_tokens_cache.clone()
            mask = (
                self.encoder.prompt_mask_cache.clone()
                if self.encoder.prompt_mask_cache is not None
                else None
            )
            self.encoder.reset()
        return tokens, mask

    def predict(self, obs: Dict[str, Any], tokens: Any, mask: Any) -> np.ndarray:
        """Predict an action chunk (H, 7) in LIBERO's 7-D OSC_POSE space for one session."""
        import torch

        with self.lock, torch.inference_mode():
            self.encoder.prompt_tokens_cache = tokens
            self.encoder.prompt_mask_cache = mask
            self.encoder.is_prompted = True
            out = self.policy.predict_action(obs)
            action10 = out["action"][0].detach().float().cpu().numpy()
            self.encoder.reset()
        assert action10.shape[-1] == ACTION_DIM, action10.shape
        rot = self.rot_axis_angle_to_6d.inverse(action10[:, 3:9].astype(np.float64)).astype(
            np.float32
        )
        return np.concatenate([action10[:, 0:3], rot, action10[:, 9:10]], axis=1).astype(np.float32)

    def axis_angle_to_rot6d(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.rot_axis_angle_to_6d.forward(np.asarray(x, dtype=np.float64)), dtype=np.float32
        )
