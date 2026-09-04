"""Load demonstrations from a LeRobotDataset v3 (requires the ``lerobot`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from icil_eval.context.types import Demonstration


class LeRobotStore:
    """One LeRobotDataset per task, under ``root/<task_id with '/'->'__'>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._datasets: Dict[str, object] = {}

    def dataset_dir(self, task_id: str) -> Path:
        return self.root / task_id.replace("/", "__")

    def available(self, task_id: str) -> bool:
        return (self.dataset_dir(task_id) / "meta" / "info.json").exists()

    def _dataset(self, task_id: str):
        if task_id not in self._datasets:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._datasets[task_id] = LeRobotDataset(
                repo_id=f"icil-eval/{task_id.replace('/', '_')}", root=self.dataset_dir(task_id)
            )
        return self._datasets[task_id]

    def n_episodes(self, task_id: str) -> int:
        return int(self._dataset(task_id).meta.total_episodes)

    def load(self, task_id: str, episode_index: int, fps: Optional[float] = None) -> Demonstration:
        ds = self._dataset(task_id)
        ep = ds.meta.episodes[episode_index]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        frames = [ds[i] for i in range(start, end)]
        images: Dict[str, np.ndarray] = {}
        for key in ds.meta.video_keys + ds.meta.image_keys:
            cam = key.split(".")[-1]
            stack = np.stack([np.asarray(f[key]) for f in frames])  # (T, C, H, W) float in [0, 1]
            images[cam] = (np.moveaxis(stack, 1, -1) * 255.0).round().astype(np.uint8)
        state = np.stack([np.asarray(f["observation.state"]) for f in frames]).astype(np.float32)
        action = np.stack([np.asarray(f["action"]) for f in frames]).astype(np.float32)
        task = frames[0].get("task") if frames else None
        return Demonstration(
            images=images,
            state=state,
            action=action,
            fps=float(fps or ds.meta.fps),
            task=task if isinstance(task, str) else None,
            robot_type=ds.meta.robot_type,
            source={
                "format": "lerobot_v3",
                "root": str(self.dataset_dir(task_id)),
                "episode_index": episode_index,
                "task_id": task_id,
            },
        )
