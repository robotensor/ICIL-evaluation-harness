"""Load LIBERO raw hdf5 demonstrations as :class:`Demonstration` (no lerobot needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from icil_eval.context.types import Demonstration
from icil_eval.registry.schema import Task


def _demo_keys(f) -> List[str]:
    return sorted(
        (k for k in f["data"].keys() if k.startswith("demo_")), key=lambda s: int(s.split("_")[1])
    )


def rotate180(frames: np.ndarray) -> np.ndarray:
    """Raw robosuite frames are stored upside-down (OpenGL); rotate to the standard convention."""
    return np.ascontiguousarray(frames[:, ::-1, ::-1])


class LiberoHdf5Store:
    """Demonstration pool of LIBERO tasks stored as the original hdf5 files under ``raw_root``."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = Path(raw_root)
        self._lengths: Dict[str, List[int]] = {}

    def path_for(self, task: Task) -> Path:
        return self.raw_root / task.demo_pool.path

    def available(self, task: Task) -> bool:
        return self.path_for(task).exists()

    def n_episodes(self, task: Task) -> int:
        return len(self.episode_lengths(task))

    def episode_lengths(self, task: Task) -> List[int]:
        if task.task_id not in self._lengths:
            import h5py

            with h5py.File(self.path_for(task), "r") as f:
                self._lengths[task.task_id] = [
                    int(f["data"][k].attrs["num_samples"]) for k in _demo_keys(f)
                ]
        return self._lengths[task.task_id]

    def episode_init_state(self, task: Task, episode_index: int) -> np.ndarray:
        import h5py

        with h5py.File(self.path_for(task), "r") as f:
            g = f["data"][_demo_keys(f)[episode_index]]
            return np.asarray(g.attrs["init_state"], dtype=np.float64)

    def load(self, task: Task, episode_index: int, with_images: bool = True) -> Demonstration:
        import h5py

        path = self.path_for(task)
        with h5py.File(path, "r") as f:
            g = f["data"][_demo_keys(f)[episode_index]]
            obs = g["obs"]
            images: Dict[str, np.ndarray] = {}
            if with_images:
                images["image"] = rotate180(np.asarray(obs["agentview_rgb"], dtype=np.uint8))
                images["image2"] = rotate180(np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8))
            state = np.concatenate(
                [
                    np.asarray(obs["ee_pos"]),
                    np.asarray(obs["ee_ori"]),
                    np.asarray(obs["gripper_states"]),
                ],
                axis=1,
            ).astype(np.float32)
            action = np.asarray(g["actions"], dtype=np.float32)
            rewards = np.asarray(g["rewards"], dtype=np.float32) if "rewards" in g else None
            init_state: Optional[np.ndarray] = (
                np.asarray(g.attrs["init_state"], dtype=np.float64)
                if "init_state" in g.attrs
                else None
            )
        return Demonstration(
            images=images,
            state=state,
            action=action,
            fps=20.0,
            task=task.language,
            robot_type="panda",
            source={
                "format": "libero_hdf5",
                "path": str(path),
                "episode_index": episode_index,
                "task_id": task.task_id,
            },
            rewards=rewards,
            init_state=init_state,
        )
