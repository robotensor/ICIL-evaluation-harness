"""Export LIBERO hdf5 demonstrations to LeRobotDataset v3 (requires the ``lerobot`` extra).

Conventions (standard S2): ``observation.images.image`` = agentview, ``observation.images.image2``
= wrist camera, both rotated 180 degrees from the raw robosuite frame (the convention shared by
``lerobot/libero``, openpi and vla-eval); ``observation.state`` = [eef_pos(3), eef_axis_angle(3),
gripper_qpos(2)]; ``action`` = 7-D OSC_POSE delta in [-1, 1]; fps 20; one language ``task`` per
episode. Episodes are ordered by the numeric demo index (``demo_0, demo_1, ...``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from icil_eval.context.loaders.hdf5 import LiberoHdf5Store
from icil_eval.registry.schema import Task

if TYPE_CHECKING:  # pragma: no cover
    from icil_eval.providers.libero.provider import LiberoProvider

STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_ax",
    "eef_ay",
    "eef_az",
    "gripper_qpos_0",
    "gripper_qpos_1",
]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def dataset_card(task: Task, provider_version: Dict[str, str]) -> Dict[str, Any]:
    return {
        "icil_card_version": 1,
        "task_id": task.task_id,
        "language": task.language,
        "fps": 20,
        "camera_map": {"image": "agentview", "image2": "robot0_eye_in_hand"},
        "image_convention": "rotate180",
        "image_convention_note": (
            "frame[::-1, ::-1] of the raw robosuite render; "
            "matches lerobot/libero, openpi and vla-eval"
        ),
        "state_layout": STATE_NAMES,
        "action_layout": ACTION_NAMES,
        "action_space": "osc_pose_delta_7d in [-1, 1]; gripper -1 open / +1 close",
        "source": task.demo_pool.__dict__,
        "source_version": provider_version,
        "episode_index_is_numeric_demo_index": True,
        "note": (
            "fps differs from lerobot/libero (10, no-op filtered); "
            "this export keeps all 20 Hz frames"
        ),
    }


def export_task_demos(
    provider: LiberoProvider, task: Task, dest: Path, max_episodes: Optional[int] = None
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lerobot extra

    store = LiberoHdf5Store(provider.raw_root)
    n = store.n_episodes(task)
    if max_episodes is not None:
        n = min(n, max_episodes)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    repo_id = f"icil-eval/{task.provider}_{task.suite}_{task.stem}"
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=20, features=features, root=dest, robot_type="panda", use_videos=True
    )
    for i in range(n):
        demo = store.load(task, i)
        T = demo.action.shape[0]
        for t in range(T):
            ds.add_frame(
                {
                    "observation.images.image": demo.images["image"][t],
                    "observation.images.image2": demo.images["image2"][t],
                    "observation.state": demo.state[t].astype(np.float32),
                    "action": demo.action[t].astype(np.float32),
                    "next.reward": np.asarray([demo.rewards[t]], dtype=np.float32),
                    "next.done": np.asarray([t == T - 1], dtype=bool),
                    "task": task.language,
                }
            )
        ds.save_episode()
    ds.finalize()
    (dest / "meta" / "icil_card.json").write_text(
        json.dumps(dataset_card(task, provider.version()), indent=2)
    )
    return dest
