# Provider: LIBERO

Source: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) at
commit `8f1084e` (the commit pinned by vla-eval's LIBERO image); code MIT. Demonstrations:
[yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)
(Apache-2.0), 50 teleoperated demonstrations per task, 20 Hz, 128×128 agentview + wrist camera,
revision recorded in the registry with each file's sha256.

## Tasks (130)

| suite | tasks | layout | `max_steps` | notes |
|---|---|---|---|---|
| `libero_goal` | 10 | one shared layout and initial state | 300 | ten goals in one scene: the demonstration alone selects the goal |
| `libero_spatial` | 10 | one layout, ten initial placements | 220 | two black bowls; goal always "bowl_1 on plate"; `chance_success = 0.5` |
| `libero_object` | 10 | one scene per task | 280 | "pick up X and place it in the basket" |
| `libero_10` | 10 | 9 scenes | 520 | long-horizon; two-predicate goals → `progress` |
| `libero_90` | 90 | 20 named scenes (KITCHEN/LIVING_ROOM/STUDY) | 400 | 12 instructions recur across scenes → `scene` track |

`language` is LIBERO's filename-derived instruction (what vla-eval, LeRobot and the BPP
checkpoints use); the BDDL `:language` text, which differs for 29 tasks, is kept as
`extra.bddl_language`. Evaluation initial states are the 50 `.pruned_init` states per task; they
are disjoint from demonstration start states (minimum L2 distance ≈ 0.29 on libero_goal and
libero_spatial), so the "evaluated initial state is never a context demonstration" rule holds by
construction and is still verified per episode (`min_context_init_l2`).

## Tracks contributed

- `configuration` — all 130 tasks. Wrong-task pools: `other_task_same_layout` for 120 tasks,
  `other_task_same_suite` for the remaining 10.
- `scene` — 31 query tasks in 13 instruction groups (29 from libero_90, plus
  `libero_goal/turn_on_the_stove` and one libero_10 task whose instruction recurs in libero_90).

## Backend binding

vla-eval `LIBEROBenchmark` (via `ICILLIBEROBenchmark`, render resolution configurable; 128 = BPP's
native training render, 256 = vla-eval default). Settle steps 10, `env_seed` 7, gripper
discretised `<0 → −1`.

## Rebuilding the registry

```bash
icil-eval registry build libero -o libero_root=/path/to/LIBERO/libero/libero \
  -o raw_root=~/.cache/icil-eval/raw/libero          # hashes come from the Hub API; add -o offline=true to skip
icil-eval registry validate
```

## Exporting demonstrations to LeRobotDataset v3

```bash
uv venv --python 3.12 .venv-lerobot && uv pip install --python .venv-lerobot/bin/python -e ".[lerobot]"
.venv-lerobot/bin/python -c "
from icil_eval.registry.io import load_registry; from icil_eval.providers.libero.provider import LiberoProvider
reg = load_registry(); p = LiberoProvider(libero_root='/path/to/LIBERO/libero/libero', offline=True)
t = reg.tasks['libero/libero_goal/turn_on_the_stove']; p.export_demos(t, '~/.cache/icil-eval/datasets/' + t.task_id.replace('/', '__'))"
```
Write to a regular filesystem (pyarrow fails on some tmpfs/FUSE mounts).
