# Policy: Behavior Prompting Policy (BPP)

Reference in-context policy: [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting)
(MIT), checkpoint `austinpatel/libero` (`libero_behavior_prompting.ckpt`, trained on all 130 LIBERO
tasks, 6.9 GB) or `austinpatel/liberogen_*`.

## Serving

```bash
# inside BPP's conda env (behavior_prompting importable, torch + CUDA)
pip install -e ".[server,bpp]"
icil-eval serve bpp --ckpt <ckpt> --port 8000 --device cuda \
  [--emulate-native 128] [--track configuration] [--condition k1] [--demo-root ~/.cache/icil-eval/raw/libero]
```

The server loads the checkpoint once (~20 s, mmap), writes `<log>.meta.json` with the policy spec
and card, and serves any number of vla-eval shards. Requires the timm backbone
`vit_base_patch16_clip_224.openai` in the local Hugging Face cache (offline mode is forced).

## Conversion table (LIBERO via vla-eval → BPP)

| input | vla-eval sends | BPP receives |
|---|---|---|
| images | `images.agentview`, `images.wrist`, HWC uint8, rotate180 | horizontal flip (`[:, ::-1]`), optional area-downsample to 128, bilinear to 224, CHW float |
| state | `states[8] = [eef_pos, eef_axis_angle, gripper_qpos]` | `ee_pos = state[0:3]`, `ee_ori = rot6d(state[3:6])`, `gripper_states = state[6:8]` |
| history | one observation per control step | 2-frame deque (first frame duplicated) |
| actions | expects 7-D `[dpos, drot_axis_angle, gripper]` | `(16, 10)` → rot6d→axis-angle → first 12 executed |

Prompt: per demonstration `L = max(1, ceil(T/20))` chunks (one observation per 20 actions,
zero-padded tail); K demonstrations are chunked independently and concatenated. Budget = 50
chunks (`prompt_pos_emb`); with LIBERO demonstrations of 6–14 chunks, K=2 and mostly K=4 fit,
K=8 fits for a few short tasks only (`icil-eval capabilities`). K>1 is out of distribution for this
checkpoint (trained with exactly one demonstration) and is reported as such. K=0 is a single
all-zero, fully masked chunk (the encoder attends only to its 4 attention-sink tokens).

## Exposure

The checkpoint stores per-demonstration training membership keyed by instruction string
(`_extra_training_split_info`), which the server publishes in the policy card; the scorer tags
every row `query_exposure = seen` for this checkpoint. Its ICIL evidence is the wrong-context
gap, not raw success.

## Reference numbers

Quick preset, `configuration` track, libero_goal (10 tasks × 10 initial states per condition,
128 px render, one L40S, 6 vla-eval shards, 2026-09-04):

| condition | success [Wilson 95%] | n |
|---|---|---|
| `k0` (blank prompt) | 8.0% [4.1%, 15.0%] | 100 |
| `k1` | 97.0% [91.5%, 99.0%] | 100 |
| `k2` | 97.0% [91.5%, 99.0%] | 100 |
| `k4` | 98.8% [93.3%, 99.8%] | 80 (8 tasks) |
| `k8` | 100.0% [88.6%, 100.0%] | 30 (3 tasks) |
| `k1.shuffled_chunks` | 88.0% [80.2%, 93.0%] | 100 |
| `k1.wrong_task` | 0.0% [0.0%, 3.7%] | 100 |
| `k{2,4,8}.wrong_task` | 0.0% | 20 / 50 / 20 (+10 unsupported) |

Δ_context@1 = +97.0 pp (paired, n=100, 97 discordant, McNemar p ≈ 1e-29); order sensitivity@1 =
+9.0 pp (p = 0.012); context AUC (mean SR over K ∈ {0,1,2,4,8}) = 80.2%; adaptation ≈ 0.04 s
(0.15 s at K=8), ≈ 0.13 s per 16-step action chunk. All rows `query_exposure = seen`. K>1 is out
of distribution for this single-demonstration checkpoint yet does not hurt; the wrong-context gap
is the ICIL evidence. Native BPP reference on LIBERO-Gen spatial-combination (BPP's own runner):
correct demo 0.776 vs wrong demo 0.124 over 10 held-out tasks × 50 episodes.
