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

Smoke preset (libero_goal, 2 tasks × 5 initial states, 128 px): `k1` 10/10, `k1.wrong_task` 0/10,
Δ_context@1 = +100 pp (McNemar p = 0.002); adaptation ≈ 0.03 s (0.5 s first call), ≈ 0.10 s per
16-step action chunk on an L40S. Native BPP reference on LIBERO-Gen spatial-combination
(BPP's own runner): correct demo 0.776 vs wrong demo 0.124 over 10 held-out tasks × 50 episodes.
