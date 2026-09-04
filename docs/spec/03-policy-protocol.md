# S3 — The ICIL policy protocol

version: 1

## 1. Protocol

```python
class ICILPolicy:
    spec: PolicySpec
    def reset(self) -> None
    def set_context(self, demos: list[Demonstration], task: TaskInfo | None) -> ContextInfo
    def observe(self, obs: Observation, executed_actions: np.ndarray | None) -> None
    def act(self) -> ActionChunk
```

Per episode the harness calls `reset()`, then `set_context(...)` exactly once, then `observe(obs)`
on **every** control step, and `act()` whenever it needs a new action chunk (every
`spec.exec_horizon` steps). `observe` on every step is what history and autoregressive policies
need (BPP's 2-frame history, ICRT, RoboSSM); a stateless policy may ignore it and read the last
observation in `act()`.

- `set_context` must raise `UnsupportedContext` instead of truncating or dropping demonstrations.
  `ContextInfo` reports `n_demos`, `n_frames`, `chunks_used` and `adaptation_latency_s`.
- `TaskInfo.language` is `None` under `language: none`; policies that need an instruction declare
  `requires_language: true` and are reported `unsupported` on demo-only tracks.
- `ActionChunk.actions` is `(H, A)` in the provider's action space; the harness executes the first
  `exec_horizon` actions, then calls `act()` again.
- `executed_actions` (optional) carries the actions actually executed since the previous `act()`;
  the reference server passes `None` in v1.

## 2. PolicySpec

| field | meaning |
|---|---|
| `context_mode` | `sequence` (encodes the ordered context) or `retrieval` (owns a demonstration buffer) |
| `k_min`, `k_max` | hard bounds on the number of demonstrations (`k_max=None` = budget-limited only) |
| `context_budget` | `{chunks, chunk_len}` \| `{frames}` \| `{seconds}` \| `{}` |
| `needs_every_observation` | whether `observe` must be called each control step |
| `order_invariant` | retrieval-style policies: order controls are not meaningful |
| `requires_language` | needs an instruction to act |
| `k0_semantics` | how K=0 is realised: `none` (unsupported), `blank_prompt`, `native` |
| `cameras`, `image_size`, `state_layout` | inputs the policy consumes |
| `action_space`, `action_horizon`, `exec_horizon`, `control_hz` | outputs |
| `context_modalities`, `embodiments` | applicability (`not_applicable` when a task's embodiment is not listed) |
| `card` | the policy card (§4) |

### Supported K per task

`k_max(task) = max{K in sweep : the K longest demonstrations of the task's pool fit the budget}`.
Using the worst case makes every episode of a condition either fully supported or reported
`unsupported`, never a mix. `kmax.<control>` conditions resolve against this value.

## 3. Reference wrapper: Behavior Prompting Policy

- Load: `torch.load(ckpt, pickle_module=dill, mmap=True, weights_only=False)`; instantiate from the
  checkpoint's own fully resolved config; the released checkpoints predate the encoder flag
  `use_pool_modality_pos_embed` (set to `false` when the state dict has no such parameter).
  Offline by default (`HF_HUB_OFFLINE=1`; the timm CLIP backbone must be in the local cache).
- Prompt: one observation per 20 actions, actions in chunks of 20 with zero-padded tail
  (`L = max(1, ceil(T/20))`); actions converted to `[pos(3), rot6d(6), gripper(1)]`; images to the
  policy's orientation (vertical flip of the raw render = horizontal flip of the `rotate180`
  standard frame), 224×224 bilinear. K demonstrations are chunked independently and concatenated;
  the budget is the checkpoint's `prompt_pos_emb` length (50 chunks). K=0 is one all-zero,
  fully masked chunk (`blank_prompt`), valid because the checkpoints use attention-sink tokens.
- Inference: 2-frame observation history (first frame duplicated), 16 predicted actions,
  `exec_horizon` 12, fp32. Many sessions share one model: each session caches its encoded prompt
  tokens and swaps them into the encoder under a lock.
- Exposure: `_extra_training_split_info` in the checkpoint gives per-demonstration training
  membership keyed by instruction string; LIBERO-90 reuses instructions across scenes, so
  ambiguous tasks are tagged `seen` only when every candidate task is seen.

## 4. Policy card

```yaml
name, version, url, license
exposure_source: checkpoint_metadata | self_reported | undisclosed
training_data: [{repo_id, revision, episodes: all | [ids] | undisclosed, description}]
training_tasks: [task ids or instruction strings] | null   # null = undisclosed
training_episodes: {task: [demo indices]}                  # when known
notes
```

## 5. Server bridge (vla-eval backend)

`ICILModelServer(PredictModelServer)`: the `EPISODE_START` payload's flat task fields
(`icil_task_id, icil_track, icil_condition, icil_seed, icil_registry_hash, episode_idx`) are
enough to recompute the context spec with the shared sampler; demonstrations are loaded locally,
transformed, and passed to `set_context`. `task_description` is stripped under `language: none`.
`predict` returns exactly `exec_horizon` actions; unsupported or failed sessions return a hold
action and are recorded as such. One JSONL row per episode (`episode_id`, task fields, context
spec, status, adaptation and control latency) is written for the scorer, which joins it to the
backend's results by `episode_id`.
