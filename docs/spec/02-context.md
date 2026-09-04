# S2 — Demonstrations as context

version: 1

## 1. Demonstration storage

Demonstrations are LeRobotDataset v3 episodes (one dataset per task in v1) with LeRobot keys:

| key | LIBERO value |
|---|---|
| `observation.images.image` | main camera (agentview), 128×128×3, `rotate180` convention |
| `observation.images.image2` | wrist camera (robot0_eye_in_hand), same convention |
| `observation.state` | float32[8] = `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]` |
| `action` | float32[7] OSC_POSE delta in [-1, 1], gripper −1 open / +1 close |
| `task` | the instruction string, per frame |
| dataset `fps` | 20 (all frames kept; `lerobot/libero` on the Hub is 10-labelled and no-op filtered) |

`image_convention: rotate180` = `frame[::-1, ::-1]` of the raw robosuite render, the convention
shared by `lerobot/libero`, openpi and vla-eval. Each dataset carries `meta/icil_card.json` with
`camera_map`, `image_convention`, `state_layout`, `action_layout`, `source` (repo, revision, path,
sha256) and `source_version`. The flattened simulator state is not stored (its dimension varies
per task); it can be re-read from the source file via `source`.

In memory a `Demonstration` is `{images {cam: (T,H,W,3) uint8}, state (T,S), action (T,A), fps,
task, robot_type, source, rewards?, init_state?}`. `state`/`action` are required only for the
`sensorimotor` modality.

## 2. Context reference and spec

```
ContextRef  = (task_id, episode_index)
ContextSpec = {track, query_task_id, condition, k, refs[], relation, context_task_id,
               transform, language, modality, seed, episode_idx, registry_hash,
               permutation_seed, context_hash}
```

- `relation` names how the context task relates to the query (S1 §4); for wrong-task controls it
  is the relation actually used (`other_task_same_layout` or the fallback), recorded per row.
- `transform ∈ {identity, shuffled_chunks, reversed_actions}`.
- `language ∈ {none, context, query}`: `none` means the policy receives no instruction at all
  (the server strips it), so demo-only evaluation is actually demo-only.
- `context_hash` = hash of `(refs, transform, language, permutation_seed, registry_hash)`.

## 3. Conditions and the K sweep

Conditions are named `k<K>` or `k<K>.<control>`; `kmax` resolves to the largest K the policy
supports. Every track uses the sweep `K ∈ {0, 1, 2, 4, 8}` and the conditions
`k0, k1, k2, k4, k8, k1.wrong_task, k1.shuffled_chunks, kmax.wrong_task`. `k0` is the only
zero-context condition. A policy that cannot accept a condition reports `unsupported`; contexts
are never truncated.

## 4. Deterministic sampling

```
rng = Random(sha256(registry_hash | seed | track | task_id | k | relation | transform |
                    language | control | episode_idx))
context task  = rng.choice(pool.context)          (wrong_task: rng.choice(pool.wrong))
episode idxs  = rng.sample(range(n_demos), k)
permutation_seed = rng.getrandbits(32)           (transforms only)
```

The seed is hash-derived, so subsetting or reordering tasks never changes any exemplar, and both
the benchmark side and the policy side recompute the identical spec from the same flat inputs.
The evaluated initial state is never a context demonstration's initial state; the loader records
`min_context_init_l2` and flags a collision instead of resampling.

## 5. Transforms

- `shuffled_chunks`: split the demonstration into contiguous chunks of
  `context_chunk_seconds` (0.5 s = 10 frames at 20 Hz) and permute them jointly for images,
  state, action and rewards with `Random(permutation_seed)`; identity permutations are rejected;
  `permutation_hash` is recorded.
- `reversed_actions`: reverse the action sequence in time, observations unchanged.
