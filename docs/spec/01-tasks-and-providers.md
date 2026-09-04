# S1 — Unified task schema, providers and the registry

version: 1

## 1. Purpose

The registry is the union of tasks contributed by existing benchmarks, expressed in one schema
under one namespace so that tracks, sampling, scoring and coverage work identically for every
source benchmark. It is committed as data (`src/icil_eval/data/registry/`), built by
`icil-eval registry build <provider>`, validated by `icil-eval registry validate`, and versioned
by a content hash (`registry_hash`) that every result row records.

## 2. Task identity

```
<provider>/<suite>/<stem>            e.g. libero/libero_goal/turn_on_the_stove
```

- `provider`: lowercase `[a-z0-9_]+`; `suite`: the provider's own grouping; `stem`: the
  provider's stable task name (LIBERO: the BDDL file stem). Never a positional index: LIBERO's
  `task_order_index` permutes indices, vla-eval's `task_id` is positional — neither is stable.
- Ids are ASCII, unique, and sorted within each tasks file.

## 3. Task fields

| field | meaning |
|---|---|
| `language` | the benchmark's instruction string (kept verbatim; normalised for relations) |
| `embodiment` | `{robot, gripper, dof, action_space, control_hz}`; used for policy applicability |
| `scene.layout_id` | `sha256(fixtures, objects, regions)[:12]` — what "same scene/layout" means |
| `scene.init_id` | `sha256(init predicates)[:12]` — the declared initial configuration |
| `scene.scene_name` | provider scene label when it exists (LIBERO-100: `KITCHEN_SCENE3`) |
| `scene.entities` | every object, fixture and `<target>_<region>` name the goal may refer to |
| `objects[]` | `{name, category, role ∈ target|fixture|distractor}` |
| `skills[]` | primitive tags derived from goal predicates (`place_on`, `place_in`, `open`, `turn_on`, …) |
| `goal.predicates` | normalised goal predicates `[[pred, arg, ...], ...]` (predicate lowercased) |
| `goal.n_effective_predicates` | goal predicates not already satisfied in the initial state |
| `horizon.max_steps`, `n_steps_wait` | pinned episode cap and settle steps (LIBERO: vla-eval / OpenVLA values) |
| `init_states` | `{file, n, sha256, verified}` — the canonical evaluation initial states |
| `demo_pool` | `{repo_id, revision, path, sha256, size_bytes, n, format, lerobot_repo_id}` |
| `bounds` | `{chance_success, expert_success}` normalisation bounds when known |
| `capabilities` | `{relations[], prompt_modalities[], perturbations[], progress_metric}` |
| `source_version` | pinned upstream versions (commit, simulator versions) |

`chance_success` is derived when the goal is a single move predicate whose moved object has
`n > 1` same-category instances in the scene (`1/n`; LIBERO-Spatial: 0.5). `progress_metric` is
true when `n_effective_predicates ≥ 2`.

## 4. Relations (context ↔ query)

Relations are functions over the schema, so any provider that fills the fields correctly
contributes to every track automatically:

| relation | candidates |
|---|---|
| `same_task` | the query task itself |
| `same_instruction_other_layout` | same normalised language, different `layout_id`, same provider (may span suites) |
| `other_task_same_layout` | same `layout_id`, different task, outcome-disjoint, feasible in query |
| `other_task_same_suite` | same suite, different task, outcome-disjoint, feasible in query |
| `other_instruction_other_layout` | different language and layout, outcome-disjoint |

- **Outcome-disjoint**: executing the candidate's behaviour must not legitimately satisfy the
  query goal. With an identical `init_id`, overlapping goal predicates share an outcome
  (`open the top drawer` ⊂ `open the top drawer and put the bowl inside`) and are excluded.
  Identical predicate strings under different initial states are distinct outcomes
  (LIBERO-Spatial: which bowl is `akita_black_bowl_1` is the task).
- **Feasible in query**: every entity the candidate goal refers to exists in the query scene.

`capabilities.relations` lists the relations with a non-empty pool for the task.

## 5. Provider contract

```python
class TaskProvider:
    name: str
    def version(self) -> Dict[str, str]
    def enumerate_tasks(self) -> List[Task]
    def info(self, tasks) -> ProviderInfo               # registry/providers/<name>.yaml
    def backend_binding(self, suite) -> {backend, inner, params}
    def export_demos(self, task, dest, max_episodes=None) -> Path   # LeRobotDataset v3
```

Rules: no simulator import at module level; ids stable across machines; every referenced file
hashed; demonstration hashes may come from the source hub API (no download required).
Conformance tests live in `tests/providers/test_conformance.py`.

## 6. Registry layout

```
providers/<provider>.yaml         version pins, license, capability matrix (track -> {relation: n})
tasks/<provider>/<suite>.yaml     {schema_version, provider, suite, n_tasks, tasks[]}
tracks/<track>.yaml               global track definitions (S4)
pools/<track>/<provider>.yaml     materialised per-query context and wrong-task pools
```

Pools are materialised so the container side can sample without provider code. Rebuilding with
the same inputs yields byte-identical files; any change is reviewed as a diff and bumps
`registry_hash`.

## 7. LIBERO provider (v1)

Source: `Lifelong-Robot-Learning/LIBERO` at `8f1084e` (the commit pinned by vla-eval's image),
robosuite 1.4.0, mujoco 3.2.3, bddl 1.0.1. Suites `libero_spatial, libero_object, libero_goal,
libero_10, libero_90` (130 tasks); demos `yifengzhu-hf/LIBERO-datasets` (Apache-2.0, 50 per
task, 20 Hz, 128×128, revision recorded). Evaluation init states (`.pruned_init`, 50 per task) are
disjoint from demonstration start states (minimum L2 distance ≈ 0.29 over all pairs on
libero_goal/libero_spatial). Pinned caps: spatial 220, object 280, goal 300, libero_10 520,
libero_90 400 steps, 10 settle steps.
