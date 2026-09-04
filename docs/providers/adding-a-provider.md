# Adding a task provider

A provider normalises one existing benchmark into the registry. Copy
`src/icil_eval/providers/_template/` to `src/icil_eval/providers/<name>/` and implement
`icil_eval.providers.base.TaskProvider`:

| method | what it must do |
|---|---|
| `version()` | pinned upstream versions (commit, simulator versions) |
| `enumerate_tasks()` | every task as `Task` (S1): stable `task_id`, filename/registry-derived `language`, `scene.layout_id`/`init_id`, `scene.entities`, `objects`, `skills`, normalised `goal.predicates`, `horizon`, `init_states` (with sha256), `demo_pool` (repo, revision, path, sha256 — from the source hub API where possible), `bounds`, `source_version` |
| `info(tasks)` | `ProviderInfo`: description, license, url, citation, suites, backend |
| `backend_binding(suite)` | `{backend, inner, params}` — the vla-eval inner benchmark import string and its constructor params |
| `export_demos(task, dest)` | LeRobotDataset v3 with LeRobot keys and a dataset card (S2) |

Rules: no simulator import at module level; ids stable across machines; hash every referenced
file. Relations (`same_task`, `same_instruction_other_layout`, `other_task_same_layout`, …) are
generic functions over the schema, so a correctly filled `Task` contributes to every track its
data supports — set `layout_id` from the scene definition (fixtures, objects, regions) and
`entities` to everything a goal may refer to.

Register the class in `icil_eval.providers.base.PROVIDER_CLASSES`, then:

```bash
icil-eval registry build <name> -o <option>=<value> ...
icil-eval registry validate
pytest tests/providers   # add your provider to the conformance suite
```

Commit the generated `src/icil_eval/data/registry/{providers,tasks,pools}` files; the registry
hash changes and is recorded by every subsequent result.

If the benchmark is already wrapped by vla-eval, the rollout side is just its import string. For a
custom render resolution or a progress metric, subclass the inner benchmark like
`icil_eval.backends.vla_eval.libero.ICILLIBEROBenchmark` and expose `goal_predicate_status()`.
