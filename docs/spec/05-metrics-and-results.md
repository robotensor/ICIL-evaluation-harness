# S5 — Metrics, scoring and results

version: 1

## 1. Per-episode record

Every evaluated episode yields one row (`episodes[]` in the results file):

```
task_id, provider, suite, track, condition, k, control, episode_idx (= evaluation initial state),
status ∈ {ok, unsupported, not_applicable, error, missing}, success, progress, steps, elapsed_s,
relation, transform, language, context_refs [[task_id, episode_index], ...], context_hash,
context_task_id, k_max, adaptation_latency_s, control_latency_s, min_context_init_l2,
query_exposure ∈ {seen, unseen, undisclosed}, context_exposure ∈ {seen, unseen, partial, none, undisclosed}
```

Rows with status `unsupported` or `error` carry no outcome (`success = null`): a hold action is not
a policy result. `progress = max_t (n_sat(t) − n_sat(t0)) / (n_goal − n_sat(t0))`, null when no goal
predicate is initially unsatisfied.

## 2. Per-condition summary

For each track, and within it per provider and per report slice (LIBERO: suite):

- `success`: mean over `ok` episodes with a Wilson 95% interval;
- `success_macro`: mean over tasks of per-task success, with a task-cluster bootstrap interval;
- `n_episodes`, `n_ok`, `n_tasks`, `status_counts`, `progress`, latency medians, `steps_mean`.

## 3. Derived metrics (paired)

Conditions share evaluation initial states, so differences are computed **per initial state**
(`task_id`, `episode_idx`) and summarised with the paired standard error, the number of
discordant pairs and an exact McNemar test:

| metric | definition |
|---|---|
| `delta_context@1` | `k1` − `k1.wrong_task` — the wrong-context gap; the headline ICIL statistic |
| `delta_context@kmax` | same at the largest supported K (`kmax.wrong_task`) |
| `context_gain@k` | `k<K>` − `k0` for K ∈ {1, 2, 4, 8} |
| `order_sensitivity@1` | `k1` − `k1.shuffled_chunks` |
| `sr_at_k`, `k_supported` | SR per K in the fixed sweep; K values with `ok` episodes |
| `context_auc` | mean of SR@K over the fixed sweep {0, 1, 2, 4, 8}; `null` if any K is unsupported |
| `delta_context_above_chance@1` | `SR@1 − max(SR@1.wrong_task, chance_success)` when the registry knows chance levels |

`kmax.*` conditions are resolved per task using the server-reported K, and reported under the
resolved name (`k<K>.wrong_task`).

## 4. Model profile and coverage

The leaderboard object is the **profile**: `tracks → {conditions, derived, providers, slices}` plus
`coverage = {tasks_run, tasks_total_in_results, episodes, status_counts, providers, tracks,
query_exposure, context_exposure}`. Scores are never compared across differing coverage
without stating it; an optional scalar summary may only be shown next to its coverage.

Leaderboard rules: rank on `delta_context@1` (exposure-robust) and on SR restricted to
`query_exposure = unseen`; `undisclosed` ranks as `seen`; `seen` SR is shown but flagged.

## 5. Exposure tagging

From the policy card: `query_exposure = seen` if the query task (id or instruction string) is in
`training_tasks`; `context_exposure` is `seen`/`partial`/`unseen` by checking each context
demonstration against `training_episodes` (falling back to task-level membership), `none` for
K=0. Without a card everything is `undisclosed`.

## 6. Results file

```
schema_version, harness_version, backend {name, version}, date, git_hash,
policy {name, spec, card}, config {registry_hash, run, backend_configs}, profile, episodes[]
```

Produced by `icil-eval report <run_dir>` from the backend's `*_aggregate.json` and the ICIL server
JSONL log (joined on the backend episode id), validated against `schemas/results.schema.json`,
and rendered as markdown tables per track and slice.
