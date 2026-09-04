# ICIL-evaluation-harness

**One global benchmark for In-Context Imitation Learning (ICIL), built from the tasks of existing robot benchmarks.**

> Status: pre-alpha (v0.1.0 in development). The first integrated task provider is LIBERO; the
> registry, standards and scorer are designed for many providers.

## Why

In-context imitation learning policies (Behavior Prompting Policy, ICRT, RoboSSM, Instant Policy, …)
are prompted at test time with a few demonstrations and must execute a new task with no weight
update. Today every paper evaluates on its own task list, its own number of demonstrations, its own
prompt-sampling rule, and almost nobody reports the controls (zero context, wrong-task context,
shuffled context) that separate *reading the demonstration* from *reading the scene*. There is no
shared benchmark.

Existing robot benchmarks (LIBERO, RoboCasa365, RoboTwin 2.0, ManiSkill3, CALVIN, …) already
provide simulators, tasks, demonstrations and success predicates, and existing VLA harnesses
already run them. What is missing is the **ICIL layer**: demonstrations as context, a policy
protocol that takes context, tracks that define generalization by the relation between context and
query, and metrics that measure whether the context caused the behaviour.

## What this project defines

- **A unified task registry.** Every task from every integrated benchmark ("task provider") is
  described in one schema under one namespace (`libero/libero_goal/turn_on_the_stove`), with
  capability tags saying which ICIL relations, prompt modalities and perturbations that task can
  contribute. Providers are added; the benchmark grows; scores stay comparable.
- **A context standard.** Demonstrations are LeRobotDataset v3 episodes; a context is a
  deterministic, hash-seeded set of demonstration references with a declared relation to the query
  (`same_task`, `same_instruction_other_layout`, …), an optional transform (`shuffled_chunks`,
  `reversed_actions`) and an explicit language mode (`none` for demo-only evaluation).
- **A policy protocol.** `reset()`, `set_context(demos)`, `observe(obs)` every control step,
  `act()` when the harness needs actions; a `PolicySpec` declaring context budget, supported K,
  embodiments and a policy card with training-data disclosure. Policies run in their own
  environment and are served over WebSocket to the rollout backend.
- **Global tracks and a K sweep.** Every policy is evaluated at K ∈ {0, 1, 2, 4, 8}
  demonstrations plus wrong-task and shuffled-context controls, on tracks defined once over the
  whole registry (`configuration`, `scene`, later `instance`, `category`, `composition`,
  `unseen_skill`, `embodiment`, `dynamics`, `long_horizon`).
- **Metrics that test in-context learning, not manipulation alone.** SR@K, context gain, context
  AUC, the wrong-context gap Δ_context (paired per initial state), order sensitivity, progress on
  long-horizon tasks, adaptation and control latency, and explicit **coverage** of which tasks a
  policy could run. Training exposure is disclosed and tagged per result row, never assumed.

Rollouts are executed by existing harnesses behind an `EnvBackend` interface; the first backend is
[allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness), whose Docker
images run LIBERO, RoboCasa365, RoboTwin, ManiSkill and more.

## Layout

```
src/icil_eval/
  registry/    unified task/track schema, loading, hashing        (Python 3.8-safe)
  providers/   TaskProvider contract; libero/ ; _template/
  context/     Demonstration, ContextSpec, sampler, transforms, loaders
  policy/      ICILPolicy protocol, PolicySpec, policy card
  policies/    reference wrappers (bpp/)
  backends/    EnvBackend interface; vla_eval/ (benchmark wrapper, model server, configs)
  scoring/     metrics, model profile, results schema, reports
  data/        committed registry (data/registry) and JSON Schemas (data/schemas)
docs/spec/     the versioned standard
```

## Quickstart (LIBERO + Behavior Prompting Policy)

The harness has three processes: a **policy server** (runs in the policy's own environment), the
**benchmark** (vla-eval, in Docker or in an environment with the simulator), and the **scorer**.

```bash
# 0. install
uv venv --python 3.10 .venv && source .venv/bin/activate && uv pip install -e ".[dev,schema]"

# 1. data: LIBERO demonstrations (Apache-2.0) and the BPP checkpoint (MIT)
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset --include 'libero_goal/*' \
   --local-dir ~/.cache/icil-eval/raw/libero
hf download austinpatel/libero --local-dir ~/.cache/icil-eval/checkpoints/austinpatel_libero

# 2. serve the policy (inside the behavior_prompting conda env, GPU)
pip install -e ".[server,bpp]"
icil-eval serve bpp --ckpt ~/.cache/icil-eval/checkpoints/austinpatel_libero/libero_behavior_prompting.ckpt

# 3. which K the checkpoint can hold per task (skips unsupported conditions before rollout)
icil-eval capabilities --policy bpp --budget-chunks 50 --chunk-len 20 --provider libero

# 4. run a track (Docker image built with docker/build.sh libero, or --no-docker with LIBERO installed)
icil-eval run --provider libero --track configuration --suites libero_goal --preset smoke \
   --capabilities ~/.cache/icil-eval/capabilities/bpp_libero.json --docker-image icil-eval/libero:dev

# 5. score: icil_results.json + markdown tables (paired Δ_context, SR@K, coverage, exposure)
icil-eval report ~/.cache/icil-eval/runs/libero_configuration_smoke
```

Presets: `smoke` (2 tasks × 5 initial states × {k1, k1.wrong_task}), `quick` (10 initial states,
every condition), `full` (50 initial states). Use `--shards N` to run N vla-eval shards in
parallel against one policy server.

### What a result looks like

`icil-eval report` on the smoke preset (BPP-LIBERO checkpoint, libero_goal, 2 tasks):

| condition | success [Wilson 95%] | n |
|---|---|---|
| `k1` | 100.0% [72.2%, 100.0%] | 10/10 |
| `k1.wrong_task` | 0.0% [0.0%, 27.8%] | 10/10 |

Δ_context@1 (paired on identical initial states) = +100 pp, n=10, McNemar p=0.002. The checkpoint
was trained on every LIBERO task, so all rows are tagged `query_exposure = seen`; the wrong-context
gap, not raw success, is the in-context-learning evidence.

## Standard documents

`docs/spec/00-overview.md` maps the standard: S1 tasks & providers, S2 context, S3 policy protocol,
S4 tracks, S5 metrics & results. Provider and policy guides live in `docs/providers/` and
`docs/policies/`.

## License

Apache-2.0. See [LICENSE](LICENSE).
