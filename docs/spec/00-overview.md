# ICIL-evaluation-harness standard — overview

version: 1 (draft)

## Purpose

This standard makes in-context imitation learning (ICIL) results comparable across policies,
laboratories and source benchmarks. It defines:

| document | defines |
|---|---|
| [01-tasks-and-providers.md](01-tasks-and-providers.md) | the unified task schema, the `TaskProvider` contract, the registry and its versioning |
| [02-context.md](02-context.md) | demonstrations as context: storage format, `Demonstration`, `ContextSpec`, deterministic sampling, transforms, controls |
| [03-policy-protocol.md](03-policy-protocol.md) | the `ICILPolicy` protocol, `PolicySpec`, policy cards, server bridges |
| [04-tracks.md](04-tracks.md) | tracks as global context↔query relations; the K sweep; conditions |
| [05-metrics-and-results.md](05-metrics-and-results.md) | per-episode records, paired metrics, the model profile with coverage, the results file, leaderboard rules |

## Design principles

1. **Union, not silo.** The benchmark's task universe is the union of tasks contributed by existing
   benchmarks. Each provider is normalised into the same schema and contributes to the capabilities
   its data supports; the harness records those capabilities explicitly.
2. **Generalization is a relation, not a split.** A global leaderboard cannot control what entrants
   trained on. Tracks are therefore defined by the relation between the context demonstrations and
   the query episode (same task / new configuration, same instruction / new layout, same skill /
   unseen object category, …). Training exposure is disclosed in the policy card and tagged on
   every result row (`query_exposure`, `context_exposure`).
3. **The context must cause the behaviour.** Every evaluation reports zero-context, wrong-task
   context and shuffled-context controls next to the correct-context result, paired on identical
   initial states, so the wrong-context gap Δ_context is the headline rather than raw success.
4. **A mandatory K sweep.** Policies are evaluated at K ∈ {0, 1, 2, 4, 8} demonstrations. A policy
   that cannot accept a given K reports `unsupported`; contexts are never silently truncated.
5. **Determinism and auditability.** Every random choice derives from a hash of the registry
   version, seed, task, condition and episode index; every result row records the demonstration
   references, hashes and versions needed to reproduce it.
6. **Coverage is first-class.** Policies differ in embodiment and inputs, so not every task applies
   to every policy. Scores are always reported with the set of tasks they cover; scalar summaries
   are never compared across differing coverage without a flag.
7. **Ecosystem-native formats.** Demonstrations are LeRobotDataset v3; naming follows LeRobot
   conventions; rollouts run on existing harnesses (vla-eval first) behind a backend interface.

## Versioning

Each document carries a `version`. A change that alters the meaning of a result (schema fields,
sampling rule, track definition, metric definition, pinned simulator versions) increments the
version of the affected document, is listed in the project CHANGELOG, and is recorded in every
results file produced afterwards.
