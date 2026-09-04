# S4 — Tracks, conditions and the K sweep

version: 1

## 1. A track is a relation

A track is defined **once, globally**, by the relation between the context demonstrations and the
query episode, not by a list of tasks:

```yaml
name: configuration
relation: same_task                     # context pool relation (S1 §4)
language: none                          # what the policy is told besides the demonstrations
wrong_task_rule: {relation: other_task_same_layout, fallback: other_task_same_suite,
                  outcome_disjoint: true, feasible_in_query: true}
k_sweep: [0, 1, 2, 4, 8]
conditions: [k0, k1, k2, k4, k8, k1.wrong_task, k1.shuffled_chunks, kmax.wrong_task]
k_ref: 1
context_chunk_seconds: 0.5
version: 1
```

The pool of a track is every registry task for which its provider can construct the relation.
Adding a provider therefore extends every track it is capable of, and results are reported
globally and per provider.

## 2. v1 tracks

| track | relation | what it measures | wrong-task control |
|---|---|---|---|
| `configuration` | `same_task` | executing a known task from a new initial configuration, specified only by demonstrations | another task feasible in the same layout whose outcome is disjoint (else same suite) |
| `scene` | `same_instruction_other_layout` | the same instruction demonstrated in a different layout | a different instruction from a different layout (both factors wrong) |

Declared for future providers (pool empty until one contributes): `instance`, `category`,
`composition`, `unseen_skill`, `embodiment`, `dynamics`, `long_horizon`.

Report **slices** (subsets of a track, reported alongside it) capture where a provider's tasks make
a track especially informative. For LIBERO `configuration`: `libero_goal` (ten goals in one
byte-identical layout and initial state, so the demonstration alone selects the goal),
`libero_spatial` (identical layout, two black bowls, `chance_success = 0.5`), `libero_10`
(two-predicate goals with a `progress` metric).

## 3. Conditions

| condition | meaning |
|---|---|
| `k0` | no context (the policy's `k0_semantics`, e.g. a blank prompt) |
| `k1, k2, k4, k8` | K correct demonstrations from the track's context pool |
| `k1.wrong_task` | K=1 demonstration of a wrong task (per the wrong-task rule) |
| `k1.shuffled_chunks` | K=1 correct demonstration with its 0.5 s chunks permuted |
| `kmax.wrong_task` | wrong-task control at the largest supported K |

Every condition runs on the same evaluation initial states (`episode_idx` ↔ initial state), so
differences are paired. A policy that cannot run a condition reports `unsupported` for it; the
condition is never silently downgraded.

## 4. Adding a track

1. Add `tracks/<name>.yaml` with a relation from S1 §4 (or add the relation function first).
2. Rebuild the registry: pools are materialised for every provider with a non-empty pool.
3. CI rejects a new track whose expanded episode set equals an existing track's for a provider
   with the same language mode.
