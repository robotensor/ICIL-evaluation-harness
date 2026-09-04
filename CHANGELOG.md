# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer. Standards
documents under `docs/spec/` carry their own `version` field; a breaking change to a standard bumps
that field and is listed here.

## [Unreleased]

### Added
- Project scaffold: package `icil_eval`, CLI `icil-eval`, Apache-2.0 license, CI.
- Standard S1 (unified task schema, relations, provider contract, registry layout) and S2
  (demonstrations as context: LeRobotDataset v3 storage, `ContextSpec`, deterministic sampling,
  transforms, conditions) with JSON Schemas.
- `TaskProvider` contract, conformance suite and the LIBERO provider (BDDL parser, 130 tasks,
  Hub-side demo hashes, LeRobotDataset export).
- Committed registry for LIBERO: 130 tasks, tracks `configuration` (130 queries) and `scene`
  (31 queries), materialised context and wrong-task pools.
- Context layer: `Demonstration`, `ContextRef`, `ContextSpec`, hash-seeded sampler, transforms,
  raw-hdf5 and LeRobot loaders.
- Standard S3: `ICILPolicy` protocol (`reset` / `set_context` / `observe` / `act`), `PolicySpec`,
  `PolicyCard`, capability math (`resolve_k_max`, worst-case demonstration lengths).
- Behavior Prompting Policy wrapper: shared `BPPModel` (mmap checkpoint load, checkpoint-compat
  flags, prompt encoding, per-session cache swap) and per-session `BPPPolicy` (prompt chunking,
  K>1 concatenation within the positional budget, K=0 blank prompt, 2-frame history).
- vla-eval backend model server `ICILModelServer`: context by reference from `EPISODE_START`
  fields, language stripping, every observation to the policy, per-episode JSONL log; CLI
  `icil-eval serve bpp`.
