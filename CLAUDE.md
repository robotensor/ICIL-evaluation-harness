# ICIL-evaluation-harness

A global benchmark standard for In-Context Imitation Learning (ICIL): existing robot benchmarks
join as *task providers*, their tasks are normalised into one registry, and policies are scored on
demonstration-conditioned tracks with paired controls. Rollouts run on existing harnesses
(vla-eval first) behind a backend interface. Start with `README.md` and `docs/spec/00-overview.md`.

## Layout

- `src/icil_eval/registry` — task/track schema, relations, registry IO (Python 3.8-safe)
- `src/icil_eval/providers` — `TaskProvider` contract; `libero/` is the reference provider
- `src/icil_eval/context` — demonstrations as context: sampler, transforms, loaders
- `src/icil_eval/policy`, `policies/` — `ICILPolicy` protocol; `bpp/` reference wrapper
- `src/icil_eval/backends/vla_eval` — benchmark wrapper, model server, run configs
- `src/icil_eval/scoring` — paired statistics, model profile, results and reports
- `src/icil_eval/data` — committed registry and JSON Schemas; `docs/spec` — the standard

## Development

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate && uv pip install -e ".[dev,schema]"
ruff check . && ruff format --check . && pytest
```

Container-side modules (`registry`, `context.sampler`, `context.transforms`, `backends.vla_eval`)
must stay Python 3.8-compatible with numpy + PyYAML only; CI runs them under 3.8. Never import a
simulator at module level. Rebuild the registry with `icil-eval registry build <provider>` and
commit the generated files; changes to a standard bump its `version` in `docs/spec/` and are
listed in `CHANGELOG.md`.

## Conventions

- Commit titles: `(feat): short description`, `(fix): …`; one coherent change per commit,
  brief bullets in the body when needed; commit at logical checkpoints.
- Keep PRs focused and reviewable; include tests and a CHANGELOG entry.
- Reports and evaluation results are plain files in the repository or run directory, not hosted
  artifacts.
