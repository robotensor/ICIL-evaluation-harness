# Contributing

Thank you for helping build a shared benchmark for in-context imitation learning.

## Ground rules

- **Python 3.8 compatibility for container-side code.** Modules under `icil_eval.registry`,
  `icil_eval.context.sampler`, `icil_eval.context.transforms` and `icil_eval.backends.vla_eval`
  execute inside simulator containers that ship Python 3.8 and numpy 1.22. Use
  `from __future__ import annotations`, `typing.List`/`Dict`/`Optional`, no `match`, no
  `dict | dict`, no `str.removeprefix`. CI runs these modules' tests under Python 3.8.
- **Core dependencies stay minimal** (numpy, PyYAML). Anything heavier goes behind an extra.
- **No simulator imports at module level.** Providers and backends import simulators lazily so
  the registry can be built and validated on any machine.
- **Standards are versioned.** Changing a schema, a sampling rule, a track definition or a metric
  definition requires bumping the `version` of the affected document in `docs/spec/` and a
  CHANGELOG entry. Results files record the versions they were produced with.
- **Determinism is a feature.** Anything that draws randomness must derive its seed from the
  hash-based scheme in `icil_eval.context.sampler`; never from global RNG state.

## Adding a task provider

See `docs/providers/adding-a-provider.md` (coming with v0.1.0) and the skeleton in
`src/icil_eval/providers/_template/`. A provider must pass the conformance suite in
`tests/providers/test_conformance.py`.

## Adding a policy

Implement `icil_eval.policy.ICILPolicy` (see `docs/spec/03-policy-protocol.md`) and a policy card.
Serve it with `icil-eval serve <name>` or subclass `icil_eval.backends.vla_eval.ICILModelServer`.

## Development

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev,schema]"
ruff check . && ruff format --check .
pytest
# container-side subset under Python 3.8
uv run --python 3.8 --isolated --no-project --with pytest --with "numpy==1.24.4" \
  --with pyyaml --with "vla-eval==0.5.0" --with-editable . python -m pytest -m container tests/container
```

## Commits and pull requests

- Commit title format: `(feat): short description`; keep commits focused on one coherent change.
- Keep PRs small and reviewable; include tests and a CHANGELOG entry.
