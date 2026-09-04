"""Provider conformance suite: every provider must satisfy these invariants."""

from __future__ import annotations

import pytest

from icil_eval.providers.libero.provider import LiberoProvider
from icil_eval.registry.relations import RELATIONS, compute_capabilities


def _providers(fixtures_dir):
    return [LiberoProvider(libero_root=str(fixtures_dir / "libero_root"), offline=True)]


@pytest.fixture(params=["libero"])
def provider(request, fixtures_dir):
    return {p.name: p for p in _providers(fixtures_dir)}[request.param]


def test_tasks_well_formed(provider):
    tasks = provider.enumerate_tasks()
    compute_capabilities(tasks)
    ids = [t.task_id for t in tasks]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for t in tasks:
        assert t.task_id.isascii() and " " not in t.task_id
        assert t.task_id.startswith(f"{provider.name}/{t.suite}/")
        assert t.language.strip()
        assert t.scene.entities and t.goal.predicates
        assert t.horizon.max_steps > 0 and t.init_states.n > 0 and t.demo_pool.n > 0
        assert "same_task" in t.capabilities.relations
        assert set(t.capabilities.relations) <= set(RELATIONS)
        assert t.source_version


def test_backend_binding_and_info(provider):
    tasks = provider.enumerate_tasks()
    binding = provider.backend_binding(tasks[0].suite)
    assert {"backend", "inner", "params"} <= set(binding)
    info = provider.info(tasks)
    assert info.name == provider.name and info.n_tasks == len(tasks)
    assert info.license and info.url and info.version
