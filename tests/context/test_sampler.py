from __future__ import annotations

import shutil

import pytest

from icil_eval.context.sampler import derive_seed, sample_context
from icil_eval.context.types import Condition, UnsupportedContext
from icil_eval.paths import REGISTRY_ROOT
from icil_eval.registry.build import build_provider
from icil_eval.registry.io import load_registry


@pytest.fixture(scope="module")
def registry(tmp_path_factory, fixtures_dir):
    root = tmp_path_factory.mktemp("registry")
    shutil.copytree(REGISTRY_ROOT / "tracks", root / "tracks")
    build_provider(
        "libero",
        root=root,
        provider_options={"libero_root": str(fixtures_dir / "libero_root"), "offline": True},
    )
    return load_registry(root)


STOVE = "libero/libero_goal/turn_on_the_stove"


def test_condition_parsing():
    assert Condition.parse("k0") == Condition(k=0)
    assert Condition.parse("k4") == Condition(k=4)
    assert Condition.parse("k1.wrong_task") == Condition(k=1, control="wrong_task")
    assert Condition.parse("kmax.wrong_task").k is None
    assert Condition.parse("kmax.wrong_task").resolve(4) == Condition(k=4, control="wrong_task")
    assert Condition.parse("k1.shuffled_chunks").transform == "shuffled_chunks"
    with pytest.raises(ValueError):
        Condition.parse("k-1")


def test_derive_seed_is_stable():
    assert derive_seed("a", 1, "b") == derive_seed("a", 1, "b")
    assert derive_seed("a", 1, "b") != derive_seed("a", 2, "b")


def test_sampling_is_deterministic_and_episode_dependent(registry):
    h = registry.registry_hash()
    a = sample_context(
        registry, h, "configuration", STOVE, Condition(k=2), episode_idx=3, seed=1000
    )
    b = sample_context(
        registry, h, "configuration", STOVE, Condition(k=2), episode_idx=3, seed=1000
    )
    c = sample_context(
        registry, h, "configuration", STOVE, Condition(k=2), episode_idx=4, seed=1000
    )
    assert a.to_dict() == b.to_dict()
    assert a.refs != c.refs and a.context_hash != c.context_hash
    assert all(r.task_id == STOVE for r in a.refs) and len(a.refs) == 2
    assert a.relation == "same_task" and a.transform == "identity" and a.language == "none"
    assert len({r.episode_index for r in a.refs}) == 2


def test_k0_has_no_refs(registry):
    spec = sample_context(
        registry, registry.registry_hash(), "configuration", STOVE, Condition(k=0), 0, 1000
    )
    assert spec.refs == [] and spec.relation == "none" and spec.context_task_id is None


def test_wrong_task_draws_from_wrong_pool(registry):
    spec = sample_context(
        registry,
        registry.registry_hash(),
        "configuration",
        STOVE,
        Condition(k=1, control="wrong_task"),
        0,
        1000,
    )
    assert spec.context_task_id != STOVE
    assert spec.relation == "other_task_same_layout"
    assert spec.refs[0].task_id == spec.context_task_id


def test_shuffled_has_permutation_seed(registry):
    spec = sample_context(
        registry,
        registry.registry_hash(),
        "configuration",
        STOVE,
        Condition(k=1, control="shuffled_chunks"),
        0,
        1000,
    )
    assert spec.transform == "shuffled_chunks" and spec.permutation_seed is not None


def test_k_exceeding_pool_is_unsupported(registry):
    with pytest.raises(UnsupportedContext):
        sample_context(
            registry,
            registry.registry_hash(),
            "configuration",
            STOVE,
            Condition(k=3),
            0,
            1000,
            n_demos={STOVE: 2},
        )


def test_scene_track_context_from_other_layout(registry):
    q = "libero/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet"
    spec = sample_context(registry, registry.registry_hash(), "scene", q, Condition(k=1), 0, 1000)
    assert spec.context_task_id in {
        "libero/libero_90/KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "libero/libero_90/KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    }
    assert spec.relation == "same_instruction_other_layout"
