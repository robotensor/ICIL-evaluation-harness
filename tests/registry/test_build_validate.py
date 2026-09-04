from __future__ import annotations

import shutil

import pytest

from icil_eval.paths import REGISTRY_ROOT
from icil_eval.registry.build import build_provider
from icil_eval.registry.io import load_registry
from icil_eval.registry.validate import validate_registry


@pytest.fixture(scope="module")
def built_root(tmp_path_factory, fixtures_dir):
    root = tmp_path_factory.mktemp("registry")
    shutil.copytree(REGISTRY_ROOT / "tracks", root / "tracks")
    rc = build_provider(
        "libero",
        root=root,
        provider_options={"libero_root": str(fixtures_dir / "libero_root"), "offline": True},
    )
    assert rc == 0
    return root


def test_registry_loads_and_validates(built_root):
    reg = load_registry(built_root)
    assert len(reg.tasks) == 9
    assert set(reg.providers) == {"libero"}
    assert validate_registry(built_root) == 0
    assert len(reg.registry_hash()) == 64


def test_configuration_pools(built_root):
    reg = load_registry(built_root)
    pools = reg.pools[("configuration", "libero")]
    assert set(pools) == set(reg.tasks)
    stove = pools["libero/libero_goal/turn_on_the_stove"]
    assert stove.context == ["libero/libero_goal/turn_on_the_stove"]
    wrong_ids = {w[0] for w in stove.wrong}
    assert "libero/libero_goal/put_the_bowl_on_the_plate" in wrong_ids
    assert all(w[1] == "other_task_same_layout" for w in stove.wrong)


def test_scene_pools(built_root):
    reg = load_registry(built_root)
    pools = reg.pools[("scene", "libero")]
    k1 = pools["libero/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet"]
    assert set(k1.context) == {
        "libero/libero_90/KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "libero/libero_90/KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    }
    assert "libero/libero_90/KITCHEN_SCENE3_turn_on_the_stove" in {w[0] for w in k1.wrong}
    # same instruction across suites in different layouts is a legitimate scene pair
    stove = pools["libero/libero_goal/turn_on_the_stove"]
    assert stove.context == ["libero/libero_90/KITCHEN_SCENE3_turn_on_the_stove"]
    assert "libero/libero_goal/put_the_bowl_on_the_plate" not in pools


def test_capability_matrix(built_root):
    reg = load_registry(built_root)
    cm = reg.providers["libero"].capability_matrix
    assert cm["configuration"] == {"same_task": 9}
    assert cm["scene"] == {"same_instruction_other_layout": 5}
