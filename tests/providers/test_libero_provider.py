from __future__ import annotations

import pytest

from icil_eval.providers.libero.provider import LiberoProvider
from icil_eval.registry.relations import compute_capabilities


@pytest.fixture(scope="module")
def tasks(fixtures_dir):
    provider = LiberoProvider(libero_root=str(fixtures_dir / "libero_root"), offline=True)
    tasks = provider.enumerate_tasks()
    compute_capabilities(tasks)
    return {t.task_id: t for t in tasks}


def test_enumeration_ids_and_order(tasks):
    ids = list(tasks)
    assert ids == sorted(ids)
    assert "libero/libero_goal/turn_on_the_stove" in ids
    assert all(t.task_id.startswith(f"libero/{t.suite}/") for t in tasks.values())


def test_language_is_filename_derived_like_libero(tasks):
    from icil_eval.providers.libero.provider import stem_language

    drawer = tasks["libero/libero_goal/open_the_middle_drawer_of_the_cabinet"]
    assert drawer.language == "open the middle drawer of the cabinet"  # what LIBERO/vla-eval use
    assert (
        drawer.extra["bddl_language"] == "Open the middle layer of the drawer"
    )  # BDDL text differs
    assert (
        stem_language("KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet")
        == "close the top drawer of the cabinet"
    )
    assert (
        stem_language("KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet")
        == "put the black bowl on top of the cabinet"
    )
    assert stem_language(
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    ).startswith("put both")


def test_libero_goal_shares_one_layout_and_init(tasks):
    goal = [t for t in tasks.values() if t.suite == "libero_goal"]
    assert len({t.scene.layout_id for t in goal}) == 1
    assert len({t.scene.init_id for t in goal}) == 1
    stove = tasks["libero/libero_goal/turn_on_the_stove"]
    assert stove.goal.predicates == [["turnon", "flat_stove_1"]]
    assert stove.skills == ["turn_on"]
    assert stove.bounds.chance_success is None
    assert stove.horizon.max_steps == 300
    # other goal tasks in the same layout are feasible, goal-disjoint wrong-task candidates
    assert "other_task_same_layout" in stove.capabilities.relations


def test_libero_spatial_chance_and_layout(tasks):
    spatial = [t for t in tasks.values() if t.suite == "libero_spatial"]
    assert len({t.scene.layout_id for t in spatial}) == 1
    assert len({t.scene.init_id for t in spatial}) == 2
    for t in spatial:
        assert t.bounds.chance_success == 0.5  # two akita_black_bowl instances
        assert t.horizon.max_steps == 220


def test_libero_90_scene_names_and_scene_relation(tasks):
    k1 = tasks["libero/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet"]
    assert k1.scene.scene_name == "KITCHEN_SCENE1"
    assert k1.language.lower() == "put the black bowl on top of the cabinet"
    assert "same_instruction_other_layout" in k1.capabilities.relations
    assert k1.horizon.max_steps == 400


def test_init_states_read_without_torch(tasks):
    t = tasks["libero/libero_goal/turn_on_the_stove"]
    assert t.init_states.verified is True
    assert t.init_states.n == 50
    assert t.init_states.sha256 and len(t.init_states.sha256) == 64
    assert t.demo_pool.path == "libero_goal/turn_on_the_stove_demo.hdf5"
    assert t.demo_pool.sha256 is None  # offline build leaves Hub hashes unknown


def test_progress_metric_flag(tasks):
    for t in tasks.values():
        assert t.capabilities.progress_metric == (t.goal.n_effective_predicates >= 2)
