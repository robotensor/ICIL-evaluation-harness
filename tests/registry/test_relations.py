from __future__ import annotations

from icil_eval.registry.relations import (
    compute_capabilities,
    feasible_in_query,
    goal_disjoint,
    other_task_same_layout,
    outcome_disjoint,
    same_instruction_other_layout,
    wrong_task_pool,
)
from tests.registry.synth import make_task


def test_goal_disjoint_and_feasibility():
    a = make_task("a", goal=[["on", "obj_1", "plate_1"]])
    b = make_task("b", goal=[["on", "obj_1", "plate_1"], ["turnon", "stove_1"]])
    c = make_task("c", goal=[["turnon", "stove_1"]])
    d = make_task("d", goal=[["in", "obj_1", "drawer_9"]])
    assert not goal_disjoint(a, b)  # b subsumes a
    assert goal_disjoint(a, c)
    assert feasible_in_query(c, a)
    assert not feasible_in_query(d, a)  # drawer_9 does not exist in a's scene


def test_other_task_same_layout_excludes_subsuming_and_infeasible():
    q = make_task("q", goal=[["on", "obj_1", "plate_1"]])
    sub = make_task("sub", goal=[["on", "obj_1", "plate_1"], ["turnon", "stove_1"]])
    ok = make_task("ok", goal=[["turnon", "stove_1"]])
    other_layout = make_task("ol", layout="cccccccccccc", goal=[["turnon", "stove_1"]])
    infeasible = make_task("inf", goal=[["in", "obj_1", "drawer_9"]])
    pool = other_task_same_layout(q, [q, sub, ok, other_layout, infeasible])
    assert [t.stem for t in pool] == ["ok"]


def test_same_instruction_other_layout_and_fallback():
    q = make_task(
        "k1_stove",
        language="Turn on the stove",
        layout="111111111111",
        goal=[["turnon", "stove_1"]],
    )
    same = make_task(
        "k3_stove",
        language="turn on the  stove",
        layout="333333333333",
        goal=[["turnon", "stove_1"]],
    )
    same_layout = make_task(
        "k1_other", language="open the drawer", layout="111111111111", goal=[["open", "d_1"]]
    )
    pool = same_instruction_other_layout(q, [q, same, same_layout])
    assert [t.stem for t in pool] == ["k3_stove"]
    # wrong-task: nothing in same layout is feasible (d_1 missing) -> fallback to same suite
    rule = {"relation": "other_task_same_layout", "fallback": "other_task_same_suite"}
    wrong = wrong_task_pool(q, [q, same, same_layout], rule)
    assert wrong == []  # same-suite candidates are also infeasible/duplicate goals
    feasible = make_task(
        "k9", language="x", layout="999999999999", goal=[["on", "obj_1", "plate_1"]]
    )
    wrong = wrong_task_pool(q, [q, same, same_layout, feasible], rule)
    assert wrong == [("synth/s1/k9", "other_task_same_suite")]


def test_compute_capabilities():
    q = make_task("q", goal=[["on", "obj_1", "plate_1"]])
    ok = make_task("ok", goal=[["turnon", "stove_1"]])
    tasks = [q, ok]
    compute_capabilities(tasks)
    assert "same_task" in q.capabilities.relations
    assert "other_task_same_layout" in q.capabilities.relations
    assert "same_instruction_other_layout" not in q.capabilities.relations


def test_identical_goal_strings_are_distinct_outcomes_under_different_init():
    # LIBERO-Spatial: every task's goal is (On akita_black_bowl_1 plate_1); only the init differs.
    a = make_task("bowl_on_stove", init="111111111111", goal=[["on", "obj_1", "plate_1"]])
    b = make_task("bowl_on_cabinet", init="222222222222", goal=[["on", "obj_1", "plate_1"]])
    same_init_subgoal = make_task(
        "sub", init="111111111111", goal=[["on", "obj_1", "plate_1"], ["turnon", "stove_1"]]
    )
    assert not goal_disjoint(a, b)
    assert outcome_disjoint(b, a)
    assert not outcome_disjoint(same_init_subgoal, a)
    pool = other_task_same_layout(a, [a, b, same_init_subgoal])
    assert [t.stem for t in pool] == ["bowl_on_cabinet"]
