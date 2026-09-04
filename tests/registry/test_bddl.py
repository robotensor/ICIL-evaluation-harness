from __future__ import annotations

from icil_eval.providers.libero.bddl import load_bddl, parse_bddl


def test_parse_turn_on_the_stove(fixtures_dir):
    p = load_bddl(fixtures_dir / "libero_root/bddl_files/libero_goal/turn_on_the_stove.bddl")
    assert p.problem_name == "LIBERO_Tabletop_Manipulation"
    assert p.domain == "robosuite"
    assert p.language == "Turn on the stove"
    assert p.fixtures["main_table"] == "table"
    assert p.fixtures["flat_stove_1"] == "flat_stove"
    assert "akita_black_bowl_1" in p.objects
    assert p.goal == [["Turnon", "flat_stove_1"]]
    assert any(pred[0] == "On" for pred in p.init)
    # regions are exposed with LIBERO's <target>_<name> convention
    assert "main_table_plate_region" in p.entities
    assert "flat_stove_1_cook_region" in p.entities


def test_layout_identical_across_libero_goal(fixtures_dir):
    root = fixtures_dir / "libero_root/bddl_files/libero_goal"
    layouts = {load_bddl(f).layout_dict().__repr__() for f in root.glob("*.bddl")}
    inits = {repr(sorted(load_bddl(f).init)) for f in root.glob("*.bddl")}
    assert len(layouts) == 1
    assert len(inits) == 1


def test_typed_names_and_and_goal():
    text = """(define (problem P) (:domain robosuite) (:language put a on b)
      (:regions (r1 (:target t) (:ranges ((0 0 1 1)))))
      (:fixtures t - table) (:objects a b - thing c - other)
      (:obj_of_interest a b)
      (:init (On a t_r1) (On b t_r1))
      (:goal (And (On a b) (Open t_r1))))"""
    p = parse_bddl(text)
    assert p.objects == {"a": "thing", "b": "thing", "c": "other"}
    assert p.goal == [["On", "a", "b"], ["Open", "t_r1"]]
    assert p.regions["r1"].full_name == "t_r1"
    assert p.regions["r1"].ranges == [[0.0, 0.0, 1.0, 1.0]]
