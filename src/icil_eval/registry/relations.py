"""Context<->query relations, defined generically over the unified task schema.

A relation maps a query task to the set of tasks whose demonstrations may serve as context (or as
wrong-task control). Because relations only read registry fields (language, layout, goal
predicates, scene entities), any provider that fills those fields correctly contributes to every
track automatically.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence, Tuple

from icil_eval.registry.schema import Task

RelationFn = Callable[[Task, Sequence[Task]], List[Task]]

_WS = re.compile(r"\s+")


def normalize_language(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def goal_tuples(task: Task) -> set:
    return {tuple(p) for p in task.goal.predicates}


def goal_disjoint(a: Task, b: Task) -> bool:
    return not (goal_tuples(a) & goal_tuples(b))


def outcome_disjoint(candidate: Task, query: Task) -> bool:
    """Executing the candidate's behaviour must not legitimately satisfy the query goal.

    Two tasks with the same initial state whose goal predicates overlap share an outcome (e.g.
    ``open the top drawer`` is a sub-goal of ``open the top drawer and put the bowl inside``).
    Identical predicate strings under *different* initial states refer to different physical
    situations (LIBERO-Spatial: which bowl is ``akita_black_bowl_1`` is what the task is about), so
    they are distinct outcomes.
    """
    if candidate.scene.init_id != query.scene.init_id:
        return True
    return goal_disjoint(candidate, query)


def feasible_in_query(candidate: Task, query: Task) -> bool:
    """Every entity the candidate's goal refers to exists in the query scene."""
    entities = set(query.scene.entities)
    for pred in candidate.goal.predicates:
        for arg in pred[1:]:
            if arg not in entities:
                return False
    return True


def _sorted(tasks: List[Task]) -> List[Task]:
    return sorted(tasks, key=lambda t: t.task_id)


def same_task(query: Task, tasks: Sequence[Task]) -> List[Task]:
    return [query]


def same_instruction_other_layout(query: Task, tasks: Sequence[Task]) -> List[Task]:
    lang = normalize_language(query.language)
    return _sorted(
        [
            t
            for t in tasks
            if t.provider == query.provider
            and t.task_id != query.task_id
            and normalize_language(t.language) == lang
            and t.scene.layout_id != query.scene.layout_id
        ]
    )


def other_task_same_layout(query: Task, tasks: Sequence[Task]) -> List[Task]:
    return _sorted(
        [
            t
            for t in tasks
            if t.provider == query.provider
            and t.task_id != query.task_id
            and t.scene.layout_id == query.scene.layout_id
            and outcome_disjoint(t, query)
            and feasible_in_query(t, query)
        ]
    )


def other_task_same_suite(query: Task, tasks: Sequence[Task]) -> List[Task]:
    return _sorted(
        [
            t
            for t in tasks
            if t.provider == query.provider
            and t.suite == query.suite
            and t.task_id != query.task_id
            and outcome_disjoint(t, query)
            and feasible_in_query(t, query)
        ]
    )


def other_instruction_other_layout(query: Task, tasks: Sequence[Task]) -> List[Task]:
    lang = normalize_language(query.language)
    return _sorted(
        [
            t
            for t in tasks
            if t.provider == query.provider
            and t.task_id != query.task_id
            and normalize_language(t.language) != lang
            and t.scene.layout_id != query.scene.layout_id
            and outcome_disjoint(t, query)
        ]
    )


RELATIONS: Dict[str, RelationFn] = {
    "same_task": same_task,
    "same_instruction_other_layout": same_instruction_other_layout,
    "other_task_same_layout": other_task_same_layout,
    "other_task_same_suite": other_task_same_suite,
    "other_instruction_other_layout": other_instruction_other_layout,
}

# Relations whose pool is non-trivial (excludes the always-available identity relation).
CAPABILITY_RELATIONS = tuple(r for r in RELATIONS if r != "same_task")


def relation_pool(query: Task, relation: str, tasks: Sequence[Task]) -> List[Task]:
    if relation not in RELATIONS:
        raise KeyError(f"unknown relation '{relation}'; known: {sorted(RELATIONS)}")
    return RELATIONS[relation](query, tasks)


def wrong_task_pool(query: Task, tasks: Sequence[Task], rule: Dict) -> List[Tuple[str, str]]:
    """Return ``[(task_id, relation_label), ...]`` for the wrong-task control.

    ``rule`` = ``{relation, fallback?}``; outcome-disjointness and feasibility are enforced by the
    relation functions themselves. The first non-empty relation in ``[relation, fallback]`` is
    used and its name is the label recorded on every result row.
    """
    for rel in [rule.get("relation"), rule.get("fallback")]:
        if not rel:
            continue
        pool = relation_pool(query, rel, tasks)
        if pool:
            return [(t.task_id, rel) for t in pool]
    return []


def compute_capabilities(tasks: List[Task]) -> None:
    """Fill ``task.capabilities.relations`` in place for a provider's task list."""
    for t in tasks:
        rels = ["same_task"]
        for rel in CAPABILITY_RELATIONS:
            if relation_pool(t, rel, tasks):
                rels.append(rel)
        t.capabilities.relations = rels
