"""Synthetic task factory for registry/context unit tests."""

from __future__ import annotations

from typing import List, Optional

from icil_eval.registry.schema import (
    Bounds,
    Capabilities,
    DemoPool,
    Embodiment,
    Goal,
    Horizon,
    InitStates,
    Scene,
    Task,
)


def make_task(
    stem: str,
    suite: str = "s1",
    language: str = "do it",
    layout: str = "aaaaaaaaaaaa",
    init: str = "bbbbbbbbbbbb",
    goal: Optional[List[List[str]]] = None,
    entities: Optional[List[str]] = None,
    n_demos: int = 50,
    provider: str = "synth",
) -> Task:
    goal = goal if goal is not None else [["on", "obj_1", "plate_1"]]
    return Task(
        task_id=f"{provider}/{suite}/{stem}",
        provider=provider,
        suite=suite,
        language=language,
        embodiment=Embodiment(robot="panda"),
        scene=Scene(
            layout_id=layout, init_id=init, entities=entities or ["obj_1", "plate_1", "stove_1"]
        ),
        objects=[],
        skills=["place_on"],
        goal=Goal(predicates=goal, n_effective_predicates=len(goal)),
        horizon=Horizon(max_steps=100),
        init_states=InitStates(file=f"init/{stem}.init", n=50),
        demo_pool=DemoPool(repo_id="synth/demos", path=f"{suite}/{stem}.hdf5", n=n_demos),
        bounds=Bounds(),
        capabilities=Capabilities(),
    )
