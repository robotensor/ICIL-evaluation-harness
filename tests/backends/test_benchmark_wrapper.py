"""ICILBenchmark wrapper tests with a stub inner benchmark (no simulator)."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from vla_eval.benchmarks.base import StepBenchmark, StepResult

from icil_eval.backends.vla_eval.benchmark import ICILBenchmark
from icil_eval.paths import REGISTRY_ROOT
from icil_eval.registry.build import build_provider


class StubLibero(StepBenchmark):
    """Mimics vla-eval's LIBEROBenchmark task dicts and episode API for libero_goal fixtures."""

    STEMS = [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_plate",
        "turn_on_the_stove",
    ]
    _ALL_RECORD_FIELDS = ("reward", "done", "success")

    def __init__(
        self,
        suite="libero_goal",
        seed=7,
        num_steps_wait=10,
        send_wrist_image=False,
        send_state=False,
        max_steps=None,
    ):
        super().__init__()
        self.suite, self.seed, self.num_steps_wait = suite, seed, num_steps_wait
        self.send_wrist_image, self.send_state = send_wrist_image, send_state
        self.reset_calls, self.step_calls = [], 0
        self._goals = [False, False]

    def get_tasks(self):
        return [
            {
                "name": s.replace("_", " "),
                "suite": self.suite,
                "task_id": i,
                "task_obj": SimpleNamespace(bddl_file=f"{s}.bddl", problem_folder=self.suite),
            }
            for i, s in enumerate(self.STEMS)
        ]

    def reset(self, task):
        self.reset_calls.append(task)
        self._goals = [False, False]
        self._recorder.record_video(np.zeros((4, 4, 3), np.uint8))
        return {"agentview_image": np.zeros((4, 4, 3), np.uint8)}

    def step(self, action):
        self.step_calls += 1
        self._goals = [True, self.step_calls >= 2]
        self._recorder.record_step(reward=0.0, done=all(self._goals), success=all(self._goals))
        return StepResult(
            obs={"agentview_image": np.zeros((4, 4, 3), np.uint8)},
            reward=0.0,
            done=all(self._goals),
            info={},
        )

    def goal_predicate_status(self):
        return list(self._goals)

    def make_obs(self, raw_obs, task):
        return {
            "images": {"agentview": raw_obs["agentview_image"]},
            "task_description": task["name"],
        }

    def get_step_result(self, step_result):
        return {"success": bool(step_result.done)}

    def get_metadata(self):
        return {"max_steps": 300, "max_episodes_per_task": 50, "suite": self.suite}

    def get_action_spec(self):
        return {}

    def get_observation_spec(self):
        return {}


@pytest.fixture(scope="module")
def registry_root(tmp_path_factory, fixtures_dir):
    root = tmp_path_factory.mktemp("registry")
    shutil.copytree(REGISTRY_ROOT / "tracks", root / "tracks")
    build_provider(
        "libero",
        root=root,
        provider_options={"libero_root": str(fixtures_dir / "libero_root"), "offline": True},
    )
    return root


def make_wrapper(registry_root, **kw):
    return ICILBenchmark(
        inner="tests.backends.test_benchmark_wrapper:StubLibero",
        track="configuration",
        registry_root=str(registry_root),
        suite="libero_goal",
        send_wrist_image=True,
        send_state=True,
        seed=1000,
        **kw,
    )


def test_expansion_is_task_major_with_unique_names_and_flat_fields(registry_root):
    bench = make_wrapper(registry_root)
    tasks = bench.get_tasks()
    conds = bench.registry.tracks["configuration"].conditions
    assert len(tasks) == 3 * len(conds)
    names = [t["name"] for t in tasks]
    assert len(set(names)) == len(names)
    assert names[0] == f"open_the_middle_drawer_of_the_cabinet|{conds[0]}"
    assert [t["icil_condition"] for t in tasks[: len(conds)]] == conds  # task-major
    first = tasks[0]
    assert first["icil_task_id"] == "libero/libero_goal/open_the_middle_drawer_of_the_cabinet"
    assert first["icil_language"] == "open the middle drawer of the cabinet"
    assert (
        first["icil_language_mode"] == "none" and first["icil_registry_hash"] == bench.registry_hash
    )
    flat = {k: v for k, v in first.items() if k.startswith("icil_")}
    assert all(isinstance(v, (str, int, float, bool, list)) for v in flat.values())
    assert bench.inner.send_wrist_image and bench.inner.send_state  # forwarded HELLO params


def test_capabilities_skip_unsupported_conditions(registry_root, tmp_path):
    caps = tmp_path / "caps.json"
    caps.write_text('{"k_max_by_task": {"libero/libero_goal/turn_on_the_stove": 2}}')
    bench = make_wrapper(
        registry_root, capabilities=str(caps), query_tasks=["libero/libero_goal/turn_on_the_stove"]
    )
    tasks = bench.get_tasks()
    conds = {t["icil_condition"] for t in tasks}
    assert "k4" not in conds and "k8" not in conds and "k2" in conds and "kmax.wrong_task" in conds
    assert {s["condition"] for s in bench.skipped} == {"k4", "k8"}
    assert bench.get_metadata()["skipped"] == bench.skipped


def test_episode_delegation_language_stripping_and_progress(registry_root):
    bench = make_wrapper(registry_root)
    task = [t for t in bench.get_tasks() if t["icil_condition"] == "k1"][0]

    async def go():
        await bench.start_episode({**task, "episode_idx": 3})
        obs0 = await bench.get_observation()
        results = []
        for _ in range(3):
            await bench.apply_action({"actions": np.zeros(7)})
            results.append(await bench.is_done())
        return obs0, results, await bench.get_result()

    obs0, dones, result = anyio.run(go)
    assert (
        bench.inner.reset_calls[0]["name"] == "open the middle drawer of the cabinet"
    )  # true instruction restored
    assert bench.inner.reset_calls[0]["episode_idx"] == 3
    assert "task_description" not in obs0  # language: none
    assert dones == [False, True, True]
    assert result["success"] is True and result["progress"] == 1.0
    assert result["icil_condition"] == "k1" and result["icil_task_id"].endswith(
        "open_the_middle_drawer_of_the_cabinet"
    )
    assert bench.get_metric_keys() == {"success": "mean", "progress": "mean"}
    assert bench._ALL_RECORD_FIELDS == ("reward", "done", "success")


def test_max_query_tasks_limits_query_tasks_not_conditions(registry_root):
    bench = make_wrapper(registry_root, max_query_tasks=1, conditions=["k1", "k1.wrong_task"])
    tasks = bench.get_tasks()
    assert len(tasks) == 2 and {t["icil_condition"] for t in tasks} == {"k1", "k1.wrong_task"}
