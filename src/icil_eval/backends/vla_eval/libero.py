"""LIBERO inner benchmark with a configurable render resolution and goal-predicate progress.

vla-eval's ``LIBEROBenchmark`` bakes ``LIBERO_ENV_RESOLUTION = 256`` into ``reset()``; this
subclass re-implements ``reset`` with ``resolution`` (128 reproduces Behavior Prompting Policy's
native training render) and exposes the per-predicate goal status that the ICIL wrapper turns into
a ``progress`` metric. Everything else is inherited.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vla_eval.benchmarks.libero.benchmark import LIBERO_DUMMY_ACTION, LIBEROBenchmark


class ICILLIBEROBenchmark(LIBEROBenchmark):
    def __init__(self, *args: Any, resolution: int = 256, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resolution = int(resolution)

    def reset(self, task: Dict[str, Any]) -> Any:
        from pathlib import Path

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_obj = task["task_obj"]
        task_id = task["task_id"]
        episode_idx = task.get("episode_idx", 0)
        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._env.close()
            bddl_file = (
                Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
            )
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=self.resolution,
                camera_widths=self.resolution,
            )
            env.seed(self.env_seed)
            self._env = env
            self._current_task_id = task_id
        self._env.reset()
        assert self._task_suite is not None
        initial_states = self._task_suite.get_task_init_states(task_id)
        obs = self._env.set_init_state(initial_states[episode_idx])
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
        if self.absolute_action:
            for robot in self._env.robots:
                robot.controller.use_delta = False
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def goal_predicate_status(self) -> Optional[List[bool]]:
        """Truth value of every BDDL goal predicate in the current simulator state."""
        if self._env is None:
            return None
        problem = getattr(self._env, "env", None)
        goal_state = (
            getattr(problem, "parsed_problem", {}).get("goal_state")
            if problem is not None
            else None
        )
        if not goal_state or not hasattr(problem, "_eval_predicate"):
            return None
        return [bool(problem._eval_predicate(state)) for state in goal_state]

    def get_metadata(self) -> Dict[str, Any]:
        meta = dict(super().get_metadata())
        meta["render_resolution"] = self.resolution
        return meta
