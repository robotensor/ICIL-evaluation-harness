"""ICIL benchmark wrapper for vla-eval (runs inside the simulator container; Python 3.8-safe).

Wraps any vla-eval ``StepBenchmark`` (the *inner* benchmark, e.g. ``LIBEROBenchmark``) and expands
its tasks into ``task x condition`` work items for one ICIL track. Context is defined *by
reference*: the expanded task dict carries flat ``icil_*`` fields that vla-eval forwards to the
model server at ``EPISODE_START`` together with ``episode_idx``, and the server recomputes the same
context spec with the shared sampler. The wrapper itself never needs demonstration data.

vla-eval keys results, sqlite merging and the ``tasks:`` filter on ``task["name"]``, so every
expanded task gets a unique name ``<stem>|<condition>`` while the inner benchmark still receives
the true instruction as ``name`` (it becomes ``task_description``, which the server strips under
``language: none``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from vla_eval.benchmarks.base import (  # noqa: E402  (import order kept for clarity)
    StepBenchmark,
    StepResult,
)

from icil_eval.context.types import Condition
from icil_eval.registry.io import Registry, load_registry
from icil_eval.registry.relations import normalize_language
from icil_eval.registry.schema import Task

logger = logging.getLogger("icil_eval.benchmark")

# Inner-benchmark constructor kwargs exposed explicitly so vla-eval's HELLO ``observation_params``
# auto-merge (which only matches explicit parameters of the *outer* class) reaches the inner env.
FORWARDED_INNER_KWARGS = (
    "suite",
    "seed",
    "num_steps_wait",
    "max_steps",
    "send_wrist_image",
    "send_state",
    "absolute_action",
    "env_seed",
    "quat_no_antipodal",
    "resolution",
)


def resolve_import_string(spec: str):
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:Class', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def _accepted_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    accepts_var = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return {k: v for k, v in kwargs.items() if accepts_var or k in sig.parameters}


class ICILBenchmark(StepBenchmark):
    """``task x condition`` expansion of an inner vla-eval benchmark for one ICIL track."""

    render_backends = frozenset({"gpu", "cpu"})

    @classmethod
    def configure_render(cls, mode: str) -> Dict[str, str]:  # MuJoCo-family inner benchmarks (v1)
        from vla_eval.render import configure_mujoco_render

        return configure_mujoco_render(mode)

    def __init__(
        self,
        inner: str = "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
        track: str = "configuration",
        provider: str = "libero",
        registry_root: Optional[str] = None,
        conditions: Optional[Sequence[str]] = None,
        query_tasks: Optional[Sequence[str]] = None,
        max_query_tasks: Optional[int] = None,
        seed: int = 1000,
        capabilities: Optional[str] = None,
        suite: Optional[str] = None,
        num_steps_wait: Optional[int] = None,
        max_steps: Optional[int] = None,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        env_seed: Optional[int] = None,
        quat_no_antipodal: bool = False,
        resolution: Optional[int] = None,
        inner_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.track_name = track
        self.provider = provider
        self.seed = int(seed)
        self.registry: Registry = load_registry(Path(registry_root) if registry_root else None)
        self.registry_hash = self.registry.registry_hash()
        if track not in self.registry.tracks:
            raise KeyError(f"unknown track '{track}'; known: {sorted(self.registry.tracks)}")
        self.track = self.registry.tracks[track]
        self.conditions = [Condition.parse(c) for c in (conditions or self.track.conditions)]
        self.query_tasks = set(query_tasks) if query_tasks else None
        self.max_query_tasks = max_query_tasks
        self.k_max_by_task: Dict[str, int] = self._load_capabilities(capabilities)

        inner_cls = resolve_import_string(inner)
        kwargs: Dict[str, Any] = dict(inner_params or {})
        explicit = {
            "suite": suite,
            "seed": seed,
            "num_steps_wait": num_steps_wait,
            "max_steps": max_steps,
            "send_wrist_image": send_wrist_image,
            "send_state": send_state,
            "absolute_action": absolute_action,
            "env_seed": env_seed,
            "quat_no_antipodal": quat_no_antipodal,
            "resolution": resolution,
        }
        kwargs.update({k: v for k, v in explicit.items() if v is not None})
        self.inner = inner_cls(**_accepted_kwargs(inner_cls, kwargs))
        # vla-eval reads recording field names from the outer benchmark
        self._ALL_RECORD_FIELDS = getattr(self.inner, "_ALL_RECORD_FIELDS", ())
        self._current: Dict[str, Any] = {}
        self._progress_max: Optional[float] = None
        self._progress_baseline: Optional[int] = None
        self.skipped: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_capabilities(path: Optional[str]) -> Dict[str, int]:
        if not path:
            return {}
        import json

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return {str(k): int(v) for k, v in (payload.get("k_max_by_task") or payload).items()}

    def task_id_for(self, inner_task: Dict[str, Any]) -> Optional[str]:
        """Map an inner task dict to a registry task id (LIBERO: from the BDDL file stem)."""
        obj = inner_task.get("task_obj")
        bddl = getattr(obj, "bddl_file", None)
        suite = inner_task.get("suite")
        if bddl and suite:
            stem = str(bddl).rsplit("/", 1)[-1]
            stem = stem[:-5] if stem.endswith(".bddl") else stem
            task_id = f"{self.provider}/{suite}/{stem}"
            if task_id in self.registry.tasks:
                return task_id
        lang = normalize_language(str(inner_task.get("name", "")))
        matches = [
            t.task_id
            for t in self.registry.tasks.values()
            if t.provider == self.provider
            and (suite is None or t.suite == suite)
            and normalize_language(t.language) == lang
        ]
        return matches[0] if len(matches) == 1 else None

    def inner_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """The task dict the inner benchmark expects (true instruction restored as ``name``)."""
        if "icil_language" not in task:
            return task
        return {**task, "name": task["icil_language"]}

    # ------------------------------------------------------------------ vla-eval Benchmark API
    def get_tasks(self) -> List[Dict[str, Any]]:
        pool = self.registry.pools.get((self.track_name, self.provider), {})
        expanded: List[Dict[str, Any]] = []
        n_query = 0
        for inner_task in self.inner.get_tasks():
            task_id = self.task_id_for(inner_task)
            if task_id is None:
                logger.warning(
                    "no registry task for inner task %r; skipped", inner_task.get("name")
                )
                continue
            if task_id not in pool or (self.query_tasks and task_id not in self.query_tasks):
                continue
            if self.max_query_tasks is not None and n_query >= self.max_query_tasks:
                break
            n_query += 1
            reg_task: Task = self.registry.tasks[task_id]
            k_max = self.k_max_by_task.get(task_id)
            for cond in self.conditions:
                if k_max is not None and cond.k is not None and cond.k > k_max:
                    self.skipped.append(
                        {"task_id": task_id, "condition": cond.name, "k_max": k_max}
                    )
                    continue
                if cond.wrong_task and not pool[task_id].wrong:
                    self.skipped.append(
                        {"task_id": task_id, "condition": cond.name, "reason": "no_wrong_task"}
                    )
                    continue
                expanded.append(
                    {
                        **inner_task,
                        "name": f"{reg_task.stem}|{cond.name}",
                        "icil_language": str(inner_task.get("name", reg_task.language)),
                        "icil_task_id": task_id,
                        "icil_provider": self.provider,
                        "icil_track": self.track_name,
                        "icil_condition": cond.name,
                        "icil_k": -1 if cond.k is None else int(cond.k),
                        "icil_control": cond.control,
                        "icil_language_mode": self.track.language,
                        "icil_seed": self.seed,
                        "icil_registry_hash": self.registry_hash,
                    }
                )
        if self.skipped:
            logger.info(
                "skipped %d task/condition pairs (see get_metadata()['skipped'])", len(self.skipped)
            )
        return expanded

    def reset(self, task: Dict[str, Any]) -> Any:
        self._current = task
        # the inner benchmark records video/steps through the recorder the orchestrator gave us
        self.inner._recorder = self._recorder
        self.inner._task = self.inner_task(task)
        self.inner._t0 = self._t0
        raw = self.inner.reset(self.inner._task)
        self._progress_baseline = self._goal_satisfied_count()
        self._progress_max = None
        self._update_progress()
        return raw

    def step(self, action: Dict[str, Any]) -> StepResult:
        result = self.inner.step(action)
        self._update_progress()
        return result

    def make_obs(self, raw_obs: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        obs = self.inner.make_obs(raw_obs, self.inner_task(task))
        if self.track.language == "none" and isinstance(obs, dict):
            obs = dict(obs)
            obs.pop("task_description", None)
        return obs

    def check_done(self, step_result: StepResult) -> bool:
        return bool(self.inner.check_done(step_result))

    def get_step_result(self, step_result: StepResult) -> Dict[str, Any]:
        result = dict(self.inner.get_step_result(step_result))
        result["progress"] = self._progress_max
        result["icil_task_id"] = self._current.get("icil_task_id")
        result["icil_condition"] = self._current.get("icil_condition")
        result["icil_track"] = self.track_name
        return result

    def get_metric_keys(self) -> Dict[str, str]:
        return {"success": "mean", "progress": "mean"}

    def get_metadata(self) -> Dict[str, Any]:
        meta = dict(self.inner.get_metadata())
        meta.update(
            {
                "icil_track": self.track_name,
                "icil_provider": self.provider,
                "icil_registry_hash": self.registry_hash,
                "icil_conditions": [c.name for c in self.conditions],
                "skipped": list(self.skipped),
            }
        )
        return meta

    def get_action_spec(self):
        return self.inner.get_action_spec()

    def get_observation_spec(self):
        return self.inner.get_observation_spec()

    def get_hold_action(self, last_action: Any = None):
        fn = getattr(self.inner, "get_hold_action", None)
        return fn(last_action) if fn else super().get_hold_action(last_action)

    def render(self):
        fn = getattr(self.inner, "render", None)
        return fn() if fn else None

    def cleanup(self) -> None:
        fn = getattr(self.inner, "cleanup", None)
        if fn:
            fn()

    # ------------------------------------------------------------------ progress
    def _goal_satisfied_count(self) -> Optional[int]:
        fn = getattr(self.inner, "goal_predicate_status", None)
        if fn is None:
            return None
        try:
            status = fn()
        except Exception:  # noqa: BLE001 - progress must never break a rollout
            return None
        return None if status is None else int(sum(bool(s) for s in status))

    def _goal_total(self) -> Optional[int]:
        fn = getattr(self.inner, "goal_predicate_status", None)
        if fn is None:
            return None
        try:
            status = fn()
        except Exception:  # noqa: BLE001
            return None
        return None if status is None else len(status)

    def _update_progress(self) -> None:
        if self._progress_baseline is None:
            return
        total = self._goal_total()
        satisfied = self._goal_satisfied_count()
        if total is None or satisfied is None:
            return
        denom = total - self._progress_baseline
        if denom <= 0:
            self._progress_max = None
            return
        value = max(0.0, (satisfied - self._progress_baseline) / denom)
        self._progress_max = value if self._progress_max is None else max(self._progress_max, value)
