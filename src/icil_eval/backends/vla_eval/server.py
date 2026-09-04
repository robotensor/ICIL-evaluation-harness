"""ICIL model server for the vla-eval backend.

Bridges the ICIL policy protocol to vla-eval's ``PredictModelServer``:

- ``EPISODE_START`` carries the flat task fields vla-eval forwards (``icil_*`` fields when the ICIL
  benchmark wrapper is used, plus ``episode_idx``); the server recomputes the context spec with the
  shared sampler, loads the demonstrations, applies transforms and calls ``policy.set_context``.
- every observation reaches ``policy.observe`` (history policies need every control step), and
  ``task_description`` is stripped when the track's language mode is ``none``;
- ``predict`` returns exactly ``exec_horizon`` actions from ``policy.act``;
- one JSONL row per episode (context spec, status, latencies) is written for the scorer.

Stock vla-eval configs (no ``icil_*`` fields) also work: the query task is resolved from the suite
and instruction, and the server's default track/condition apply.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from icil_eval.context.sampler import sample_context
from icil_eval.context.transforms import apply_transform
from icil_eval.context.types import Condition, ContextSpec, Demonstration, UnsupportedContext
from icil_eval.policy.protocol import (
    ICILPolicy,
    Observation,
    PolicyFactory,
    TaskInfo,
    resolve_k_max,
)
from icil_eval.registry.io import Registry
from icil_eval.registry.relations import normalize_language
from icil_eval.registry.schema import Task

try:  # the server runs where vla-eval is installed; keep the module importable without it
    from vla_eval.model_servers.base import SessionContext
    from vla_eval.model_servers.predict import PredictModelServer
    from vla_eval.specs import (
        GRIPPER_CLOSE_POS,
        IMAGE_RGB,
        POSITION_DELTA,
        ROTATION_AA,
        STATE_EEF_POS_AA_GRIP,
    )
except ImportError:  # pragma: no cover - exercised only without the server extra
    PredictModelServer = object  # type: ignore[misc,assignment]
    SessionContext = Any  # type: ignore[misc,assignment]

logger = logging.getLogger("icil_eval.server")

DEFAULT_CAMERA_MAP = {"agentview": "image", "wrist": "image2", "image": "image", "image2": "image2"}
LIBERO_HOLD_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


class DemoStore:
    """Structural interface of a demonstration store keyed by registry task."""

    def load(self, task: Task, episode_index: int) -> Demonstration:  # pragma: no cover
        raise NotImplementedError

    def episode_lengths(self, task: Task) -> List[int]:  # pragma: no cover
        raise NotImplementedError

    def available(self, task: Task) -> bool:  # pragma: no cover
        raise NotImplementedError


@dataclass
class _Session:
    policy: Optional[ICILPolicy]
    status: str  # ok | unsupported | error
    task: Optional[Task]
    language_mode: str
    spec: Optional[ContextSpec] = None
    context_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    control_latencies: List[float] = field(default_factory=list)
    n_observations: int = 0
    started_at: float = field(default_factory=time.time)
    task_fields: Dict[str, Any] = field(default_factory=dict)
    k_max: Optional[int] = None
    min_context_init_l2: Optional[float] = None


class ICILModelServer(PredictModelServer):  # type: ignore[misc]
    def __init__(
        self,
        policy_factory: PolicyFactory,
        registry: Registry,
        store: DemoStore,
        *,
        seed: int = 1000,
        default_track: str = "configuration",
        default_condition: str = "k1",
        camera_map: Optional[Dict[str, str]] = None,
        hold_action: Sequence[float] = LIBERO_HOLD_ACTION,
        log_path: Optional[Path] = None,
        registry_hash: Optional[str] = None,
        max_batch_size: int = 1,
    ) -> None:
        probe = policy_factory()
        self.policy_spec = probe.spec
        del probe
        super().__init__(chunk_size=self.policy_spec.exec_horizon, max_batch_size=max_batch_size)
        self.policy_factory = policy_factory
        self.registry = registry
        self.registry_hash = registry_hash or registry.registry_hash()
        self.store = store
        self.seed = seed
        self.default_track = default_track
        self.default_condition = default_condition
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self.hold_action = np.asarray(hold_action, dtype=np.float32)
        self.log_path = Path(log_path) if log_path else None
        self.sessions: Dict[str, _Session] = {}
        self._k_max_cache: Dict[str, int] = {}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "policy_spec": self.policy_spec.to_dict(),
                "registry_hash": self.registry_hash,
                "seed": self.seed,
                "default_track": self.default_track,
                "default_condition": self.default_condition,
                "camera_map": self.camera_map,
            }
            self.log_path.with_suffix(".meta.json").write_text(
                json.dumps(meta, indent=1, default=str)
            )

    # ------------------------------------------------------------------ vla-eval declarations
    def get_observation_params(self) -> Dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self):
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self):
        return {"image": IMAGE_RGB, "state": STATE_EEF_POS_AA_GRIP}

    # ------------------------------------------------------------------ task resolution
    def resolve_task(self, fields: Dict[str, Any]) -> Task:
        task_id = fields.get("icil_task_id")
        if task_id:
            return self.registry.tasks[task_id]
        suite = fields.get("suite")
        lang = normalize_language(str(fields.get("name", "")))
        matches = [
            t
            for t in self.registry.tasks.values()
            if (suite is None or t.suite == suite) and normalize_language(t.language) == lang
        ]
        if len(matches) != 1:
            raise KeyError(
                f"cannot resolve task from suite={suite!r} name={fields.get('name')!r}: "
                f"{len(matches)} matches"
            )
        return matches[0]

    def k_max_for(self, task: Task, k_sweep: Sequence[int]) -> int:
        if task.task_id not in self._k_max_cache:
            lengths = self.store.episode_lengths(task)
            self._k_max_cache[task.task_id] = resolve_k_max(self.policy_spec, k_sweep, lengths)
        return self._k_max_cache[task.task_id]

    # ------------------------------------------------------------------ episode lifecycle
    async def on_episode_start(self, config: Dict[str, Any], ctx: SessionContext) -> None:
        fields = dict(config.get("task") or {})
        session = _Session(
            policy=None, status="ok", task=None, language_mode="none", task_fields=fields
        )
        self.sessions[ctx.session_id] = session
        try:
            self._start_session(session, fields)
        except UnsupportedContext as e:
            session.status, session.error, session.policy = "unsupported", str(e), None
            logger.info("unsupported condition for session %s: %s", ctx.session_id[:8], e)
        except Exception as e:  # noqa: BLE001 - any failure must not kill the server
            session.status, session.error, session.policy = (
                "error",
                f"{type(e).__name__}: {e}",
                None,
            )
            logger.exception("episode start failed for session %s", ctx.session_id[:8])
        await super().on_episode_start(config, ctx)

    def _start_session(self, session: _Session, fields: Dict[str, Any]) -> None:
        task = self.resolve_task(fields)
        session.task = task
        track_name = fields.get("icil_track", self.default_track)
        track = self.registry.tracks[track_name]
        session.language_mode = fields.get("icil_language", track.language)
        condition = Condition.parse(fields.get("icil_condition", self.default_condition))
        seed = int(fields.get("icil_seed", self.seed))
        episode_idx = int(fields.get("episode_idx", 0))
        expected_hash = fields.get("icil_registry_hash")
        if expected_hash and expected_hash != self.registry_hash:
            raise RuntimeError(
                f"registry hash mismatch: benchmark {expected_hash} vs server {self.registry_hash}"
            )

        k_max = self.k_max_for(task, track.k_sweep)
        session.k_max = k_max
        condition = condition.resolve(k_max)
        if condition.k > k_max:
            raise UnsupportedContext(
                f"k={condition.k} exceeds supported k_max={k_max} for {task.task_id}"
            )

        n_demos = {}
        spec = sample_context(
            self.registry,
            self.registry_hash,
            track_name,
            task.task_id,
            condition,
            episode_idx,
            seed,
            n_demos or None,
        )
        session.spec = spec
        demos = self._load_demos(spec, track.context_chunk_seconds)
        if demos and demos[0].init_state is not None:
            session.min_context_init_l2 = self._min_init_distance(task, episode_idx, demos)

        policy = self.policy_factory()
        policy.reset()
        info = policy.set_context(
            demos,
            TaskInfo(
                task_id=task.task_id,
                provider=task.provider,
                suite=task.suite,
                language=task.language if session.language_mode != "none" else None,
                embodiment=task.embodiment.__dict__,
            ),
        )
        session.policy = policy
        session.context_info = info.to_dict()

    def _load_demos(self, spec: ContextSpec, chunk_seconds: float) -> List[Demonstration]:
        demos: List[Demonstration] = []
        for i, ref in enumerate(spec.refs):
            task = self.registry.tasks[ref.task_id]
            demo = self.store.load(task, ref.episode_index)
            if spec.transform != "identity":
                chunk_len = max(1, int(round(chunk_seconds * demo.fps)))
                demo = apply_transform(
                    demo, spec.transform, chunk_len, (spec.permutation_seed or 0) + i
                )
            demos.append(demo)
        return demos

    def _min_init_distance(
        self, task: Task, episode_idx: int, demos: List[Demonstration]
    ) -> Optional[float]:
        loader = getattr(self.store, "eval_init_state", None)
        if loader is None:
            return None
        try:
            init = np.asarray(loader(task, episode_idx), dtype=np.float64)
        except Exception:
            return None
        dists = [
            float(np.linalg.norm(init - d.init_state))
            for d in demos
            if d.init_state is not None and d.init_state.shape == init.shape
        ]
        return min(dists) if dists else None

    async def on_observation(self, obs: Dict[str, Any], ctx: SessionContext) -> None:
        session = self.sessions.get(ctx.session_id)
        if session is None:
            logger.warning(
                "observation before EPISODE_START on session %s; using defaults", ctx.session_id[:8]
            )
            await self.on_episode_start({"task": {}}, ctx)
            session = self.sessions[ctx.session_id]
        if session.language_mode == "none" and "task_description" in obs:
            obs = dict(obs)
            obs.pop("task_description", None)
        if session.policy is not None:
            try:
                session.policy.observe(self._to_observation(obs, session), None)
            except Exception as e:  # noqa: BLE001
                session.status, session.error, session.policy = (
                    "error",
                    f"observe: {type(e).__name__}: {e}",
                    None,
                )
                logger.exception("observe failed for session %s", ctx.session_id[:8])
        session.n_observations += 1
        await super().on_observation(obs, ctx)

    def _to_observation(self, obs: Dict[str, Any], session: _Session) -> Observation:
        raw_images = obs.get("images") or {}
        images = {}
        for cam, arr in raw_images.items():
            images[self.camera_map.get(cam, cam)] = np.asarray(arr)
        state = obs.get("states", obs.get("state"))
        state = None if state is None else np.asarray(state, dtype=np.float32).reshape(-1)
        return Observation(
            images=images,
            state=state,
            language=obs.get("task_description") if session.language_mode != "none" else None,
            step=session.n_observations,
        )

    def predict(self, obs: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
        session = self.sessions.get(ctx.session_id)
        if session is None or session.policy is None:
            hold = np.tile(self.hold_action, (self.policy_spec.exec_horizon, 1))
            return {"actions": hold}
        t0 = time.monotonic()
        try:
            chunk = session.policy.act()
        except Exception as e:  # noqa: BLE001
            session.status, session.error, session.policy = (
                "error",
                f"act: {type(e).__name__}: {e}",
                None,
            )
            logger.exception("act failed for session %s", ctx.session_id[:8])
            return {"actions": np.tile(self.hold_action, (self.policy_spec.exec_horizon, 1))}
        session.control_latencies.append(time.monotonic() - t0)
        actions = np.asarray(chunk.actions, dtype=np.float32)[: chunk.exec_horizon]
        return {"actions": actions}

    async def on_episode_end(self, result: Dict[str, Any], ctx: SessionContext) -> None:
        session = self.sessions.pop(ctx.session_id, None)
        if session is not None:
            self._write_log(session, ctx, result)
        await super().on_episode_end(result, ctx)

    def _write_log(self, session: _Session, ctx: SessionContext, result: Dict[str, Any]) -> None:
        row = {
            "eval_id": ctx.eval_id,
            "session_id": ctx.session_id,
            "episode_id": ctx.episode_id,
            "task_fields": {
                k: v
                for k, v in session.task_fields.items()
                if isinstance(v, (str, int, float, bool, list))
            },
            "task_id": session.task.task_id if session.task else None,
            "status": session.status,
            "error": session.error,
            "language_mode": session.language_mode,
            "k_max": session.k_max,
            "context": session.spec.to_dict() if session.spec else None,
            "context_info": session.context_info,
            "min_context_init_l2": session.min_context_init_l2,
            "adaptation_latency_s": session.context_info.get("adaptation_latency_s"),
            "control_latency_s": {
                "median": statistics.median(session.control_latencies)
                if session.control_latencies
                else None,
                "mean": statistics.fmean(session.control_latencies)
                if session.control_latencies
                else None,
                "n": len(session.control_latencies),
            },
            "n_observations": session.n_observations,
            "policy": self.policy_spec.name,
            "result": {
                k: v
                for k, v in (result or {}).items()
                if isinstance(v, (str, int, float, bool, list))
            },
            "wall_s": time.time() - session.started_at,
        }
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        logger.info(
            "episode %s task=%s status=%s condition=%s success=%s",
            ctx.episode_id[:8],
            row["task_id"],
            session.status,
            session.spec.condition if session.spec else None,
            (result or {}).get("success"),
        )
