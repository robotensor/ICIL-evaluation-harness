"""The ICIL policy protocol (standard S3).

A policy is evaluated per episode: ``reset()``, ``set_context(demos)`` once, ``observe(obs)`` on
every control step, and ``act()`` whenever the harness needs a new action chunk. ``PolicySpec``
declares capabilities so the harness can decide applicability and ``unsupported`` conditions
*before* any rollout, and ``PolicyCard`` discloses training data for exposure tagging.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from icil_eval.context.types import Demonstration, UnsupportedContext

POLICY_SPEC_VERSION = 1

__all__ = [
    "ActionChunk",
    "ContextInfo",
    "ICILPolicy",
    "Observation",
    "PolicyCard",
    "PolicySpec",
    "TaskInfo",
    "UnsupportedContext",
    "chunks_needed",
    "resolve_k_max",
]


@dataclass
class TaskInfo:
    """What a policy may know about the query task (language is None under ``language: none``)."""

    task_id: str
    provider: str
    suite: str
    language: Optional[str]
    embodiment: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """One control-step observation in the provider's standard conventions."""

    images: Dict[str, np.ndarray]  # camera -> (H, W, 3) uint8
    state: Optional[
        np.ndarray
    ]  # provider state layout (LIBERO: [eef_pos, eef_axis_angle, gripper])
    language: Optional[str]
    step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionChunk:
    actions: np.ndarray  # (H, A) float32 in the provider's action space
    exec_horizon: int  # how many leading actions the harness should execute before asking again


@dataclass
class ContextInfo:
    n_demos: int
    n_frames: int
    adaptation_latency_s: float
    chunks_used: Optional[int] = None
    truncated: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyCard:
    """Training-data disclosure used for ``query_exposure`` / ``context_exposure`` tagging."""

    name: str
    version: str
    exposure_source: str  # checkpoint_metadata | self_reported | undisclosed
    training_data: List[Dict[str, Any]] = field(default_factory=list)
    training_tasks: Optional[List[str]] = (
        None  # task ids or instruction strings; None = undisclosed
    )
    training_episodes: Optional[Dict[str, List[int]]] = None  # task -> demo indices when known
    url: Optional[str] = None
    license: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicySpec:
    name: str
    context_mode: str  # sequence | retrieval
    k_min: int
    k_max: Optional[int]  # hard cap on demonstrations, None = unbounded (budget-limited only)
    context_budget: Dict[str, Any]  # {chunks, chunk_len} | {frames} | {seconds} | {}
    needs_every_observation: bool
    order_invariant: bool
    requires_language: bool
    k0_semantics: str  # none | blank_prompt | native
    cameras: List[str]
    action_space: str
    action_horizon: int
    exec_horizon: int
    control_hz: float
    card: PolicyCard
    image_size: Optional[int] = None
    state_layout: Optional[str] = None
    context_modalities: List[str] = field(default_factory=lambda: ["sensorimotor"])
    embodiments: List[str] = field(default_factory=lambda: ["panda"])
    spec_version: int = POLICY_SPEC_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ICILPolicy:
    """Structural interface every wrapped policy implements (duck-typed; subclassing optional)."""

    spec: PolicySpec

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def set_context(self, demos: List[Demonstration], task: Optional[TaskInfo]) -> ContextInfo:
        raise NotImplementedError  # pragma: no cover - interface

    def observe(self, obs: Observation, executed_actions: Optional[np.ndarray]) -> None:
        raise NotImplementedError  # pragma: no cover - interface

    def act(self) -> ActionChunk:  # pragma: no cover - interface
        raise NotImplementedError


# ----------------------------------------------------------------------------- capability math
def chunks_needed(lengths: Sequence[int], chunk_len: int) -> int:
    """Prompt chunks a sequence-mode policy needs for demonstrations of the given lengths."""
    return sum(max(1, math.ceil(int(t) / chunk_len)) for t in lengths)


def fits_budget(spec: PolicySpec, lengths: Sequence[int], fps: float = 20.0) -> bool:
    budget = spec.context_budget or {}
    if "chunks" in budget:
        return chunks_needed(lengths, int(budget.get("chunk_len", 1))) <= int(budget["chunks"])
    if "frames" in budget:
        return sum(int(t) for t in lengths) <= int(budget["frames"])
    if "seconds" in budget:
        return sum(int(t) for t in lengths) / fps <= float(budget["seconds"])
    return True


def resolve_k_max(spec: PolicySpec, k_sweep: Sequence[int], pool_lengths: Sequence[int]) -> int:
    """Largest K in the sweep supported for every possible draw from a demonstration pool.

    Worst case = the K longest demonstrations; using it makes every episode of a condition either
    fully supported or reported ``unsupported``, never a mix.
    """
    longest = sorted((int(t) for t in pool_lengths), reverse=True)
    best = -1
    for k in sorted(k_sweep):
        if k < spec.k_min or (spec.k_max is not None and k > spec.k_max):
            continue
        if k > len(longest) or not fits_budget(spec, longest[:k]):
            continue
        best = k
    if best < 0:
        raise UnsupportedContext(f"policy '{spec.name}' supports no K in {list(k_sweep)}")
    return best


def supports_k(spec: PolicySpec, k: int, pool_lengths: Sequence[int]) -> bool:
    try:
        return k <= resolve_k_max(spec, [k], pool_lengths)
    except UnsupportedContext:
        return False


PolicyFactory = Callable[[], ICILPolicy]
