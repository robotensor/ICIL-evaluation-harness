"""Context standard types (S2): demonstrations, context references and context specs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

CONTEXT_SCHEMA_VERSION = 1

_CONDITION = re.compile(r"^k(?P<k>\d+|max)(?:\.(?P<control>[a-z_]+))?$")


class UnsupportedContext(Exception):
    """Raised by a policy that cannot accept the requested context (K, length, modality)."""


@dataclass(frozen=True)
class Condition:
    """An evaluation condition such as ``k1``, ``k4``, ``k1.wrong_task`` or ``kmax.wrong_task``."""

    k: Optional[int]  # None means 'kmax' (largest K the policy supports)
    control: str = "none"

    @property
    def name(self) -> str:
        k = "max" if self.k is None else str(self.k)
        return f"k{k}" if self.control == "none" else f"k{k}.{self.control}"

    @property
    def transform(self) -> str:
        return (
            self.control if self.control in ("shuffled_chunks", "reversed_actions") else "identity"
        )

    @property
    def wrong_task(self) -> bool:
        return self.control == "wrong_task"

    @classmethod
    def parse(cls, name: str) -> Condition:
        m = _CONDITION.match(name)
        if not m:
            raise ValueError(f"malformed condition '{name}' (expected e.g. k1, k4, k1.wrong_task)")
        k = None if m.group("k") == "max" else int(m.group("k"))
        return cls(k=k, control=m.group("control") or "none")

    def resolve(self, k_max: int) -> Condition:
        return self if self.k is not None else Condition(k=k_max, control=self.control)


@dataclass(frozen=True)
class ContextRef:
    """A demonstration reference: an episode of a task's demonstration pool."""

    task_id: str
    episode_index: int

    def to_list(self) -> List[Any]:
        return [self.task_id, self.episode_index]


@dataclass
class ContextSpec:
    """Everything needed to rebuild the context handed to a policy for one episode."""

    track: str
    query_task_id: str
    condition: str
    k: int
    refs: List[ContextRef]
    relation: str  # relation of the context task(s) to the query
    context_task_id: Optional[str]
    transform: str
    language: str
    modality: str
    seed: int
    episode_idx: int
    registry_hash: str
    context_hash: str = ""
    permutation_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["refs"] = [r.to_list() for r in self.refs]
        return d


@dataclass
class Demonstration:
    """One demonstration held in memory (T frames)."""

    images: Dict[str, np.ndarray]  # cam -> (T, H, W, 3) uint8, standard image convention
    state: Optional[np.ndarray]  # (T, S) float32
    action: Optional[np.ndarray]  # (T, A) float32
    fps: float
    task: Optional[str] = None
    robot_type: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)
    rewards: Optional[np.ndarray] = None  # (T,) float32
    init_state: Optional[np.ndarray] = None  # simulator state at t=0 (provider-specific)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        if self.action is not None:
            return int(self.action.shape[0])
        first = next(iter(self.images.values()))
        return int(first.shape[0])
