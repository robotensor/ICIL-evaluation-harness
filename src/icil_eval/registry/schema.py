"""Unified task registry schema (standard S1; Python 3.8-safe dataclasses).

Every task from every provider is described by :class:`Task`. Providers are described by
:class:`ProviderInfo`, tracks by :class:`Track`, and materialised context/wrong-task pools by
:class:`PoolEntry`. All types round-trip through plain dicts (``to_dict``/``from_dict``) so the
committed YAML is the single source of truth and needs no code to be read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

TASK_SCHEMA_VERSION = 1
TRACK_SCHEMA_VERSION = 1
PROVIDER_SCHEMA_VERSION = 1
POOL_SCHEMA_VERSION = 1

PROMPT_MODALITIES = ("sensorimotor", "robot_video", "human_video", "cross_robot", "language_demo")
LANGUAGE_MODES = ("none", "context", "query")
TRANSFORMS = ("identity", "shuffled_chunks", "reversed_actions")
CONTROLS = ("none", "wrong_task", "shuffled_chunks", "reversed_actions")


@dataclass
class Embodiment:
    robot: str
    gripper: Optional[str] = None
    dof: Optional[int] = None
    action_space: Optional[str] = None
    control_hz: Optional[float] = None


@dataclass
class Scene:
    layout_id: str
    init_id: str
    scene_name: Optional[str] = None
    entities: List[str] = field(default_factory=list)


@dataclass
class ObjectRef:
    name: str
    category: str
    role: str  # target | fixture | distractor


@dataclass
class Goal:
    predicates: List[List[str]]
    n_effective_predicates: int
    initially_satisfied: List[List[str]] = field(default_factory=list)


@dataclass
class Horizon:
    max_steps: int
    n_steps_wait: int = 0


@dataclass
class InitStates:
    file: str
    n: int
    sha256: Optional[str] = None
    verified: bool = False


@dataclass
class DemoPool:
    repo_id: str
    path: str
    n: int
    format: str = "libero_hdf5"
    revision: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    lerobot_repo_id: Optional[str] = None


@dataclass
class Bounds:
    chance_success: Optional[float] = None
    expert_success: Optional[float] = None


@dataclass
class Capabilities:
    relations: List[str] = field(default_factory=list)
    prompt_modalities: List[str] = field(default_factory=lambda: ["sensorimotor"])
    perturbations: List[str] = field(default_factory=list)
    progress_metric: bool = False


@dataclass
class Task:
    task_id: str
    provider: str
    suite: str
    language: str
    embodiment: Embodiment
    scene: Scene
    objects: List[ObjectRef]
    skills: List[str]
    goal: Goal
    horizon: Horizon
    init_states: InitStates
    demo_pool: DemoPool
    bounds: Bounds = field(default_factory=Bounds)
    capabilities: Capabilities = field(default_factory=Capabilities)
    source_version: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.task_id.rsplit("/", 1)[1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Task:
        return cls(
            task_id=d["task_id"],
            provider=d["provider"],
            suite=d["suite"],
            language=d["language"],
            embodiment=Embodiment(**d["embodiment"]),
            scene=Scene(**d["scene"]),
            objects=[ObjectRef(**o) for o in d.get("objects", [])],
            skills=list(d.get("skills", [])),
            goal=Goal(**d["goal"]),
            horizon=Horizon(**d["horizon"]),
            init_states=InitStates(**d["init_states"]),
            demo_pool=DemoPool(**d["demo_pool"]),
            bounds=Bounds(**d.get("bounds", {})),
            capabilities=Capabilities(**d.get("capabilities", {})),
            source_version=dict(d.get("source_version", {})),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class ProviderInfo:
    name: str
    description: str
    license: str
    url: str
    version: Dict[str, str]
    suites: List[str]
    n_tasks: int
    backend: Dict[str, Any]
    capability_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    citation: Optional[str] = None
    schema_version: int = PROVIDER_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProviderInfo:
        return cls(**d)


@dataclass
class Track:
    name: str
    description: str
    relation: str
    language: str
    wrong_task_rule: Dict[str, Any]
    k_sweep: List[int]
    conditions: List[str]
    k_ref: int = 1
    context_chunk_seconds: float = 0.5
    version: int = TRACK_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Track:
        return cls(**d)


@dataclass
class PoolEntry:
    """Materialised pools for one query task within one track."""

    task_id: str
    context: List[str]
    wrong: List[List[str]]  # [task_id, relation_label]
    flags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PoolEntry:
        return cls(
            task_id=d["task_id"],
            context=list(d.get("context", [])),
            wrong=[list(x) for x in d.get("wrong", [])],
            flags=dict(d.get("flags", {})),
        )
