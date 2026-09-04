"""Task providers normalise existing benchmarks into the unified registry.

A provider enumerates tasks with capability tags, exports demonstrations to LeRobotDataset v3,
and binds a rollout backend. Simulators are imported lazily, never at module level.
"""

from icil_eval.providers.base import PROVIDER_CLASSES, TaskProvider, get_provider

__all__ = ["PROVIDER_CLASSES", "TaskProvider", "get_provider"]
