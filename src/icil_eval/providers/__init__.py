"""Task providers normalise existing benchmarks into the unified registry.

A provider enumerates tasks with capability tags, exports demonstrations to LeRobotDataset v3,
declares which context<->query relations it can construct, and binds a rollout backend.
Simulators are imported lazily, never at module level.
"""
