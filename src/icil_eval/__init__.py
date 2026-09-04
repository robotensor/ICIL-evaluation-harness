"""ICIL-evaluation-harness: one global benchmark standard for In-Context Imitation Learning.

The package is organised in layers, from the standard outward:

- ``icil_eval.registry``  unified task/track schema, loading, hashing (Python 3.8-safe)
- ``icil_eval.providers`` task providers that normalise existing benchmarks into the registry
- ``icil_eval.context``   demonstrations as context: types, deterministic sampler, transforms
- ``icil_eval.policy``    the ICIL policy protocol and policy card
- ``icil_eval.backends``  environment/rollout backends (vla-eval first)
- ``icil_eval.scoring``   metrics, model profile, results
"""

__version__ = "0.1.0.dev0"
