"""Behavior Prompting Policy (real-stanford/behavior_prompting) wrapper.

``BPPModel`` loads a checkpoint once; ``BPPPolicy`` is a lightweight per-session view that holds
the encoded context and the observation history, so many vla-eval shard sessions can share one
GPU-resident model.
"""
