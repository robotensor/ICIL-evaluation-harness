"""Exercise ICILModelServer with a fake policy and fake demonstration store (no GPU, no sim)."""

from __future__ import annotations

import json
import shutil
from typing import List, Optional

import anyio
import numpy as np
import pytest

from icil_eval.backends.vla_eval.server import ICILModelServer
from icil_eval.context.types import Demonstration, UnsupportedContext
from icil_eval.paths import REGISTRY_ROOT
from icil_eval.policy import ActionChunk, ContextInfo, Observation, PolicyCard, PolicySpec, TaskInfo
from icil_eval.registry.build import build_provider
from icil_eval.registry.io import load_registry

STOVE = "libero/libero_goal/turn_on_the_stove"


class FakeStore:
    def __init__(self, lengths=None):
        self.lengths = lengths or [60] * 50

    def episode_lengths(self, task):
        return list(self.lengths)

    def available(self, task):
        return True

    def load(self, task, episode_index):
        T = self.lengths[episode_index]
        return Demonstration(
            images={
                "image": np.zeros((T, 8, 8, 3), np.uint8),
                "image2": np.zeros((T, 8, 8, 3), np.uint8),
            },
            state=np.zeros((T, 8), np.float32),
            action=np.full((T, 7), episode_index, np.float32),
            fps=20.0,
            task=task.language,
            init_state=np.ones(79),
        )

    def eval_init_state(self, task, episode_idx):
        return np.zeros(79)


class FakePolicy:
    def __init__(self, budget_chunks=50, log=None):
        self.spec = PolicySpec(
            name="fake",
            context_mode="sequence",
            k_min=0,
            k_max=None,
            context_budget={"chunks": budget_chunks, "chunk_len": 20},
            needs_every_observation=True,
            order_invariant=False,
            requires_language=False,
            k0_semantics="blank_prompt",
            cameras=["image", "image2"],
            action_space="osc_pose_delta_7d",
            action_horizon=16,
            exec_horizon=12,
            control_hz=20.0,
            card=PolicyCard(name="fake", version="0", exposure_source="undisclosed"),
        )
        self.log = log if log is not None else []
        self.demos: List[Demonstration] = []
        self.task: Optional[TaskInfo] = None
        self.observed: List[Observation] = []

    def reset(self):
        self.log.append("reset")

    def set_context(self, demos, task):
        chunks = sum(max(1, -(-d.length // 20)) for d in demos)
        if chunks > self.spec.context_budget["chunks"]:
            raise UnsupportedContext("too long")
        self.demos, self.task = demos, task
        self.log.append(("set_context", len(demos), task.language if task else None))
        return ContextInfo(
            n_demos=len(demos), n_frames=sum(d.length for d in demos), adaptation_latency_s=0.01
        )

    def observe(self, obs, executed_actions):
        self.observed.append(obs)

    def act(self):
        self.log.append("act")
        return ActionChunk(actions=np.full((16, 7), len(self.demos), np.float32), exec_horizon=12)


@pytest.fixture(scope="module")
def registry(tmp_path_factory, fixtures_dir):
    root = tmp_path_factory.mktemp("registry")
    shutil.copytree(REGISTRY_ROOT / "tracks", root / "tracks")
    build_provider(
        "libero",
        root=root,
        provider_options={"libero_root": str(fixtures_dir / "libero_root"), "offline": True},
    )
    return load_registry(root)


def make_ctx(session="sess-1", episode="ep-1"):
    from vla_eval.model_servers.base import SessionContext

    ctx = SessionContext(session_id=session, episode_id=episode, eval_id="eval-1")
    sent = []

    async def send(action):
        sent.append(action)

    ctx._send_action_fn = send
    return ctx, sent


def libero_obs(step):
    return {
        "images": {
            "agentview": np.zeros((16, 16, 3), np.uint8),
            "wrist": np.zeros((16, 16, 3), np.uint8),
        },
        "task_description": "Turn on the stove",
        "states": np.arange(8, dtype=np.float32) + step,
    }


def run_episode(server, task_fields, n_steps=13, session="sess-1"):
    ctx, sent = make_ctx(session)
    policies = []
    orig = server.policy_factory

    def factory():
        p = orig()
        policies.append(p)
        return p

    server.policy_factory = factory

    async def go():
        await server.on_episode_start({"task": task_fields}, ctx)
        for step in range(n_steps):
            await server.on_observation(libero_obs(step), ctx)
        await server.on_episode_end({"success": True}, ctx)

    anyio.run(go)
    return sent, policies[-1] if policies else None


def test_context_by_reference_and_every_observation_reaches_policy(registry, tmp_path):
    log_path = tmp_path / "server.jsonl"
    server = ICILModelServer(lambda: FakePolicy(), registry, FakeStore(), seed=7, log_path=log_path)
    fields = {
        "name": "Turn on the stove",
        "suite": "libero_goal",
        "task_id": 3,
        "episode_idx": 2,
        "icil_task_id": STOVE,
        "icil_track": "configuration",
        "icil_condition": "k2",
        "icil_seed": 7,
        "icil_registry_hash": registry.registry_hash(),
    }
    sent, policy = run_episode(server, fields, n_steps=13)
    # context: two demos of the same task, deterministic, matching the sampler
    assert len(policy.demos) == 2 and all(
        d.task.lower() == "turn on the stove" for d in policy.demos
    )
    assert policy.task.language is None  # language: none strips the instruction
    # every observation reached the policy; predict only every exec_horizon steps
    assert len(policy.observed) == 13 and all(o.language is None for o in policy.observed)
    assert policy.log.count("act") == 2
    assert len(sent) == 13 and sent[0]["actions"].shape == (7,)
    assert float(sent[0]["actions"][0]) == 2.0  # fake policy encodes n_demos in its actions
    assert set(policy.observed[0].images) == {"image", "image2"}
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "ok" and rows[0]["context"]["k"] == 2
    assert len(rows[0]["context"]["refs"]) == 2 and rows[0]["control_latency_s"]["n"] == 2
    assert rows[0]["min_context_init_l2"] == pytest.approx(np.sqrt(79))


def test_stock_vla_eval_task_resolves_by_suite_and_language(registry):
    server = ICILModelServer(lambda: FakePolicy(), registry, FakeStore(), default_condition="k1")
    sent, policy = run_episode(
        server,
        {"name": "Turn on the stove", "suite": "libero_goal", "task_id": 0, "episode_idx": 0},
    )
    assert len(policy.demos) == 1


def test_unsupported_condition_holds_and_is_logged(registry, tmp_path):
    log_path = tmp_path / "server.jsonl"
    # 200-step demos = 10 chunks each; budget 15 chunks -> k_max 1, k2 unsupported
    server = ICILModelServer(
        lambda: FakePolicy(budget_chunks=15), registry, FakeStore([200] * 50), log_path=log_path
    )
    fields = {"icil_task_id": STOVE, "icil_condition": "k2", "episode_idx": 0}
    sent, policy = run_episode(server, fields, n_steps=3)
    assert policy is None  # no policy was created for an unsupported condition
    assert np.array_equal(sent[0]["actions"], np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    row = json.loads(log_path.read_text().splitlines()[0])
    assert row["status"] == "unsupported" and row["k_max"] == 1


def test_kmax_condition_resolves_against_policy(registry):
    server = ICILModelServer(lambda: FakePolicy(budget_chunks=50), registry, FakeStore([60] * 50))
    fields = {"icil_task_id": STOVE, "icil_condition": "kmax.wrong_task", "episode_idx": 1}
    sent, policy = run_episode(server, fields, n_steps=1)
    assert len(policy.demos) == 8  # 3 chunks each, 8 x 3 = 24 <= 50
    assert all(d.task.lower() != "turn on the stove" for d in policy.demos)


def test_registry_hash_mismatch_is_an_error(registry, tmp_path):
    log_path = tmp_path / "server.jsonl"
    server = ICILModelServer(lambda: FakePolicy(), registry, FakeStore(), log_path=log_path)
    fields = {
        "icil_task_id": STOVE,
        "icil_condition": "k1",
        "episode_idx": 0,
        "icil_registry_hash": "deadbeef",
    }
    run_episode(server, fields, n_steps=1)
    row = json.loads(log_path.read_text().splitlines()[0])
    assert row["status"] == "error" and "registry hash mismatch" in row["error"]
