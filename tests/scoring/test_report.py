"""Ingest a synthetic vla-eval aggregate + server log and build the profile/report."""

from __future__ import annotations

import json
import shutil

import pytest

from icil_eval.paths import REGISTRY_ROOT
from icil_eval.registry.build import build_provider
from icil_eval.registry.io import load_registry
from icil_eval.scoring.report import build_results, render_markdown, validate_results

STOVE = "libero/libero_goal/turn_on_the_stove"
BOWL = "libero/libero_goal/put_the_bowl_on_the_plate"


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


def _episode(eid, task_id, cond, idx, success, name):
    return {
        "sid": "s",
        "eid": eid,
        "episode_id": idx,
        "metrics": {"success": success, "progress": 1.0 if success else 0.0},
        "steps": 100,
        "elapsed_sec": 10.0,
        "name": f"{task_id.rsplit('/', 1)[1]}|{cond}",
        "suite": "libero_goal",
        "task_id": 0,
        "episode_idx": idx,
        "icil_task_id": task_id,
        "icil_condition": cond,
        "icil_track": "configuration",
        "icil_language": name,
    }


def _server_row(eid, task_id, cond, idx, k, status="ok", refs=None):
    return {
        "eval_id": "e",
        "session_id": "s",
        "episode_id": eid,
        "task_id": task_id,
        "status": status,
        "error": None if status == "ok" else "too long",
        "language_mode": "none",
        "k_max": 2,
        "context": None
        if status != "ok"
        else {
            "track": "configuration",
            "query_task_id": task_id,
            "condition": cond,
            "k": k,
            "refs": refs or [],
            "relation": "same_task" if "wrong" not in cond else "other_task_same_layout",
            "context_task_id": task_id,
            "transform": "identity",
            "language": "none",
            "modality": "sensorimotor",
            "seed": 1000,
            "episode_idx": idx,
            "registry_hash": "h",
            "context_hash": "c",
        },
        "context_info": {"adaptation_latency_s": 0.05},
        "min_context_init_l2": 0.3,
        "adaptation_latency_s": 0.05,
        "control_latency_s": {"median": 0.1, "mean": 0.1, "n": 5},
        "n_observations": 100,
        "policy": "fake",
        "result": {"metrics": {"success": True}},
        "wall_s": 12.0,
    }


@pytest.fixture
def run_dir(tmp_path, registry):
    results = tmp_path / "run" / "results"
    results.mkdir(parents=True)
    episodes, server = [], []
    pattern = {
        "k1": [1, 1, 1, 0],
        "k1.wrong_task": [0, 0, 1, 0],
        "k0": [0, 0, 0, 0],
        "k2": [1, 1, 1, 1],
        "kmax.wrong_task": [0, 0, 0, 0],
    }
    n = 0
    for task in (STOVE, BOWL):
        name = registry.tasks[task].language
        for cond, outcomes in pattern.items():
            for idx, ok in enumerate(outcomes):
                eid = f"eid-{n}"
                n += 1
                episodes.append(_episode(eid, task, cond, idx, bool(ok), name))
                k = {"k0": 0, "k1": 1, "k1.wrong_task": 1, "k2": 2, "kmax.wrong_task": 2}[cond]
                server.append(_server_row(eid, task, cond, idx, k, refs=[[task, 7]] * k))
        # an unsupported condition that ran with hold actions
        eid = f"eid-{n}"
        n += 1
        episodes.append(_episode(eid, task, "k4", 0, False, name))
        server.append(_server_row(eid, task, "k4", 0, 4, status="unsupported"))
    agg = {
        "benchmark": "ICILBenchmark_test",
        "harness_version": "0.5.0",
        "tasks": [{"task": "all", "episodes": episodes}],
        "config": {"params": {"provider": "libero", "track": "configuration"}},
        "server_info": {"harness_version": "0.5.0"},
        "seed": 7,
    }
    (results / "ICILBenchmark_test_aggregate.json").write_text(json.dumps(agg))
    log = tmp_path / "server.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in server) + "\n")
    log.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "policy_spec": {
                    "name": "fake",
                    "card": {
                        "training_tasks": ["turn on the stove"],
                        "training_episodes": {"turn on the stove": [7]},
                    },
                }
            }
        )
    )
    (tmp_path / "run" / "config.yaml").write_text(
        "icil:\n  track: configuration\n  preset: smoke\n"
    )
    return tmp_path / "run", log


def test_build_results_profile_and_schema(run_dir, registry):
    rd, log = run_dir
    results = build_results(rd, registry, server_logs=[log])
    assert validate_results(results) == []
    track = results["profile"]["tracks"]["configuration"]
    conds = track["conditions"]
    assert conds["k1"]["success"]["value"] == pytest.approx(0.75)
    assert conds["k1.wrong_task"]["success"]["value"] == pytest.approx(0.25)
    assert conds["k4"]["status_counts"] == {"unsupported": 2} and conds["k4"]["n_ok"] == 0
    d = track["derived"]
    assert d["delta_context@1"]["value"] == pytest.approx(0.5) and d["delta_context@1"]["n"] == 8
    assert d["sr_at_k"] == {"0": 0.0, "1": 0.75, "2": 1.0, "4": None, "8": None}
    assert d["context_auc"] is None and d["k_supported"] == [0, 1, 2]
    assert d["k_max_observed"] == 2 and d["delta_context@kmax"]["value"] == pytest.approx(1.0)
    assert d["context_gain"]["2"]["value"] == pytest.approx(1.0)
    # exposure: the stove task and its demo 7 are declared trained; the bowl task is not
    eps = {(e["task_id"], e["condition"]): e for e in results["episodes"]}
    assert (
        eps[(STOVE, "k1")]["query_exposure"] == "seen"
        and eps[(STOVE, "k1")]["context_exposure"] == "seen"
    )
    assert (
        eps[(BOWL, "k1")]["query_exposure"] == "unseen"
        and eps[(BOWL, "k1")]["context_exposure"] == "unseen"
    )
    assert eps[(STOVE, "k0")]["context_exposure"] == "none"
    assert results["profile"]["coverage"]["status_counts"] == {"ok": 40, "unsupported": 2}
    assert "libero_goal" in track["slices"]
    md = render_markdown(results)
    assert "Δ_context@1" in md and "`k1.wrong_task`" in md
