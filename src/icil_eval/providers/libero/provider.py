"""LIBERO task provider: the five original suites (130 tasks) in the unified schema.

Source of truth is an upstream LIBERO checkout (``Lifelong-Robot-Learning/LIBERO`` at commit
``8f1084e``, the commit vla-eval's Docker image pins): ``libero/libero/bddl_files/<suite>/*.bddl``
and ``libero/libero/init_files/<suite>/*.pruned_init``. Demonstrations come from the Hugging Face
dataset ``yifengzhu-hf/LIBERO-datasets`` (Apache-2.0); their sha256 is read from the Hub API so the
registry can be built without downloading 100 GB.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from icil_eval.paths import cache_root
from icil_eval.providers.base import TaskProvider
from icil_eval.providers.libero.bddl import BddlProblem, load_bddl
from icil_eval.registry.hashing import hash_obj, sha256_file
from icil_eval.registry.schema import (
    Bounds,
    Capabilities,
    DemoPool,
    Embodiment,
    Goal,
    Horizon,
    InitStates,
    ObjectRef,
    ProviderInfo,
    Scene,
    Task,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

# Episode caps as pinned by vla-eval's LIBERO adapter (MAX_STEP_MAPPING) and the OpenVLA protocol.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
N_STEPS_WAIT = 10
N_INIT_STATES = 50
N_DEMOS = 50

DEMO_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SOURCE_VERSION = {
    "libero_commit": LIBERO_COMMIT,
    "robosuite": "1.4.0",
    "mujoco": "3.2.3",
    "bddl": "1.0.1",
}

EMBODIMENT = Embodiment(
    robot="panda",
    gripper="panda_default",
    dof=7,
    action_space="osc_pose_delta_7d",
    control_hz=20.0,
)

# Goal predicate -> skill tag (predicate names are case-insensitive in LIBERO).
SKILL_OF_PREDICATE = {
    "on": "place_on",
    "in": "place_in",
    "stack": "stack",
    "up": "lift",
    "open": "open",
    "close": "close",
    "turnon": "turn_on",
    "turnoff": "turn_off",
}
MOVE_PREDICATES = ("on", "in", "stack")


def _norm_pred(pred: List[str]) -> List[str]:
    return [pred[0].lower()] + list(pred[1:])


def stem_language(stem: str) -> str:
    """LIBERO's canonical instruction, derived from the BDDL file name.

    Mirrors ``libero.libero.benchmark.grab_language_from_filename``: LIBERO-100 stems drop the
    ``<SCENE>_`` prefix (``KITCHEN_SCENE10_`` has one more character), underscores become spaces.
    This is the string every downstream tool (LIBERO, vla-eval, LeRobot, BPP checkpoints) uses; the
    BDDL ``:language`` field occasionally differs and is kept as ``extra.bddl_language``.
    """
    if stem[0].isupper():
        offset = 8 if "SCENE10" in stem else 7
        return " ".join(stem[stem.find("SCENE") + offset :].split("_"))
    return " ".join(stem.split("_"))


def _scene_name(stem: str) -> Optional[str]:
    """LIBERO-100 stems start with a scene prefix (``KITCHEN_SCENE3_``, ``STUDY_SCENE1_``, ...)."""
    parts = stem.split("_")
    for i in range(1, min(4, len(parts))):
        if parts[i - 1].startswith("SCENE") and parts[i - 1][5:].isdigit():
            return "_".join(parts[:i])
    return None


def _chance_success(problem: BddlProblem, goal: List[List[str]]) -> Optional[float]:
    """1/n when a single move-predicate goal targets an object with n same-category instances."""
    if len(goal) != 1:
        return None
    pred = goal[0]
    if pred[0] not in MOVE_PREDICATES or len(pred) < 2:
        return None
    moved = pred[1]
    category = problem.objects.get(moved)
    if category is None:
        return None
    n_same = sum(1 for c in problem.objects.values() if c == category)
    return 1.0 / n_same if n_same > 1 else None


def _read_pruned_init_shape(path: Path) -> Optional[List[int]]:
    """Read the array shape from a ``.pruned_init`` file without torch when possible."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith("data.pkl")]
            if not names:
                return None
            data = zf.read(names[0])
        arr = pickle.loads(data)  # numpy arrays pickled by torch.save need no torch
        shape = getattr(arr, "shape", None)
        return list(shape) if shape is not None else None
    except Exception:
        pass
    try:  # torch-saved tensors
        import torch

        arr = torch.load(path, weights_only=False, map_location="cpu")
        return list(arr.shape)
    except Exception:
        return None


class LiberoProvider(TaskProvider):
    name = "libero"

    def __init__(
        self,
        libero_root: Optional[str] = None,
        raw_root: Optional[str] = None,
        offline: bool = False,
        verify_local: bool = False,
    ) -> None:
        self.libero_root = Path(libero_root or self._default_libero_root())
        self.raw_root = Path(raw_root) if raw_root else cache_root() / "raw" / "libero"
        self.offline = offline
        self.verify_local = verify_local
        self._hf_tree: Dict[str, Dict[str, Any]] = {}
        self._hf_revision: Optional[str] = None

    # ------------------------------------------------------------------ locations
    @staticmethod
    def _default_libero_root() -> str:
        env = os.environ.get("LIBERO_ROOT")
        if env:
            return env
        try:
            from libero.libero import get_libero_path

            return get_libero_path("benchmark_root")
        except Exception as e:  # pragma: no cover - depends on machine
            raise RuntimeError(
                "LIBERO root not found: pass libero_root=..., set LIBERO_ROOT, or install libero"
            ) from e

    @property
    def bddl_root(self) -> Path:
        return self.libero_root / "bddl_files"

    @property
    def init_root(self) -> Path:
        return self.libero_root / "init_files"

    # ------------------------------------------------------------------ contract
    def version(self) -> Dict[str, str]:
        v = dict(SOURCE_VERSION)
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(self.libero_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if commit:
                v["libero_commit"] = commit
        except Exception:
            pass
        return v

    def backend_binding(self, suite: str) -> Dict[str, Any]:
        return {
            "backend": "vla_eval",
            "inner": "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
            "params": {"suite": suite, "seed": 7, "num_steps_wait": N_STEPS_WAIT},
        }

    def info(self, tasks: List[Task]) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            description=(
                "LIBERO lifelong robot learning benchmark, original suites (Panda, OSC_POSE)."
            ),
            license="MIT (code), Apache-2.0 (demonstrations on Hugging Face)",
            url="https://github.com/Lifelong-Robot-Learning/LIBERO",
            citation=(
                "Liu et al., LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning, "
                "NeurIPS 2023"
            ),
            version=self.version(),
            suites=sorted({t.suite for t in tasks}),
            n_tasks=len(tasks),
            backend={
                "backend": "vla_eval",
                "image": "ghcr.io/allenai/vla-evaluation-harness/libero:0.5.0",
            },
        )

    def enumerate_tasks(self) -> List[Task]:
        version = self.version()
        tasks: List[Task] = []
        for suite in SUITES:
            suite_dir = self.bddl_root / suite
            if not suite_dir.exists():
                continue
            tree = self._hf_tree_for(suite)
            for bddl_path in sorted(suite_dir.glob("*.bddl")):
                tasks.append(self._task_from_bddl(suite, bddl_path, tree, version))
        return sorted(tasks, key=lambda t: t.task_id)

    # ------------------------------------------------------------------ internals
    def _task_from_bddl(
        self, suite: str, bddl_path: Path, tree: Dict[str, Dict[str, Any]], version: Dict[str, str]
    ) -> Task:
        problem = load_bddl(bddl_path)
        stem = bddl_path.stem
        goal = [_norm_pred(p) for p in problem.goal]
        init = {tuple(_norm_pred(p)) for p in problem.init}
        initially_satisfied = [p for p in goal if tuple(p) in init]
        n_effective = len(goal) - len(initially_satisfied)

        targets = {p[1] for p in goal if len(p) > 1}
        objects = [
            ObjectRef(name=n, category=c, role="target" if n in targets else "distractor")
            for n, c in sorted(problem.objects.items())
        ] + [
            ObjectRef(name=n, category=c, role="fixture")
            for n, c in sorted(problem.fixtures.items())
        ]
        skills = sorted({SKILL_OF_PREDICATE.get(p[0], p[0]) for p in goal})

        init_file = self.init_root / suite / f"{stem}.pruned_init"
        init_states = InitStates(file=f"init_files/{suite}/{stem}.pruned_init", n=N_INIT_STATES)
        if init_file.exists():
            init_states.sha256 = sha256_file(init_file)
            shape = _read_pruned_init_shape(init_file)
            if shape:
                init_states.n = int(shape[0])
                init_states.verified = True

        demo_rel = f"{suite}/{stem}_demo.hdf5"
        entry = tree.get(demo_rel, {})
        demo_pool = DemoPool(
            repo_id=DEMO_REPO_ID,
            path=demo_rel,
            n=N_DEMOS,
            format="libero_hdf5",
            revision=self._hf_revision,
            sha256=entry.get("sha256"),
            size_bytes=entry.get("size"),
        )
        local = self.raw_root / demo_rel
        if self.verify_local and local.exists():
            digest = sha256_file(local)
            if demo_pool.sha256 and digest != demo_pool.sha256:
                raise ValueError(f"{local}: sha256 {digest} != Hub {demo_pool.sha256}")
            demo_pool.sha256 = digest

        return Task(
            task_id=f"{self.name}/{suite}/{stem}",
            provider=self.name,
            suite=suite,
            language=stem_language(stem),
            embodiment=EMBODIMENT,
            scene=Scene(
                layout_id=hash_obj(problem.layout_dict(), 12),
                init_id=hash_obj(sorted(problem.init), 12),
                scene_name=_scene_name(stem),
                entities=problem.entities,
            ),
            objects=objects,
            skills=skills,
            goal=Goal(
                predicates=goal,
                n_effective_predicates=n_effective,
                initially_satisfied=initially_satisfied,
            ),
            horizon=Horizon(max_steps=MAX_STEPS[suite], n_steps_wait=N_STEPS_WAIT),
            init_states=init_states,
            demo_pool=demo_pool,
            bounds=Bounds(chance_success=_chance_success(problem, goal)),
            capabilities=Capabilities(progress_metric=n_effective >= 2),
            source_version=version,
            extra={
                "bddl_sha256": sha256_file(bddl_path),
                "obj_of_interest": problem.obj_of_interest,
            },
        )

    def _hf_tree_for(self, suite: str) -> Dict[str, Dict[str, Any]]:
        """``{relative path: {sha256, size}}`` for a suite, from the Hub API (cached on disk)."""
        if suite in self._hf_tree:
            return self._hf_tree[suite]
        cache_dir = cache_root() / "hf_api"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{DEMO_REPO_ID.replace('/', '__')}__{suite}.json"
        payload: Optional[Dict[str, Any]] = None
        if cache_file.exists():
            payload = json.loads(cache_file.read_text())
        elif not self.offline:
            try:
                payload = self._fetch_hf(suite)
                cache_file.write_text(json.dumps(payload, indent=1, sort_keys=True))
            except Exception as e:  # network failure: leave hashes unknown
                print(f"warning: Hub API unavailable for {suite}: {e}")
        tree: Dict[str, Dict[str, Any]] = {}
        if payload:
            self._hf_revision = payload.get("revision") or self._hf_revision
            for item in payload.get("files", []):
                tree[item["path"]] = {"sha256": item.get("sha256"), "size": item.get("size")}
        self._hf_tree[suite] = tree
        return tree

    @staticmethod
    def _fetch_hf(suite: str) -> Dict[str, Any]:
        base = f"https://huggingface.co/api/datasets/{DEMO_REPO_ID}"
        with urllib.request.urlopen(base, timeout=30) as r:
            meta = json.loads(r.read().decode())
        with urllib.request.urlopen(f"{base}/tree/main/{suite}", timeout=30) as r:
            items = json.loads(r.read().decode())
        files = []
        for x in items:
            if x.get("type") != "file":
                continue
            lfs = x.get("lfs") or {}
            files.append({"path": x["path"], "sha256": lfs.get("oid"), "size": x.get("size")})
        return {"revision": meta.get("sha"), "files": files}

    def export_demos(self, task: Task, dest: Path, max_episodes: Optional[int] = None) -> Path:
        from icil_eval.providers.libero.export import export_task_demos

        return export_task_demos(self, task, dest, max_episodes=max_episodes)
