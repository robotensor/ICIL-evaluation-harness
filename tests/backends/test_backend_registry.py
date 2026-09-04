from __future__ import annotations

import pytest

from icil_eval.backends import get_backend


def test_vla_eval_backend_registered():
    b = get_backend("vla_eval")
    assert b.name == "vla_eval"
    assert b.benchmark_import_string("libero").endswith(":ICILBenchmark")
    cfg = b.build_run_config(
        provider="libero",
        track="configuration",
        suites=["libero_goal"],
        preset="smoke",
        output_dir="/tmp/x",
    )
    assert cfg["benchmarks"][0]["benchmark"] == b.benchmark_import_string("libero")
    with pytest.raises(KeyError):
        get_backend("nope")
