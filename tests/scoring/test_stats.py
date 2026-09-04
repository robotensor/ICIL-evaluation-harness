from __future__ import annotations

import math

from icil_eval.scoring.stats import cluster_bootstrap_mean, paired_difference, rate, wilson_interval


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(50, 50)
    assert math.isclose(hi, 1.0) and 0.9 < lo < 1.0
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    assert wilson_interval(0, 0) == (None, None)


def test_rate_ignores_missing():
    r = rate([True, False, None, True])
    assert r["n"] == 3 and math.isclose(r["value"], 2 / 3)


def test_paired_difference_and_mcnemar():
    a = [True, True, True, False, True, None]
    b = [False, False, True, False, True, True]
    d = paired_difference(a, b)
    assert d["n"] == 5 and math.isclose(d["value"], 0.4)
    assert (
        d["n_discordant"] == 2 and d["mcnemar_p"] == 0.5
    )  # 2 discordant, both favour a: p = 2 * (1/4)
    assert d["stderr"] is not None and d["ci95"][0] < d["value"] < d["ci95"][1]
    none = paired_difference([None], [True])
    assert none["value"] is None and none["n"] == 0


def test_cluster_bootstrap_is_macro_and_deterministic():
    groups = {"t1": [1.0, 1.0, 1.0, 1.0], "t2": [0.0], "t3": [0.5, 0.5]}
    r1 = cluster_bootstrap_mean(groups, n_boot=500, seed=1)
    r2 = cluster_bootstrap_mean(groups, n_boot=500, seed=1)
    assert math.isclose(r1["value"], 0.5) and r1 == r2
    assert r1["ci95"][0] <= r1["value"] <= r1["ci95"][1]
