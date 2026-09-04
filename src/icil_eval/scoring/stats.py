"""Statistics for paired ICIL comparisons (standard S5)."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def wilson_interval(
    successes: int, n: int, z: float = 1.959964
) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(values: Sequence[Optional[bool]]) -> Dict[str, Optional[float]]:
    xs = [bool(v) for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return {"value": None, "n": 0, "ci95": [None, None], "stderr": None}
    k = sum(xs)
    lo, hi = wilson_interval(k, n)
    p = k / n
    return {
        "value": p,
        "n": n,
        "ci95": [lo, hi],
        "stderr": math.sqrt(p * (1 - p) / n) if n > 1 else None,
    }


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for p = 0.5 (McNemar on discordant pairs)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_difference(
    a: Sequence[Optional[bool]], b: Sequence[Optional[bool]]
) -> Dict[str, Optional[float]]:
    """Mean of (a - b) over pairs where both outcomes exist, with paired stderr and McNemar p."""
    pairs = [(bool(x), bool(y)) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {
            "value": None,
            "n": 0,
            "stderr": None,
            "ci95": [None, None],
            "n_discordant": 0,
            "mcnemar_p": None,
        }
    d = [float(x) - float(y) for x, y in pairs]
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else None
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)
    ci = [mean - 1.959964 * se, mean + 1.959964 * se] if se is not None else [None, None]
    return {
        "value": mean,
        "n": n,
        "stderr": se,
        "ci95": ci,
        "n_discordant": a_only + b_only,
        "mcnemar_p": _binom_two_sided_p(a_only, a_only + b_only),
    }


def cluster_bootstrap_mean(
    groups: Dict[str, Sequence[float]], n_boot: int = 2000, seed: int = 0
) -> Dict[str, Optional[float]]:
    """Macro mean over clusters (tasks) with a cluster-bootstrap 95% interval."""
    keys = sorted(k for k, v in groups.items() if len(v) > 0)
    if not keys:
        return {"value": None, "n_clusters": 0, "ci95": [None, None]}
    means = {k: sum(groups[k]) / len(groups[k]) for k in keys}
    point = sum(means.values()) / len(keys)
    if len(keys) < 2:
        return {"value": point, "n_clusters": len(keys), "ci95": [None, None]}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [means[rng.choice(keys)] for _ in keys]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    return {"value": point, "n_clusters": len(keys), "ci95": [lo, hi]}


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def median_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def group_by(rows: List, key) -> Dict:
    out: Dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out
