"""Statistics: Wilson score intervals, Cohen's kappa, and a paired bootstrap.

Stdlib only -- no numpy/scipy dependency, matching the repo's house style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Rate:
    """A proportion with a Wilson 95% confidence interval."""

    successes: int
    n: int

    @property
    def point(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def ci(self) -> Tuple[float, float]:
        return wilson_interval(self.successes, self.n)

    def pct(self) -> str:
        lo, hi = self.ci
        return f"{100 * self.point:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at the tails)."""

    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary labelers over the same items.

    kappa = 1 perfect agreement, 0 chance-level, <0 worse than chance.
    Returns 1.0 in the degenerate case where both labelers are constant and agree.
    """

    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(1 for x in a if x) / n
    pb1 = sum(1 for x in b if x) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def raw_agreement(a: Sequence[bool], b: Sequence[bool]) -> float:
    if not a:
        return float("nan")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def paired_bootstrap_diff(
    before: Sequence[bool],
    after: Sequence[bool],
    *,
    iters: int = 5000,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """Mean(after) - mean(before) for paired binary outcomes, with a bootstrap 95% CI."""

    pairs: List[Tuple[int, int]] = [(int(x), int(y)) for x, y in zip(before, after)]
    if not pairs:
        return (0.0, (0.0, 0.0))
    point = sum(y for _, y in pairs) / len(pairs) - sum(x for x, _ in pairs) / len(pairs)
    rng = Random(seed)
    diffs: List[float] = []
    n = len(pairs)
    for _ in range(iters):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(y for _, y in sample) / n - sum(x for x, _ in sample) / n)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    return (point, (lo, hi))


# --------------------------------------------------------------------------- #
# Scale: clustered CIs, multiplicity correction, two-proportion test          #
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[bool]],
    *,
    iters: int = 4000,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """Rate + 95% CI for binary outcomes grouped by item (cluster).

    With K samples per item, samples within an item are correlated; resampling whole
    items (not individual samples) gives an honest interval that a naive Wilson on the
    flattened samples would understate.
    """

    items = [list(c) for c in clusters if len(c)]
    flat = [int(x) for c in items for x in c]
    if not flat:
        return (0.0, (0.0, 0.0))
    point = sum(flat) / len(flat)
    rng = Random(seed)
    n = len(items)
    means: List[float] = []
    for _ in range(iters):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        vals = [int(x) for c in sample for x in c]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    means.sort()
    return (point, (means[int(0.025 * iters)], means[int(0.975 * iters)]))


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value for H0: p1 == p2 (pooled-variance z-test)."""

    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return min(1.0, 2 * _normal_sf(z))


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05):
    """Benjamini-Hochberg FDR control. Returns (rejected, qvalues) in input order.

    Controls the expected proportion of false discoveries among the comparisons you
    call significant -- essential once you run many model x probe comparisons.
    """

    m = len(pvalues)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    rejected = [False] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvalues[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = min(1.0, prev)
    thresh = -1
    for rank in range(m):
        i = order[rank]
        if pvalues[i] <= (rank + 1) / m * alpha:
            thresh = rank
    for rank in range(thresh + 1):
        rejected[order[rank]] = True
    return rejected, q
