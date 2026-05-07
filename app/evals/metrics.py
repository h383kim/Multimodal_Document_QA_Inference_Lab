"""Aggregate metric helpers for the benchmark layer."""

from __future__ import annotations

from collections.abc import Iterable


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(values)
    if p == 0:
        return s[0]
    rank = max(1, int((p / 100.0) * len(s) + 0.999999))
    rank = min(rank, len(s))
    return s[rank - 1]


def mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


def fraction(predicate_results: Iterable[bool]) -> float:
    rs = list(predicate_results)
    return sum(1 for r in rs if r) / len(rs) if rs else 0.0
