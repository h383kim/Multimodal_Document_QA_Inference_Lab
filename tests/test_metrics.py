"""Aggregate metric helpers."""
from __future__ import annotations

import pytest

from app.evals.metrics import fraction, mean, percentile


def test_percentile_basic():
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(vals, 50) == 50
    assert percentile(vals, 95) == 100
    assert percentile(vals, 0) == 10


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_percentile_invalid():
    with pytest.raises(ValueError):
        percentile([1.0], 150)


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5
    assert mean([]) == 0.0


def test_fraction():
    assert fraction([True, True, False, False]) == 0.5
    assert fraction([]) == 0.0
    assert fraction([True, True, True]) == 1.0
