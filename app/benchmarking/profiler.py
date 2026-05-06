"""Lightweight profiler context manager used to wrap benchmark calls."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class ProfileResult:
    wall_ms: float
    rss_delta_mb: float
    peak_rss_mb: float


@contextmanager
def profile():
    """Yields a ProfileResult that fills in once the block exits."""
    proc = psutil.Process()
    start_rss = proc.memory_info().rss / (1024 * 1024)
    start = time.perf_counter_ns()
    result = ProfileResult(wall_ms=0.0, rss_delta_mb=0.0, peak_rss_mb=start_rss)
    try:
        yield result
    finally:
        end = time.perf_counter_ns()
        end_rss = proc.memory_info().rss / (1024 * 1024)
        result.wall_ms = (end - start) / 1_000_000.0
        result.rss_delta_mb = end_rss - start_rss
        result.peak_rss_mb = max(start_rss, end_rss)
