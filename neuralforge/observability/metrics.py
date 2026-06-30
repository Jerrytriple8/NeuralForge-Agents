"""Prometheus-style metrics collector."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """Lightweight in-memory metrics collector.

    Supports:
        - Counters (increment-only)
        - Gauges (up/down)
        - Histograms (value observations)
        - Tag-based dimensionality
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value

    # ------------------------------------------------------------------
    # Gauges
    # ------------------------------------------------------------------

    def gauge_set(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def gauge_inc(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] += value

    def gauge_dec(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] -= value

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record an observation in a histogram."""
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_counter(self, name: str, tags: dict[str, str] | None = None) -> float:
        return self._counters.get(self._make_key(name, tags), 0.0)

    def get_gauge(self, name: str, tags: dict[str, str] | None = None) -> float:
        return self._gauges.get(self._make_key(name, tags), 0.0)

    def get_histogram(self, name: str, tags: dict[str, str] | None = None) -> dict[str, float]:
        values = self._histograms.get(self._make_key(name, tags), [])
        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "count": n,
            "sum": round(sum(sorted_v), 4),
            "min": round(sorted_v[0], 4),
            "max": round(sorted_v[-1], 4),
            "avg": round(sum(sorted_v) / n, 4),
            "p50": round(sorted_v[n // 2], 4),
            "p95": round(sorted_v[int(n * 0.95)], 4),
            "p99": round(sorted_v[int(n * 0.99)], 4),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Export all metrics as a single dict."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram(k.split("|")[0])
                    for k in self._histograms
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @staticmethod
    def _make_key(name: str, tags: dict[str, str] | None = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"
