"""
Metrics collection and aggregation.
"""

import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class Metric:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"


class MetricsCollector:
    """
    Thread-safe metrics collector with counters, gauges, histograms, and timers.
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._metrics_history: List[Metric] = []

    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] += value
            self._record(name, self._counters[key], tags, "counter")

    def decrement(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        self.increment(name, -value, tags)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value
            self._record(name, value, tags, "gauge")

    def observe(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        with self._lock:
            key = self._make_key(name, tags)
            self._histograms[key].append(value)
            self._record(name, value, tags, "histogram")

    def record_time(self, name: str, duration: float, tags: Dict[str, str] = None) -> None:
        with self._lock:
            key = self._make_key(name, tags)
            self._timers[key].append(duration)
            self._record(name, duration * 1000, tags, "timer")

    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}#{tag_str}"
        return name

    def _record(self, name: str, value: float, tags: Dict[str, str], metric_type: str) -> None:
        self._metrics_history.append(Metric(
            name=name, value=value, tags=tags or {}, metric_type=metric_type
        ))
        if len(self._metrics_history) > 10000:
            self._metrics_history = self._metrics_history[-5000:]

    def get_counter(self, name: str, tags: Dict[str, str] = None) -> float:
        return self._counters.get(self._make_key(name, tags), 0.0)

    def get_gauge(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        return self._gauges.get(self._make_key(name, tags))

    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        values = self._histograms.get(self._make_key(name, tags), [])
        if not values:
            return {}
        sorted_vals = sorted(values)
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "p50": sorted_vals[len(sorted_vals) // 2],
            "p90": sorted_vals[int(len(sorted_vals) * 0.9)],
            "p99": sorted_vals[int(len(sorted_vals) * 0.99)],
        }

    def get_timer_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        return self.get_histogram_stats(name, tags)

    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: self.get_histogram_stats(k.split("#")[0]) for k in self._histograms},
            "timers": {k: self.get_timer_stats(k.split("#")[0]) for k in self._timers},
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._metrics_history.clear()
