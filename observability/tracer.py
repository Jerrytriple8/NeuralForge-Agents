"""
Distributed tracing implementation.
"""

import time
import uuid
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def end(self, status: str = "OK") -> None:
        self.end_time = time.time()
        self.status = status

    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration * 1000 if self.duration else None,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


class Tracer:
    """
    Distributed tracer for tracking execution across agents and pipelines.
    """

    def __init__(self, service_name: str = "neuralforge"):
        self._service_name = service_name
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()

    @contextmanager
    def start_span(self, name: str, parent_id: str = None, attributes: Dict[str, Any] = None):
        """Context manager for creating spans."""
        trace_id = str(uuid.uuid4())[:16]
        span_id = str(uuid.uuid4())[:8]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            parent_id=parent_id,
            attributes=attributes or {},
        )

        with self._lock:
            self._active_spans[span_id] = span

        try:
            yield span
            span.end("OK")
        except Exception as e:
            span.set_attribute("error", str(e))
            span.end("ERROR")
            raise
        finally:
            with self._lock:
                self._spans.append(span)
                self._active_spans.pop(span_id, None)

    def get_trace(self, trace_id: str) -> List[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def get_all_spans(self) -> List[Span]:
        return self._spans.copy()

    def get_stats(self) -> Dict[str, Any]:
        if not self._spans:
            return {"total_spans": 0}
        durations = [s.duration for s in self._spans if s.duration is not None]
        return {
            "total_spans": len(self._spans),
            "active_spans": len(self._active_spans),
            "avg_duration_ms": sum(durations) / len(durations) * 1000 if durations else 0,
            "max_duration_ms": max(durations) * 1000 if durations else 0,
            "error_count": sum(1 for s in self._spans if s.status == "ERROR"),
        }

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._active_spans.clear()
