"""Distributed tracing with span hierarchy."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Span:
    """A single trace span representing a unit of work."""

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    parent_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:32])
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.ended_at == 0.0:
            return (time.monotonic() - self.started_at) * 1000
        return (self.ended_at - self.started_at) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, data: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "ts": time.monotonic() - self.started_at,
            "data": data or {},
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """In-process distributed tracer.

    Collects spans and supports parent-child relationships for
    reconstructing execution trees.
    """

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._active_span_id: str | None = None
        self._current_trace_id: str | None = None

    def start_span(
        self, name: str, parent_id: str | None = None, trace_id: str | None = None
    ) -> Span:
        """Start a new span, optionally as a child of the current active span."""
        span = Span(
            name=name,
            parent_id=parent_id or self._active_span_id,
            trace_id=trace_id or self._current_trace_id or uuid.uuid4().hex[:32],
        )
        self._spans.append(span)
        self._active_span_id = span.span_id
        self._current_trace_id = span.trace_id
        return span

    def end_span(self, span: Span) -> None:
        """Mark a span as finished."""
        span.ended_at = time.monotonic()
        # Restore parent as active
        if span.parent_id:
            self._active_span_id = span.parent_id
        else:
            self._active_span_id = None

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a given trace."""
        return [s for s in self._spans if s.trace_id == trace_id]

    @property
    def all_spans(self) -> list[Span]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()
        self._active_span_id = None
        self._current_trace_id = None

    def stats(self) -> dict[str, Any]:
        durations = [s.duration_ms for s in self._spans if s.ended_at > 0]
        return {
            "total_spans": len(self._spans),
            "completed_spans": len(durations),
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
            "traces": len({s.trace_id for s in self._spans}),
        }
