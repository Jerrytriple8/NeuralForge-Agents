"""Pipeline execution context — shared state across all nodes."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineContext:
    """Thread-safe shared context passed to every agent during a pipeline run.

    Provides:
        - Key/value store scoped to this pipeline run
        - Per-node results storage
        - Structured event log for observability
        - Cancellation support via asyncio.Event
    """

    pipeline_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    pipeline_name: str = ""
    started_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal state
    _store: dict[str, Any] = field(default_factory=dict, repr=False)
    _node_results: dict[str, Any] = field(default_factory=dict, repr=False)
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _cancel: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False, init=False
    )
    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, init=False
    )

    # ------------------------------------------------------------------
    # Key/value store
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    # ------------------------------------------------------------------
    # Node results
    # ------------------------------------------------------------------

    def set_node_result(self, node_id: str, result: Any) -> None:
        self._node_results[node_id] = result

    def get_node_result(self, node_id: str, default: Any = None) -> Any:
        return self._node_results.get(node_id, default)

    def get_dependency_results(self, depends_on: list[str]) -> dict[str, Any]:
        """Collect results from upstream nodes."""
        return {
            dep: self._node_results[dep]
            for dep in depends_on
            if dep in self._node_results
        }

    @property
    def all_results(self) -> dict[str, Any]:
        return dict(self._node_results)

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    async def log_event(
        self, event_type: str, node_id: str = "", data: Any = None
    ) -> None:
        async with self._lock:
            self._events.append(
                {
                    "ts": time.monotonic() - self.started_at,
                    "type": event_type,
                    "node": node_id,
                    "data": data,
                }
            )

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "elapsed": time.monotonic() - self.started_at,
            "store": dict(self._store),
            "node_results": dict(self._node_results),
            "event_count": len(self._events),
        }

    def __repr__(self) -> str:
        return (
            f"PipelineContext(pipeline={self.pipeline_name!r}, "
            f"run={self.run_id}, nodes={len(self._node_results)})"
        )
