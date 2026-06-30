"""Pipeline lifecycle state machine."""

from __future__ import annotations

import enum
from typing import Optional


class PipelineState(enum.Enum):
    """Lifecycle states of a pipeline execution."""

    CREATED = "created"
    VALIDATING = "validating"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# Valid transitions: (from_state) -> {allowed_to_states}
_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.CREATED: {PipelineState.VALIDATING, PipelineState.CANCELLED},
    PipelineState.VALIDATING: {PipelineState.RUNNING, PipelineState.FAILED, PipelineState.CANCELLED},
    PipelineState.RUNNING: {
        PipelineState.SUCCESS,
        PipelineState.FAILED,
        PipelineState.PAUSED,
        PipelineState.CANCELLED,
        PipelineState.TIMEOUT,
    },
    PipelineState.PAUSED: {PipelineState.RUNNING, PipelineState.CANCELLED},
    PipelineState.SUCCESS: set(),
    PipelineState.FAILED: set(),
    PipelineState.CANCELLED: set(),
    PipelineState.TIMEOUT: set(),
}


class InvalidTransition(Exception):
    """Attempted an illegal state transition."""


class StateMachine:
    """Tracks and enforces pipeline state transitions.

    Usage:
        sm = StateMachine()
        sm.transition(PipelineState.VALIDATING)
        sm.transition(PipelineState.RUNNING)
        assert sm.state == PipelineState.RUNNING
    """

    def __init__(self, initial: PipelineState = PipelineState.CREATED) -> None:
        self._state = initial
        self._history: list[tuple[PipelineState, PipelineState]] = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return len(_TRANSITIONS.get(self._state, set())) == 0

    @property
    def history(self) -> list[tuple[PipelineState, PipelineState]]:
        return list(self._history)

    def can_transition(self, target: PipelineState) -> bool:
        return target in _TRANSITIONS.get(self._state, set())

    def transition(self, target: PipelineState) -> None:
        allowed = _TRANSITIONS.get(self._state, set())
        if target not in allowed:
            raise InvalidTransition(
                f"Cannot transition from {self._state.value} to {target.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self._history.append((self._state, target))
        self._state = target

    def __repr__(self) -> str:
        return f"StateMachine(state={self._state.value}, transitions={len(self._history)})"
