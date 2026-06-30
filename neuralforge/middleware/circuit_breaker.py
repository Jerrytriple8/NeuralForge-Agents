"""Circuit breaker pattern implementation."""

from __future__ import annotations

import enum
import time


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    Args:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before trying half-open.
        success_threshold: Successes needed in half-open to close circuit.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._total_failures = 0
        self._total_successes = 0

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def allow(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # Allow probe requests
        return False  # OPEN — reject

    def record_success(self) -> None:
        """Record a successful call."""
        self._total_successes += 1
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                self._close()
        else:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._total_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._open()
            return

        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._open()

    def force_open(self) -> None:
        """Manually open the circuit."""
        self._open()

    def force_close(self) -> None:
        """Manually close the circuit."""
        self._close()

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._failure_count = 0

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int | str]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "failure_threshold": self._failure_threshold,
        }

    def __repr__(self) -> str:
        return f"CircuitBreaker(state={self.state.value}, failures={self._failure_count}/{self._failure_threshold})"
