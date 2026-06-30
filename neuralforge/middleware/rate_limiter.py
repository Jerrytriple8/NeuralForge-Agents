"""Token bucket rate limiter — async-compatible."""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Classic token bucket algorithm with async support.

    Args:
        rate: Tokens added per second.
        burst: Maximum bucket size (default = rate).
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        self._rate = rate
        self._burst = burst or rate
        self._tokens = self._burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Caller must hold lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, waiting if necessary. Returns wait time."""
        waited = 0.0
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            # Calculate wait time
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate
        # Wait outside lock
        await asyncio.sleep(wait_time)
        waited = wait_time
        async with self._lock:
            self._refill()
            self._tokens -= tokens
        return waited

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking attempt to acquire tokens."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens

    def stats(self) -> dict[str, float]:
        return {
            "rate": self._rate,
            "burst": self._burst,
            "available": round(self.available_tokens, 2),
        }
