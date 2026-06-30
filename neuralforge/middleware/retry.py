"""Retry policy with configurable backoff strategies."""

from __future__ import annotations

import asyncio
import enum
import functools
import random
from typing import Any, Callable, Optional, Type


class BackoffStrategy(enum.Enum):
    """Backoff calculation strategy."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class RetryPolicy:
    """Configurable retry policy with multiple backoff strategies.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap (default 60s).
        strategy: Backoff strategy to use.
        retryable_exceptions: Tuple of exception types to retry on.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
        retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.retryable_exceptions = retryable_exceptions

    def compute_delay(self, attempt: int) -> float:
        """Compute the delay for a given attempt number."""
        if self.strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            delay = self.base_delay * (2 ** attempt)
            delay = delay * (0.5 + random.random() * 0.5)
        else:
            delay = self.base_delay
        return min(delay, self.max_delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if a retry should be attempted."""
        if attempt >= self.max_retries:
            return False
        return isinstance(exception, self.retryable_exceptions)


def with_retry(
    policy: RetryPolicy | None = None,
    max_retries: int = 3,
    on_retry: Callable[[Exception, int], None] | None = None,
):
    """Decorator that applies retry logic to an async function.

    Usage:
        @with_retry(max_retries=3)
        async def flaky_api_call():
            ...
    """
    if policy is None:
        policy = RetryPolicy(max_retries=max_retries)

    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(policy.max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except policy.retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < policy.max_retries:
                        delay = policy.compute_delay(attempt)
                        if on_retry:
                            on_retry(exc, attempt)
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
