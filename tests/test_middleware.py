"""Tests for middleware components."""

import asyncio
import time

import pytest

from neuralforge.middleware.cache import LRUCache, TTLCache
from neuralforge.middleware.rate_limiter import TokenBucketRateLimiter
from neuralforge.middleware.retry import RetryPolicy, BackoffStrategy, with_retry
from neuralforge.middleware.circuit_breaker import CircuitBreaker, CircuitState


class TestLRUCache:
    def test_basic_set_get(self):
        cache = LRUCache(capacity=3)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_eviction(self):
        cache = LRUCache(capacity=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # "a" evicted
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_ordering(self):
        cache = LRUCache(capacity=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Touch "a" to make it recently used
        cache.set("c", 3)  # "b" evicted (not "a")
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_hit_rate(self):
        cache = LRUCache(capacity=10)
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate == 0.5

    def test_delete(self):
        cache = LRUCache(capacity=10)
        cache.set("a", 1)
        assert cache.delete("a")
        assert cache.get("a") is None
        assert not cache.delete("nonexistent")

    def test_stats(self):
        cache = LRUCache(capacity=5)
        cache.set("x", 42)
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["capacity"] == 5


class TestTTLCache:
    def test_basic_set_get(self):
        cache = TTLCache(default_ttl=10.0)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_expiration(self):
        cache = TTLCache(default_ttl=0.05)
        cache.set("a", 1)
        time.sleep(0.1)
        assert cache.get("a") is None

    def test_custom_ttl(self):
        cache = TTLCache(default_ttl=0.05)
        cache.set("short", 1, ttl=0.02)
        cache.set("long", 2, ttl=10.0)
        time.sleep(0.05)
        assert cache.get("short") is None
        assert cache.get("long") == 2


class TestTokenBucketRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_burst(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5.0)
        wait = await limiter.acquire(3.0)
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_empty(self):
        limiter = TokenBucketRateLimiter(rate=100.0, burst=1.0)
        await limiter.acquire(1.0)  # Drain
        start = asyncio.get_event_loop().time()
        await limiter.acquire(1.0)  # Must wait
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.005  # Should have waited

    def test_try_acquire(self):
        limiter = TokenBucketRateLimiter(rate=10.0, burst=2.0)
        assert limiter.try_acquire(2.0)
        assert not limiter.try_acquire(1.0)  # Empty


class TestRetryPolicy:
    def test_exponential_backoff(self):
        policy = RetryPolicy(max_retries=5, base_delay=1.0, strategy=BackoffStrategy.EXPONENTIAL)
        assert policy.compute_delay(0) == 1.0
        assert policy.compute_delay(1) == 2.0
        assert policy.compute_delay(2) == 4.0

    def test_max_delay_cap(self):
        policy = RetryPolicy(max_retries=10, base_delay=1.0, max_delay=5.0, strategy=BackoffStrategy.EXPONENTIAL)
        assert policy.compute_delay(10) == 5.0

    def test_should_retry(self):
        policy = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,))
        assert policy.should_retry(ValueError("bad"), 0)
        assert not policy.should_retry(ValueError("bad"), 3)
        assert not policy.should_retry(TypeError("bad"), 0)

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        attempt = 0

        @with_retry(max_retries=3)
        async def flaky():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ValueError("fail")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert attempt == 3


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow()

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow()

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05, success_threshold=2)
        cb.force_open()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_stats(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_success()
        stats = cb.stats()
        assert stats["total_failures"] == 1
        assert stats["total_successes"] == 1
