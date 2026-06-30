"""LRU and TTL cache implementations."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """Thread-safe Least Recently Used cache.

    Args:
        capacity: Maximum number of entries.
    """

    def __init__(self, capacity: int = 128) -> None:
        self._cap = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return default

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._cap:
                self._data.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "capacity": self._cap,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"LRUCache(size={self.size}/{self._cap}, hit_rate={self.hit_rate:.1%})"


class TTLCache:
    """Cache with per-entry time-to-live expiration.

    Args:
        default_ttl: Default TTL in seconds for entries.
        max_size: Maximum number of entries (oldest evicted first).
    """

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1024) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._data: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._data:
                value, expires_at = self._data[key]
                if time.monotonic() < expires_at:
                    return value
                del self._data[key]
            return default

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._data[key] = (value, expires_at)
            self._evict_expired()

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_expired(self) -> None:
        """Remove expired entries and enforce max size. Caller must hold lock."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._data.items() if now >= exp]
        for k in expired:
            del self._data[k]
        # Enforce max size by removing oldest (earliest expiry)
        while len(self._data) > self._max_size:
            oldest_key = min(self._data, key=lambda k: self._data[k][1])
            del self._data[oldest_key]

    @property
    def size(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"TTLCache(size={self.size}, default_ttl={self._default_ttl}s)"
