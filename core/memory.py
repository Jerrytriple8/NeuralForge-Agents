"""
Memory systems for AI agents.
Implements episodic memory, working memory, and long-term storage.
"""

import time
import json
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import OrderedDict
import threading


@dataclass
class MemoryEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.accessed_at = time.time()
        self.access_count += 1

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def recency(self) -> float:
        return time.time() - self.accessed_at


class MemoryStore:
    """
    Thread-safe key-value memory store with eviction policies.
    
    Supports LRU, LFU, and importance-based eviction.
    """

    def __init__(self, max_size: int = 10000, eviction_policy: str = "lru"):
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._max_size = max_size
        self._eviction_policy = eviction_policy
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def store(self, key: str, value: Any, importance: float = 1.0, 
              tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> None:
        with self._lock:
            if key in self._store:
                entry = self._store[key]
                entry.value = value
                entry.importance = importance
                if tags:
                    entry.tags = tags
                if metadata:
                    entry.metadata.update(metadata)
                entry.touch()
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._max_size:
                    self._evict()
                self._store[key] = MemoryEntry(
                    key=key, value=value, importance=importance,
                    tags=tags or [], metadata=metadata or {},
                )

    def recall(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                entry = self._store[key]
                entry.touch()
                self._store.move_to_end(key)
                self._stats["hits"] += 1
                return entry.value
            self._stats["misses"] += 1
            return None

    def forget(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def search_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        with self._lock:
            results = []
            for entry in self._store.values():
                if any(t in entry.tags for t in tags):
                    results.append(entry)
            return results

    def get_important(self, top_k: int = 10) -> List[MemoryEntry]:
        with self._lock:
            sorted_entries = sorted(
                self._store.values(), key=lambda e: e.importance, reverse=True
            )
            return sorted_entries[:top_k]

    def get_recent(self, top_k: int = 10) -> List[MemoryEntry]:
        with self._lock:
            sorted_entries = sorted(
                self._store.values(), key=lambda e: e.accessed_at, reverse=True
            )
            return sorted_entries[:top_k]

    def _evict(self) -> None:
        if not self._store:
            return
        if self._eviction_policy == "lru":
            self._store.popitem(last=False)
        elif self._eviction_policy == "lfu":
            min_key = min(self._store, key=lambda k: self._store[k].access_count)
            del self._store[min_key]
        elif self._eviction_policy == "importance":
            min_key = min(self._store, key=lambda k: self._store[k].importance)
            del self._store[min_key]
        self._stats["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> dict:
        return {**self._stats, "size": self.size, "max_size": self._max_size}

    def export_json(self) -> str:
        data = {}
        for key, entry in self._store.items():
            data[key] = {
                "value": str(entry.value),
                "importance": entry.importance,
                "tags": entry.tags,
                "access_count": entry.access_count,
            }
        return json.dumps(data, indent=2)


class EpisodicMemory:
    """
    Episodic memory for storing sequences of experiences.
    Organized as episodes with temporal context.
    """

    def __init__(self, max_episodes: int = 1000):
        self._episodes: List[Dict[str, Any]] = []
        self._max_episodes = max_episodes
        self._current_episode: List[Dict[str, Any]] = []

    def record(self, event: str, data: Dict[str, Any]) -> None:
        self._current_episode.append({
            "event": event,
            "data": data,
            "timestamp": time.time(),
        })

    def close_episode(self, summary: Optional[str] = None) -> int:
        if not self._current_episode:
            return -1
        episode = {
            "id": len(self._episodes),
            "events": self._current_episode.copy(),
            "summary": summary,
            "start_time": self._current_episode[0]["timestamp"],
            "end_time": time.time(),
            "event_count": len(self._current_episode),
        }
        self._episodes.append(episode)
        self._current_episode.clear()

        if len(self._episodes) > self._max_episodes:
            self._episodes.pop(0)

        return episode["id"]

    def recall_episode(self, episode_id: int) -> Optional[Dict[str, Any]]:
        if 0 <= episode_id < len(self._episodes):
            return self._episodes[episode_id]
        return None

    def search_episodes(self, keyword: str) -> List[Dict[str, Any]]:
        results = []
        for ep in self._episodes:
            for event in ep["events"]:
                if keyword.lower() in str(event).lower():
                    results.append(ep)
                    break
        return results

    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._episodes[-n:]

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def current_events(self) -> int:
        return len(self._current_episode)


class WorkingMemory:
    """
    Short-term working memory with capacity limits.
    Implements chunking and rehearsal mechanisms.
    """

    def __init__(self, capacity: int = 7):
        self._slots: List[Optional[Dict]] = [None] * capacity
        self._capacity = capacity
        self._focus: int = 0

    def attend(self, item: Dict[str, Any]) -> bool:
        for i in range(self._capacity):
            if self._slots[i] is None:
                self._slots[i] = {**item, "attended_at": time.time()}
                self._focus = i
                return True
        return False

    def retrieve_focus(self) -> Optional[Dict]:
        return self._slots[self._focus]

    def shift_focus(self, direction: int) -> None:
        self._focus = (self._focus + direction) % self._capacity

    def displace(self, index: int) -> Optional[Dict]:
        if 0 <= index < self._capacity:
            item = self._slots[index]
            self._slots[index] = None
            return item
        return None

    def clear(self) -> None:
        self._slots = [None] * self._capacity
        self._focus = 0

    @property
    def occupancy(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    @property
    def is_full(self) -> bool:
        return all(s is not None for s in self._slots)
