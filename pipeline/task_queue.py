"""
Async task queue with priority support.
"""

import asyncio
import time
import uuid
import heapq
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class QueueItem:
    priority: int
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    created_at: float = field(default_factory=time.time)
    max_retries: int = 0
    retry_count: int = 0

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class TaskQueue:
    """
    Async priority task queue with worker pool, retry logic, and metrics.
    """

    def __init__(self, max_workers: int = 5, max_size: int = 1000):
        self._queue: List[QueueItem] = []
        self._max_workers = max_workers
        self._max_size = max_size
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._results: Dict[str, Any] = {}
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0,
        }
        self._event = asyncio.Event()

    async def start(self) -> None:
        """Start worker tasks."""
        self._running = True
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        logger.info("TaskQueue started with %d workers", self._max_workers)

    async def stop(self) -> None:
        """Stop all workers."""
        self._running = False
        self._event.set()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(
        self,
        func: Callable,
        *args,
        priority: Priority = Priority.NORMAL,
        task_id: Optional[str] = None,
        max_retries: int = 0,
        **kwargs,
    ) -> str:
        """Submit a task to the queue."""
        if len(self._queue) >= self._max_size:
            raise RuntimeError("Queue is full")

        task_id = task_id or str(uuid.uuid4())[:8]
        item = QueueItem(
            priority=int(priority),
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries,
        )
        heapq.heappush(self._queue, item)
        self._stats["submitted"] += 1
        self._event.set()
        return task_id

    async def _worker(self, name: str) -> None:
        """Worker coroutine that processes tasks."""
        while self._running:
            if not self._queue:
                self._event.clear()
                await self._event.wait()
                continue

            item = heapq.heappop(self._queue)
            try:
                if asyncio.iscoroutinefunction(item.func):
                    result = await item.func(*item.args, **item.kwargs)
                else:
                    result = item.func(*item.args, **item.kwargs)

                self._results[item.task_id] = result
                self._stats["completed"] += 1

            except Exception as e:
                if item.retry_count < item.max_retries:
                    item.retry_count += 1
                    heapq.heappush(self._queue, item)
                    self._stats["retried"] += 1
                    logger.warning("Task %s failed, retrying (%d/%d)", item.task_id, item.retry_count, item.max_retries)
                else:
                    self._results[item.task_id] = {"error": str(e)}
                    self._stats["failed"] += 1
                    logger.error("Task %s failed: %s", item.task_id, e)

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Wait for and return a task result."""
        start = time.time()
        while time.time() - start < timeout:
            if task_id in self._results:
                return self._results.pop(task_id)
            await asyncio.sleep(0.1)
        raise asyncio.TimeoutError(f"Task {task_id} timed out")

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict:
        return {**self._stats, "queue_size": self.size, "workers": len(self._workers)}
