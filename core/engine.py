"""
NeuralForge Engine - Core AI orchestration engine.
Manages agent lifecycle, task routing, and inference pipelines.
"""

import asyncio
import time
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    agent_name: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> Optional[float]:
        if self.completed_at:
            return self.completed_at - self.created_at
        return None


@dataclass
class EngineConfig:
    max_concurrent_tasks: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 300.0
    enable_caching: bool = True
    cache_ttl: float = 3600.0
    log_level: str = "INFO"
    worker_threads: int = 4


class TaskCache:
    """LRU cache for task results with TTL expiration."""

    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self._cache: Dict[str, tuple] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._access_order.remove(key)
                self._access_order.append(key)
                return value
            else:
                del self._cache[key]
                self._access_order.remove(key)
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = (value, time.time())
        self._access_order.append(key)

    def invalidate(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class NeuralEngine:
    """
    Main orchestration engine for AI agents.
    
    Manages agent registration, task routing, concurrent execution,
    result caching, and failure recovery.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._agents: Dict[str, Any] = {}
        self._tasks: Dict[str, Task] = {}
        self._cache = TaskCache(ttl=self.config.cache_ttl) if self.config.enable_caching else None
        self._executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._running = False
        self._hooks: Dict[str, List[Callable]] = {
            "pre_task": [],
            "post_task": [],
            "on_error": [],
            "on_complete": [],
        }
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        logger.info("NeuralEngine initialized with config: %s", self.config)

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent with the engine."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        self._agents[name] = agent
        logger.info("Registered agent: %s", name)

    def unregister_agent(self, name: str) -> None:
        """Remove an agent from the engine."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found")
        del self._agents[name]
        logger.info("Unregistered agent: %s", name)

    def get_agent(self, name: str) -> Any:
        """Get a registered agent by name."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found")
        return self._agents[name]

    @property
    def registered_agents(self) -> List[str]:
        return list(self._agents.keys())

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a lifecycle hook."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}")
        self._hooks[event].append(callback)

    async def _fire_hooks(self, event: str, **kwargs) -> None:
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(**kwargs)
                else:
                    hook(**kwargs)
            except Exception as e:
                logger.error("Hook error (%s): %s", event, e)

    async def submit_task(
        self,
        agent_name: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
        priority: int = 0,
        use_cache: bool = True,
    ) -> Task:
        """Submit a task for async execution."""
        if agent_name not in self._agents:
            raise KeyError(f"Agent '{agent_name}' not registered")

        task_id = task_id or str(uuid.uuid4())

        # Check cache
        cache_key = f"{agent_name}:{hash(str(sorted(payload.items())))}"
        if use_cache and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                task = Task(
                    task_id=task_id,
                    agent_name=agent_name,
                    payload=payload,
                    status=TaskStatus.COMPLETED,
                    result=cached,
                    completed_at=time.time(),
                )
                self._tasks[task_id] = task
                return task
            self._stats["cache_misses"] += 1

        task = Task(task_id=task_id, agent_name=agent_name, payload=payload)
        self._tasks[task_id] = task
        self._stats["tasks_submitted"] += 1

        # Execute async
        asyncio.create_task(self._execute_task(task, cache_key))
        return task

    async def _execute_task(self, task: Task, cache_key: str) -> None:
        """Execute a task with retry logic."""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            await self._fire_hooks("pre_task", task=task)

            agent = self._agents[task.agent_name]
            retries = 0

            while retries <= self.config.max_retries:
                try:
                    if asyncio.iscoroutinefunction(agent.execute):
                        result = await asyncio.wait_for(
                            agent.execute(task.payload),
                            timeout=self.config.timeout,
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                self._executor, agent.execute, task.payload
                            ),
                            timeout=self.config.timeout,
                        )

                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    self._stats["tasks_completed"] += 1

                    if self._cache:
                        self._cache.set(cache_key, result)

                    await self._fire_hooks("post_task", task=task)
                    await self._fire_hooks("on_complete", task=task)
                    logger.info(
                        "Task %s completed in %.2fs",
                        task.task_id,
                        task.elapsed or 0,
                    )
                    return

                except asyncio.TimeoutError:
                    retries += 1
                    if retries > self.config.max_retries:
                        task.error = f"Timeout after {self.config.timeout}s"
                        break
                    logger.warning(
                        "Task %s timeout, retry %d/%d",
                        task.task_id,
                        retries,
                        self.config.max_retries,
                    )
                    await asyncio.sleep(self.config.retry_delay * retries)

                except Exception as e:
                    retries += 1
                    if retries > self.config.max_retries:
                        task.error = str(e)
                        break
                    logger.warning(
                        "Task %s error: %s, retry %d/%d",
                        task.task_id,
                        e,
                        retries,
                        self.config.max_retries,
                    )
                    await asyncio.sleep(self.config.retry_delay * retries)

            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            self._stats["tasks_failed"] += 1
            await self._fire_hooks("on_error", task=task)
            logger.error("Task %s failed: %s", task.task_id, task.error)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        return [t for t in self._tasks.values() if t.status == status]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "active_agents": len(self._agents),
            "total_tasks": len(self._tasks),
            "cache_size": self._cache.size if self._cache else 0,
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        self._running = False
        pending = self.get_tasks_by_status(TaskStatus.PENDING)
        for task in pending:
            task.status = TaskStatus.CANCELLED
        self._executor.shutdown(wait=False)
        logger.info("Engine shutdown complete")

    def __repr__(self) -> str:
        return f"NeuralEngine(agents={len(self._agents)}, tasks={len(self._tasks)})"
