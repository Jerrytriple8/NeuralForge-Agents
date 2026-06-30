"""
Base agent class for NeuralForge.
All agents inherit from this and implement execute().
"""

import asyncio
import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    name: str
    description: str = ""
    max_retries: int = 3
    timeout: float = 60.0
    verbose: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent_name: str
    task_id: str
    output: Any
    confidence: float = 1.0
    reasoning: str = ""
    elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all NeuralForge agents.
    
    Provides common functionality:
    - Lifecycle hooks (on_start, on_complete, on_error)
    - Retry logic with exponential backoff
    - Execution history tracking
    - Inter-agent communication via shared context
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._history: List[AgentResult] = []
        self._running = False
        self._context: Dict[str, Any] = {}
        self._hooks = {"on_start": [], "on_complete": [], "on_error": []}

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def history(self) -> List[AgentResult]:
        return self._history.copy()

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def add_hook(self, event: str, callback) -> None:
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _fire_hooks(self, event: str, **kwargs) -> None:
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(**kwargs)
                else:
                    hook(**kwargs)
            except Exception as e:
                logger.error("Hook error (%s.%s): %s", self.name, event, e)

    @abstractmethod
    async def execute(self, payload: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main task. Must be implemented by subclasses."""
        raise NotImplementedError

    async def run(self, payload: Dict[str, Any]) -> AgentResult:
        """Run with lifecycle hooks and retry logic."""
        task_id = str(uuid.uuid4())[:8]
        await self._fire_hooks("on_start", agent=self, task_id=task_id)

        retries = 0
        last_error = None

        while retries <= self.config.max_retries:
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    self.execute(payload),
                    timeout=self.config.timeout,
                )
                result.elapsed = time.time() - start_time
                self._history.append(result)
                await self._fire_hooks("on_complete", agent=self, result=result)
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.config.timeout}s"
                retries += 1
            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(2 ** retries * 0.1)

        error_result = AgentResult(
            agent_name=self.name,
            task_id=task_id,
            output=None,
            confidence=0.0,
            reasoning=f"Failed after {retries} retries: {last_error}",
            elapsed=time.time() - start_time,
        )
        self._history.append(error_result)
        await self._fire_hooks("on_error", agent=self, error=last_error)
        return error_result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
