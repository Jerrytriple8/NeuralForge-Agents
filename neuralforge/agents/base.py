"""Base agent interface and lifecycle hooks."""

from __future__ import annotations

import abc
import enum
from typing import Any, Optional

from neuralforge.core.context import PipelineContext


class AgentCapability(enum.Enum):
    """Declarative capabilities that agents can advertise."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    WEB_SEARCH = "web_search"
    FILE_IO = "file_io"
    API_CALL = "api_call"
    DATA_TRANSFORM = "data_transform"
    CUSTOM = "custom"


class BaseAgent(abc.ABC):
    """Abstract base class for all NeuralForge agents.

    Subclasses must implement:
        - execute(): the core work
        - name property: human-readable identifier

    Optionally override lifecycle hooks:
        - on_init(): called once when agent is registered
        - on_before_execute(): pre-execution setup
        - on_after_execute(): post-execution cleanup
        - on_error(): error handling
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique agent name used for registry lookups."""
        ...

    @property
    def capabilities(self) -> list[AgentCapability]:
        """List of capabilities this agent provides."""
        return [AgentCapability.CUSTOM]

    @property
    def description(self) -> str:
        """Human-readable description for documentation."""
        return f"Agent: {self.name}"

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides)
    # ------------------------------------------------------------------

    def on_init(self) -> None:
        """Called once when the agent is registered in the registry."""

    def on_before_execute(self, config: dict[str, Any]) -> None:
        """Called before execute(). Use for setup, logging, etc."""

    def on_after_execute(self, result: Any) -> None:
        """Called after successful execute(). Use for cleanup."""

    def on_error(self, error: Exception) -> None:
        """Called when execute() raises. Use for error reporting."""

    # ------------------------------------------------------------------
    # Core execution (must implement)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def execute(
        self,
        config: dict[str, Any],
        dependencies: dict[str, Any],
        context: PipelineContext,
    ) -> Any:
        """Execute this agent's task.

        Args:
            config: Node configuration from the DAG definition.
            dependencies: Results from upstream dependency nodes.
            context: Shared pipeline context for cross-node communication.

        Returns:
            Arbitrary result that downstream nodes can consume.
        """
        ...

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate node configuration. Return list of error messages (empty = ok)."""
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
