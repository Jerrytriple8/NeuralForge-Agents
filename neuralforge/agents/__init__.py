"""Built-in agents package."""

from neuralforge.agents.base import BaseAgent, AgentCapability
from neuralforge.agents.registry import AgentRegistry
from neuralforge.agents.researcher import ResearchAgent
from neuralforge.agents.coder import CoderAgent
from neuralforge.agents.reviewer import ReviewAgent

__all__ = [
    "BaseAgent",
    "AgentCapability",
    "AgentRegistry",
    "ResearchAgent",
    "CoderAgent",
    "ReviewAgent",
]


def register_builtin_agents(registry: AgentRegistry) -> None:
    """Register all built-in agents in the given registry."""
    registry.register(ResearchAgent())
    registry.register(CoderAgent())
    registry.register(ReviewAgent())
