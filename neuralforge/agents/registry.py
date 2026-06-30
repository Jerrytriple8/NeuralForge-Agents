"""Agent registry — plugin-style registration and lookup."""

from __future__ import annotations

from typing import Any, Optional, Type

from neuralforge.agents.base import AgentCapability, BaseAgent


class AgentAlreadyRegistered(Exception):
    """Attempted to register an agent with a name that already exists."""


class AgentNotFound(Exception):
    """Requested agent name was not found in the registry."""


class AgentRegistry:
    """Central registry for all agent implementations.

    Agents can be registered via:
        1. Direct instance: registry.register(MyAgent())
        2. Decorator: @registry.agent("my_agent") class MyAgent(BaseAgent): ...
        3. Bulk: registry.register_many([agent1, agent2])
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._factories: dict[str, Type[BaseAgent]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        """Register a single agent instance."""
        if agent.name in self._agents:
            raise AgentAlreadyRegistered(f"Agent '{agent.name}' already registered")
        agent.on_init()
        self._agents[agent.name] = agent

    def register_many(self, agents: list[BaseAgent]) -> None:
        """Register multiple agents at once."""
        for agent in agents:
            self.register(agent)

    def register_factory(self, name: str, cls: Type[BaseAgent]) -> None:
        """Register a lazy factory — agent is instantiated on first use."""
        self._factories[name] = cls

    def agent(self, name: str):
        """Decorator for registering agent classes.

        Usage:
            @registry.agent("researcher")
            class ResearchAgent(BaseAgent):
                ...
        """

        def decorator(cls: Type[BaseAgent]):
            self.register_factory(name, cls)
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseAgent:
        """Get an agent by name, instantiating from factory if needed."""
        if name in self._agents:
            return self._agents[name]
        if name in self._factories:
            instance = self._factories[name]()
            self._agents[name] = instance
            instance.on_init()
            return instance
        raise AgentNotFound(
            f"Agent '{name}' not found. Available: {list(self._agents.keys()) + list(self._factories.keys())}"
        )

    def has(self, name: str) -> bool:
        return name in self._agents or name in self._factories

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return list(set(self._agents.keys()) | set(self._factories.keys()))

    def by_capability(self, cap: AgentCapability) -> list[BaseAgent]:
        """Find all agents that advertise a given capability."""
        return [a for a in self._agents.values() if cap in a.capabilities]

    def describe_all(self) -> dict[str, str]:
        """Map of agent name to description."""
        result = {}
        for name in self.names:
            try:
                result[name] = self.get(name).description
            except Exception:
                result[name] = "(factory not yet instantiated)"
        return result

    def __len__(self) -> int:
        return len(set(self._agents.keys()) | set(self._factories.keys()))

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={len(self)}, factories={len(self._factories)})"
