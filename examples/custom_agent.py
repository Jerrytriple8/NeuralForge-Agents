"""Custom agent example — demonstrates how to create and register a custom agent."""

import asyncio
from typing import Any

from neuralforge.agents.base import BaseAgent, AgentCapability
from neuralforge.agents.registry import AgentRegistry
from neuralforge.core.context import PipelineContext


class SummarizerAgent(BaseAgent):
    """Custom agent that summarizes text content.

    This example shows how to:
        1. Subclass BaseAgent
        2. Define capabilities
        3. Implement execute()
        4. Use dependency data from upstream nodes
    """

    @property
    def name(self) -> str:
        return "summarizer"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [AgentCapability.TEXT_GENERATION, AgentCapability.DATA_TRANSFORM]

    @property
    def description(self) -> str:
        return "Summarizes text content into bullet points."

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if "max_points" in config and not isinstance(config["max_points"], int):
            errors.append("'max_points' must be an integer")
        return errors

    async def execute(
        self,
        config: dict[str, Any],
        dependencies: dict[str, Any],
        context: PipelineContext,
    ) -> dict[str, Any]:
        max_points = config.get("max_points", 5)

        # Extract content from dependencies
        texts: list[str] = []
        for dep_name, dep_result in dependencies.items():
            if isinstance(dep_result, dict):
                if "results" in dep_result:
                    for r in dep_result["results"]:
                        texts.append(r.get("title", ""))
                if "code" in dep_result:
                    texts.append(dep_result["code"][:200])

        # Simulate summarization
        points = [f"Summary point {i+1}" for i in range(min(max_points, len(texts) or 1))]

        return {
            "summary": points,
            "point_count": len(points),
            "source_count": len(texts),
        }


async def main():
    # Register custom agent
    registry = AgentRegistry()
    registry.register(SummarizerAgent())

    # Use it
    agent = registry.get("summarizer")
    ctx = PipelineContext(pipeline_id="example")
    result = await agent.execute(
        {"max_points": 3},
        {"research": {"results": [{"title": "Finding 1"}, {"title": "Finding 2"}]}},
        ctx,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
