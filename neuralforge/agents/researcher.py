"""Built-in Research Agent — gathers and synthesizes information."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from neuralforge.agents.base import AgentCapability, BaseAgent
from neuralforge.core.context import PipelineContext


class ResearchAgent(BaseAgent):
    """Research agent that queries multiple sources and synthesizes results.

    Config:
        query: Search query string or template (supports {dep_name} interpolation).
        sources: List of source identifiers to query.
        max_results: Maximum results per source (default 10).
        strategy: "merge" | "intersect" | "best" (default "merge").
    """

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [AgentCapability.WEB_SEARCH, AgentCapability.DATA_TRANSFORM]

    @property
    def description(self) -> str:
        return "Gathers information from multiple sources and synthesizes findings."

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if "query" not in config:
            errors.append("Missing required config key: 'query'")
        if "sources" in config and not isinstance(config["sources"], list):
            errors.append("'sources' must be a list")
        return errors

    async def execute(
        self,
        config: dict[str, Any],
        dependencies: dict[str, Any],
        context: PipelineContext,
    ) -> dict[str, Any]:
        query = self._interpolate(config["query"], dependencies)
        sources = config.get("sources", ["default"])
        max_results = config.get("max_results", 10)
        strategy = config.get("strategy", "merge")

        # Parallel query all sources
        tasks = [
            self._query_source(source, query, max_results) for source in sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = {
            src: res
            for src, res in zip(sources, results)
            if not isinstance(res, Exception)
        }

        # Apply strategy
        synthesized = self._synthesize(valid_results, strategy)

        return {
            "query": query,
            "sources_queried": len(sources),
            "sources_succeeded": len(valid_results),
            "strategy": strategy,
            "results": synthesized,
            "count": len(synthesized),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(template: str, deps: dict[str, Any]) -> str:
        """Replace {dep_name} placeholders with dependency results."""
        result = template
        for key, value in deps.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    async def _query_source(
        self, source: str, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Simulate querying a data source. Override for real implementations."""
        await asyncio.sleep(0.01)  # Simulate network latency
        # Placeholder: generate deterministic results from query hash
        digest = hashlib.md5(f"{source}:{query}".encode()).hexdigest()
        return [
            {
                "source": source,
                "index": i,
                "title": f"Result {i} from {source}",
                "score": round(0.9 - i * 0.05, 2),
                "ref": f"{digest[:8]}_{i}",
            }
            for i in range(min(max_results, 3))
        ]

    @staticmethod
    def _synthesize(
        source_results: dict[str, list[dict]], strategy: str
    ) -> list[dict[str, Any]]:
        """Combine results from multiple sources based on strategy."""
        if strategy == "merge":
            combined = []
            for results in source_results.values():
                combined.extend(results)
            return sorted(combined, key=lambda r: r.get("score", 0), reverse=True)

        if strategy == "intersect":
            if not source_results:
                return []
            refs_sets = [
                {r["ref"] for r in results} for results in source_results.values()
            ]
            common = refs_sets[0].intersection(*refs_sets[1:])
            all_results = {
                r["ref"]: r
                for results in source_results.values()
                for r in results
            }
            return [all_results[ref] for ref in common]

        if strategy == "best":
            all_results = []
            for results in source_results.values():
                all_results.extend(results)
            return sorted(all_results, key=lambda r: r.get("score", 0))[:1]

        return []
