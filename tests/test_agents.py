"""Tests for agent system."""

import asyncio
import pytest

from neuralforge.agents.base import BaseAgent, AgentCapability
from neuralforge.agents.registry import AgentRegistry, AgentNotFound, AgentAlreadyRegistered
from neuralforge.agents.researcher import ResearchAgent
from neuralforge.agents.coder import CoderAgent
from neuralforge.agents.reviewer import ReviewAgent
from neuralforge.core.context import PipelineContext


class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = ResearchAgent()
        registry.register(agent)
        assert registry.get("researcher") is agent

    def test_duplicate_raises(self):
        registry = AgentRegistry()
        registry.register(ResearchAgent())
        with pytest.raises(AgentAlreadyRegistered):
            registry.register(ResearchAgent())

    def test_not_found_raises(self):
        registry = AgentRegistry()
        with pytest.raises(AgentNotFound):
            registry.get("nonexistent")

    def test_register_many(self):
        registry = AgentRegistry()
        registry.register_many([ResearchAgent(), CoderAgent(), ReviewAgent()])
        assert len(registry) == 3

    def test_names(self):
        registry = AgentRegistry()
        registry.register(ResearchAgent())
        assert "researcher" in registry.names

    def test_by_capability(self):
        registry = AgentRegistry()
        registry.register(ResearchAgent())
        registry.register(CoderAgent())
        agents = registry.by_capability(AgentCapability.WEB_SEARCH)
        assert len(agents) == 1
        assert agents[0].name == "researcher"

    def test_factory_registration(self):
        registry = AgentRegistry()
        registry.register_factory("researcher", ResearchAgent)
        assert "researcher" in registry
        agent = registry.get("researcher")
        assert isinstance(agent, ResearchAgent)

    def test_decorator_registration(self):
        registry = AgentRegistry()

        @registry.agent("custom")
        class CustomAgent(BaseAgent):
            @property
            def name(self):
                return "custom"

            async def execute(self, config, dependencies, context):
                return "ok"

        assert "custom" in registry


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_execute(self):
        agent = ResearchAgent()
        ctx = PipelineContext(pipeline_id="test")
        result = await agent.execute(
            {"query": "test", "sources": ["src1"], "max_results": 3},
            {},
            ctx,
        )
        assert result["count"] > 0
        assert result["sources_queried"] == 1

    def test_validate_config(self):
        agent = ResearchAgent()
        errors = agent.validate_config({})
        assert len(errors) > 0


class TestCoderAgent:
    @pytest.mark.asyncio
    async def test_execute(self):
        agent = CoderAgent()
        ctx = PipelineContext(pipeline_id="test")
        result = await agent.execute(
            {"task": "generate a function", "language": "python"},
            {},
            ctx,
        )
        assert "code" in result
        assert result["language"] == "python"
        assert result["lines"] > 0


class TestReviewAgent:
    @pytest.mark.asyncio
    async def test_review_clean_code(self):
        agent = ReviewAgent()
        ctx = PipelineContext(pipeline_id="test")
        result = await agent.execute(
            {"code": "x = 1\ny = 2\nprint(x + y)", "language": "python"},
            {},
            ctx,
        )
        assert "findings" in result
        assert "score" in result

    @pytest.mark.asyncio
    async def test_review_code_with_issues(self):
        agent = ReviewAgent()
        ctx = PipelineContext(pipeline_id="test")
        code = 'try:\n    pass\nexcept:\n    pass\napi_key = "sk-1234567890abcdef"'
        result = await agent.execute(
            {"code": code, "language": "python", "severity_threshold": "info"},
            {},
            ctx,
        )
        assert result["finding_count"] > 0
        rule_ids = {f["rule_id"] for f in result["findings"]}
        assert "PY002" in rule_ids  # bare except
        assert "PY005" in rule_ids  # hardcoded secret
