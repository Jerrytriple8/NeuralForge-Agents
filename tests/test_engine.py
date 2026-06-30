"""Tests for the pipeline execution engine."""

import asyncio

import pytest

from neuralforge.agents.base import BaseAgent
from neuralforge.agents.registry import AgentRegistry
from neuralforge.core.context import PipelineContext
from neuralforge.core.state import PipelineState
from neuralforge.core.dag import DAG, DAGNode
from neuralforge.core.engine import PipelineEngine


class DummyAgent(BaseAgent):
    """Agent that records its execution and returns config."""

    def __init__(self, agent_name: str = "dummy", delay: float = 0.0, fail_until: int = 0):
        self._name = agent_name
        self._delay = delay
        self._fail_until = fail_until
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, config, dependencies, context):
        self.call_count += 1
        if self.call_count <= self._fail_until:
            raise RuntimeError(f"Simulated failure #{self.call_count}")
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return {"agent": self._name, "config": config, "deps": dependencies}


class TestPipelineEngine:
    def _make_engine(self, agents: list[BaseAgent] | None = None) -> PipelineEngine:
        registry = AgentRegistry()
        for agent in (agents or [DummyAgent()]):
            registry.register(agent)
        return PipelineEngine(registry=registry)

    def _make_context(self, **kwargs) -> PipelineContext:
        return PipelineContext(**kwargs)

    @pytest.mark.asyncio
    async def test_single_node(self):
        engine = self._make_engine()
        dag = DAG()
        dag.add_node(DAGNode(node_id="test", name="Test", agent="dummy"))
        ctx = self._make_context(pipeline_name="single-node-test")

        result = await engine.execute(dag, ctx)
        assert result.get_node_result("test") is not None
        assert result.get_node_result("test")["agent"] == "dummy"

    @pytest.mark.asyncio
    async def test_sequential_pipeline(self):
        agent = DummyAgent()
        engine = self._make_engine([agent])
        dag = DAG()
        dag.add_node(DAGNode(node_id="step1", name="Step 1", agent="dummy"))
        dag.add_node(DAGNode(node_id="step2", name="Step 2", agent="dummy", depends_on=["step1"]))
        ctx = self._make_context(pipeline_name="sequential-test")

        result = await engine.execute(dag, ctx)
        assert result.get_node_result("step1") is not None
        assert result.get_node_result("step2") is not None
        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        agent = DummyAgent(fail_until=2)
        engine = self._make_engine([agent])
        dag = DAG()
        dag.add_node(DAGNode(
            node_id="flaky",
            name="Flaky Task",
            agent="dummy",
            retry_policy={"max_retries": 3, "backoff_base": 0.01},
        ))
        ctx = self._make_context(pipeline_name="retry-test")

        result = await engine.execute(dag, ctx)
        assert result.get_node_result("flaky") is not None
        assert agent.call_count == 3  # failed twice, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        agent = DummyAgent(fail_until=10)
        engine = self._make_engine([agent])
        dag = DAG()
        dag.add_node(DAGNode(
            node_id="bad",
            name="Bad Task",
            agent="dummy",
            retry_policy={"max_retries": 2, "backoff_base": 0.01},
        ))
        ctx = self._make_context(pipeline_name="exhaustion-test")

        result = await engine.execute(dag, ctx)
        node = dag.get_node("bad")
        assert node.error is not None

    @pytest.mark.asyncio
    async def test_context_propagation(self):
        engine = self._make_engine()
        dag = DAG()
        dag.add_node(DAGNode(node_id="test", name="Test", agent="dummy"))
        ctx = self._make_context(pipeline_name="context-test")
        ctx.set("custom_key", "value")

        result = await engine.execute(dag, ctx)
        assert result.get("custom_key") == "value"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        slow = DummyAgent(agent_name="slow", delay=0.1)
        fast = DummyAgent(agent_name="fast", delay=0.0)
        engine = self._make_engine([slow, fast])
        dag = DAG()
        dag.add_node(DAGNode(node_id="a", name="Slow", agent="slow"))
        dag.add_node(DAGNode(node_id="b", name="Fast", agent="fast"))
        ctx = self._make_context(pipeline_name="parallel-test")

        start = asyncio.get_event_loop().time()
        result = await engine.execute(dag, ctx)
        elapsed = asyncio.get_event_loop().time() - start

        assert result.get_node_result("a") is not None
        assert result.get_node_result("b") is not None
        assert elapsed < 0.2  # Should be ~0.1s (parallel), not 0.2s (sequential)
