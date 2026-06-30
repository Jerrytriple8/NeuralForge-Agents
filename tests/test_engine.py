"""
Tests for NeuralEngine.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from core.engine import NeuralEngine, EngineConfig, TaskStatus
from agents.base import BaseAgent, AgentConfig, AgentResult


class MockAgent(BaseAgent):
    def __init__(self, name: str = "mock", delay: float = 0):
        super().__init__(AgentConfig(name=name))
        self.delay = delay

    async def execute(self, payload):
        if self.delay:
            await asyncio.sleep(self.delay)
        return AgentResult(
            agent_name=self.name,
            task_id="",
            output={"processed": True, "input": payload},
            confidence=1.0,
        )


class FailingAgent(BaseAgent):
    def __init__(self, fail_count: int = 1):
        super().__init__(AgentConfig(name="failing"))
        self.fail_count = fail_count
        self.call_count = 0

    async def execute(self, payload):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise ValueError(f"Fail #{self.call_count}")
        return AgentResult(agent_name=self.name, task_id="", output="success")


class TestEngine:
    def setup_method(self):
        self.config = EngineConfig(
            max_concurrent_tasks=5,
            max_retries=2,
            retry_delay=0.01,
            timeout=5.0,
        )
        self.engine = NeuralEngine(self.config)

    def teardown_method(self):
        asyncio.get_event_loop().run_until_complete(self.engine.shutdown())

    def test_register_agent(self):
        agent = MockAgent()
        self.engine.register_agent("test", agent)
        assert "test" in self.engine.registered_agents
        assert self.engine.get_agent("test") is agent

    def test_register_duplicate_raises(self):
        agent = MockAgent()
        self.engine.register_agent("test", agent)
        with pytest.raises(ValueError, match="already registered"):
            self.engine.register_agent("test", agent)

    def test_unregister_agent(self):
        agent = MockAgent()
        self.engine.register_agent("test", agent)
        self.engine.unregister_agent("test")
        assert "test" not in self.engine.registered_agents

    def test_get_missing_agent_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.engine.get_agent("nonexistent")

    def test_submit_task(self):
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        task = asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", {"data": "test"})
        )
        assert task.task_id is not None
        assert task.agent_name == "mock"

    def test_submit_unknown_agent_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            asyncio.get_event_loop().run_until_complete(
                self.engine.submit_task("unknown", {})
            )

    def test_task_completion(self):
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        task = asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", {"key": "value"})
        )
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        updated = self.engine.get_task(task.task_id)
        assert updated.status == TaskStatus.COMPLETED
        assert updated.result.output["processed"] is True

    def test_task_retry_on_failure(self):
        agent = FailingAgent(fail_count=1)
        self.engine.register_agent("failing", agent)
        task = asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("failing", {})
        )
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        updated = self.engine.get_task(task.task_id)
        assert updated.status == TaskStatus.COMPLETED

    def test_task_timeout(self):
        config = EngineConfig(timeout=0.1, max_retries=0, retry_delay=0.01)
        engine = NeuralEngine(config)
        agent = MockAgent(delay=10)
        engine.register_agent("slow", agent)
        task = asyncio.get_event_loop().run_until_complete(
            engine.submit_task("slow", {})
        )
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        updated = engine.get_task(task.task_id)
        assert updated.status == TaskStatus.FAILED
        asyncio.get_event_loop().run_until_complete(engine.shutdown())

    def test_cancel_task(self):
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        task = asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", {})
        )
        result = asyncio.get_event_loop().run_until_complete(
            self.engine.cancel_task(task.task_id)
        )
        # May or may not succeed depending on timing

    def test_cache_hit(self):
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        payload = {"key": "value"}
        task1 = asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", payload)
        )
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        task2 = asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", payload)
        )
        assert task2.status == TaskStatus.COMPLETED

    def test_stats(self):
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        stats = self.engine.stats
        assert "tasks_submitted" in stats
        assert "active_agents" in stats
        assert stats["active_agents"] == 1

    def test_hook_pre_task(self):
        hook_called = []
        self.engine.add_hook("pre_task", lambda task: hook_called.append(task.task_id))
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", {})
        )
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        assert len(hook_called) > 0

    def test_hook_on_complete(self):
        hook_called = []
        self.engine.add_hook("on_complete", lambda task: hook_called.append(task.task_id))
        agent = MockAgent()
        self.engine.register_agent("mock", agent)
        asyncio.get_event_loop().run_until_complete(
            self.engine.submit_task("mock", {})
        )
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
        assert len(hook_called) > 0

    def test_concurrent_tasks(self):
        agent = MockAgent(delay=0.1)
        self.engine.register_agent("mock", agent)
        tasks = []
        for i in range(5):
            task = asyncio.get_event_loop().run_until_complete(
                self.engine.submit_task("mock", {"i": i})
            )
            tasks.append(task)
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
        completed = sum(
            1 for t in tasks
            if self.engine.get_task(t.task_id).status == TaskStatus.COMPLETED
        )
        assert completed == 5

    def test_repr(self):
        assert "NeuralEngine" in repr(self.engine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
