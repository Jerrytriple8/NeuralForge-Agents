"""
Tests for pipeline components.
"""

import asyncio
import pytest
from pipeline.orchestrator import PipelineOrchestrator, StepStatus
from pipeline.task_queue import TaskQueue, Priority
from pipeline.workflow import Workflow, WorkflowStep, StepType


class TestPipelineOrchestrator:
    def setup_method(self):
        self.orch = PipelineOrchestrator()

    def test_create_pipeline(self):
        steps = [{"id": "step1", "type": "transform", "data": "test"}]
        pid = self.orch.create_pipeline("test", steps)
        assert pid is not None

    def test_execute_pipeline(self):
        results = {}
        def handler(ctx):
            return {"result": "done"}

        steps = [{"id": "step1", "type": "transform", "handler": handler}]
        pid = self.orch.create_pipeline("test", steps)
        results = asyncio.get_event_loop().run_until_complete(
            self.orch.execute(pid)
        )
        assert "step1" in results
        assert results["step1"].status == StepStatus.COMPLETED

    def test_pipeline_with_dependencies(self):
        def handler1(ctx):
            return "data1"

        def handler2(ctx):
            return f"processed: {ctx.get('step1', '')}"

        steps = [
            {"id": "step1", "type": "transform", "handler": handler1},
            {"id": "step2", "type": "transform", "handler": handler2, "depends_on": ["step1"]},
        ]
        pid = self.orch.create_pipeline("test", steps)
        results = asyncio.get_event_loop().run_until_complete(
            self.orch.execute(pid)
        )
        assert results["step2"].output == "processed: data1"

    def test_pipeline_failure(self):
        def failing_handler(ctx):
            raise ValueError("Test error")

        steps = [
            {"id": "step1", "type": "transform", "handler": failing_handler},
            {"id": "step2", "type": "transform", "data": "test"},
        ]
        pid = self.orch.create_pipeline("test", steps)
        results = asyncio.get_event_loop().run_until_complete(
            self.orch.execute(pid)
        )
        assert results["step1"].status == StepStatus.FAILED

    def test_unknown_pipeline_raises(self):
        with pytest.raises(KeyError):
            asyncio.get_event_loop().run_until_complete(
                self.orch.execute("nonexistent")
            )


class TestTaskQueue:
    def setup_method(self):
        self.queue = TaskQueue(max_workers=2)

    def test_start_stop(self):
        asyncio.get_event_loop().run_until_complete(self.queue.start())
        assert self.queue.stats["workers"] == 2
        asyncio.get_event_loop().run_until_complete(self.queue.stop())

    def test_submit_task(self):
        async def run():
            await self.queue.start()
            task_id = await self.queue.submit(lambda: 42)
            result = await self.queue.get_result(task_id)
            await self.queue.stop()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == 42

    def test_priority_order(self):
        results = []

        async def run():
            await self.queue.start()
            await self.queue.submit(lambda: "low", priority=Priority.LOW)
            await self.queue.submit(lambda: "high", priority=Priority.HIGH)
            await asyncio.sleep(0.5)
            await self.queue.stop()

        asyncio.get_event_loop().run_until_complete(run())

    def test_queue_stats(self):
        stats = self.queue.stats
        assert "submitted" in stats
        assert "completed" in stats


class TestWorkflow:
    def test_create_workflow(self):
        wf = Workflow("test", "Test workflow")
        assert wf.name == "test"

    def test_add_steps(self):
        wf = Workflow("test")
        wf.transform("step1", lambda ctx: "data1")
        wf.transform("step2", lambda ctx: "data2", depends_on=["step1"])
        assert len(wf._steps) == 2

    def test_execute_workflow(self):
        wf = Workflow("test")
        wf.transform("step1", lambda ctx: "hello")
        wf.transform("step2", lambda ctx: f"{ctx['step1']} world")

        result = asyncio.get_event_loop().run_until_complete(wf.execute())
        assert result["step1"] == "hello"
        assert result["step2"] == "hello world"

    def test_workflow_with_filter(self):
        wf = Workflow("test")
        wf.transform("step1", lambda ctx: [1, 2, 3, 4, 5])
        wf.filter("step2", lambda x: x > 3, depends_on=["step1"])

        result = asyncio.get_event_loop().run_until_complete(wf.execute())
        assert result["step2"] == [4, 5]

    def test_workflow_context(self):
        wf = Workflow("test")
        wf.transform("step1", lambda ctx: ctx.get("input", "default"))

        result = asyncio.get_event_loop().run_until_complete(
            wf.execute({"input": "custom"})
        )
        assert result["step1"] == "custom"

    def test_executed_steps(self):
        wf = Workflow("test")
        wf.transform("step1", lambda ctx: "a")
        wf.transform("step2", lambda ctx: "b")

        asyncio.get_event_loop().run_until_complete(wf.execute())
        assert wf.executed_steps == ["step1", "step2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
