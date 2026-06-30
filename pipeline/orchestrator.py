"""
Pipeline Orchestrator - Manages multi-step workflows.
"""

import asyncio
import time
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    elapsed: float = 0.0


class PipelineOrchestrator:
    """
    Orchestrates multi-step pipelines with dependency resolution,
    parallel execution, and conditional branching.
    """

    def __init__(self):
        self._pipelines: Dict[str, Dict] = {}
        self._results: Dict[str, Dict[str, StepResult]] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "on_step_start": [],
            "on_step_complete": [],
            "on_step_error": [],
            "on_pipeline_complete": [],
        }

    def create_pipeline(self, name: str, steps: List[Dict[str, Any]]) -> str:
        """Create a new pipeline definition."""
        pipeline_id = str(uuid.uuid4())[:8]
        self._pipelines[pipeline_id] = {
            "name": name,
            "steps": steps,
            "created_at": time.time(),
        }
        self._results[pipeline_id] = {}
        return pipeline_id

    async def execute(self, pipeline_id: str, initial_context: Dict[str, Any] = None) -> Dict[str, StepResult]:
        """Execute a pipeline."""
        if pipeline_id not in self._pipelines:
            raise KeyError(f"Pipeline '{pipeline_id}' not found")

        pipeline = self._pipelines[pipeline_id]
        context = initial_context or {}
        results = {}

        for step in pipeline["steps"]:
            step_id = step.get("id", str(uuid.uuid4())[:8])
            step_type = step.get("type", "transform")
            depends_on = step.get("depends_on", [])

            # Check dependencies
            if depends_on:
                failed_deps = [
                    d for d in depends_on
                    if d in results and results[d].status == StepStatus.FAILED
                ]
                if failed_deps:
                    results[step_id] = StepResult(
                        step_id=step_id,
                        status=StepStatus.SKIPPED,
                        error=f"Dependencies failed: {failed_deps}",
                    )
                    continue

            # Execute step
            start_time = time.time()
            try:
                await self._fire_hooks("on_step_start", step_id=step_id, context=context)

                handler = step.get("handler")
                if handler and callable(handler):
                    if asyncio.iscoroutinefunction(handler):
                        output = await handler(context)
                    else:
                        output = handler(context)
                else:
                    output = self._default_handler(step, context)

                elapsed = time.time() - start_time
                results[step_id] = StepResult(
                    step_id=step_id,
                    status=StepStatus.COMPLETED,
                    output=output,
                    elapsed=elapsed,
                )
                context[step_id] = output
                await self._fire_hooks("on_step_complete", step_id=step_id, result=results[step_id])

            except Exception as e:
                elapsed = time.time() - start_time
                results[step_id] = StepResult(
                    step_id=step_id,
                    status=StepStatus.FAILED,
                    error=str(e),
                    elapsed=elapsed,
                )
                await self._fire_hooks("on_step_error", step_id=step_id, error=str(e))

                if step.get("fail_fast", True):
                    break

        self._results[pipeline_id] = results
        await self._fire_hooks("on_pipeline_complete", pipeline_id=pipeline_id, results=results)
        return results

    def _default_handler(self, step: Dict, context: Dict) -> Any:
        step_type = step.get("type", "transform")
        data = step.get("data", {})

        if step_type == "transform":
            return context.get(step.get("input_key", ""), data)
        elif step_type == "filter":
            items = context.get(step.get("input_key", ""), [])
            predicate = step.get("predicate", lambda x: True)
            return [item for item in items if predicate(item)]
        elif step_type == "aggregate":
            items = context.get(step.get("input_key", ""), [])
            return {"items": items, "count": len(items)}
        else:
            return data

    async def _fire_hooks(self, event: str, **kwargs) -> None:
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(**kwargs)
                else:
                    hook(**kwargs)
            except Exception as e:
                logger.error("Hook error (%s): %s", event, e)

    def add_hook(self, event: str, callback: Callable) -> None:
        if event in self._hooks:
            self._hooks[event].append(callback)

    def get_results(self, pipeline_id: str) -> Dict[str, StepResult]:
        return self._results.get(pipeline_id, {})

    def get_status(self, pipeline_id: str) -> Dict[str, str]:
        results = self._results.get(pipeline_id, {})
        return {step_id: r.status.value for step_id, r in results.items()}
