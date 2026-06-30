"""
Workflow definitions and execution engine.
"""

import asyncio
import time
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepType(Enum):
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    BRANCH = "branch"
    PARALLEL = "parallel"
    LOOP = "loop"


@dataclass
class WorkflowStep:
    step_id: str
    step_type: StepType
    handler: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None

    def should_run(self, context: Dict[str, Any]) -> bool:
        if self.condition:
            return self.condition(context)
        return True


class Workflow:
    """
    Declarative workflow definition with conditional branching,
    parallel execution, and loop support.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._steps: List[WorkflowStep] = []
        self._executed: List[str] = []
        self._context: Dict[str, Any] = {}

    def add_step(self, step: WorkflowStep) -> "Workflow":
        self._steps.append(step)
        return self

    def transform(self, step_id: str, handler: Callable, depends_on: List[str] = None) -> "Workflow":
        self.add_step(WorkflowStep(
            step_id=step_id,
            step_type=StepType.TRANSFORM,
            handler=handler,
            depends_on=depends_on or [],
        ))
        return self

    def filter(self, step_id: str, predicate: Callable, depends_on: List[str] = None) -> "Workflow":
        self.add_step(WorkflowStep(
            step_id=step_id,
            step_type=StepType.FILTER,
            handler=predicate,
            depends_on=depends_on or [],
        ))
        return self

    def branch(self, step_id: str, condition: Callable, true_step: str, false_step: str, depends_on: List[str] = None) -> "Workflow":
        self.add_step(WorkflowStep(
            step_id=step_id,
            step_type=StepType.BRANCH,
            config={"true_step": true_step, "false_step": false_step},
            condition=condition,
            depends_on=depends_on or [],
        ))
        return self

    async def execute(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the workflow."""
        self._context = initial_context or {}
        self._executed = []

        for step in self._steps:
            if step.step_id in self._executed:
                continue

            if not self._check_dependencies(step):
                logger.warning("Skipping step %s: dependencies not met", step.step_id)
                continue

            if not step.should_run(self._context):
                logger.info("Skipping step %s: condition not met", step.step_id)
                continue

            try:
                if step.step_type == StepType.BRANCH:
                    await self._execute_branch(step)
                elif step.step_type == StepType.PARALLEL:
                    await self._execute_parallel(step)
                elif step.step_type == StepType.LOOP:
                    await self._execute_loop(step)
                else:
                    await self._execute_step(step)

                self._executed.append(step.step_id)

            except Exception as e:
                logger.error("Step %s failed: %s", step.step_id, e)
                self._context[f"{step.step_id}_error"] = str(e)
                raise

        return self._context

    async def _execute_step(self, step: WorkflowStep) -> None:
        if step.handler:
            if asyncio.iscoroutinefunction(step.handler):
                result = await step.handler(self._context)
            else:
                result = step.handler(self._context)
            self._context[step.step_id] = result

    async def _execute_branch(self, step: WorkflowStep) -> None:
        if step.condition and step.condition(self._context):
            target = step.config.get("true_step")
        else:
            target = step.config.get("false_step")

        if target:
            target_step = next((s for s in self._steps if s.step_id == target), None)
            if target_step:
                await self._execute_step(target_step)
                self._executed.append(target)

    async def _execute_parallel(self, step: WorkflowStep) -> None:
        parallel_steps = step.config.get("steps", [])
        tasks = []
        for pid in parallel_steps:
            pstep = next((s for s in self._steps if s.step_id == pid), None)
            if pstep:
                tasks.append(self._execute_step(pstep))
        await asyncio.gather(*tasks)
        self._executed.extend(parallel_steps)

    async def _execute_loop(self, step: WorkflowStep) -> None:
        items = self._context.get(step.config.get("input_key", ""), [])
        handler = step.handler
        results = []
        for item in items:
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(self._context, item)
                else:
                    result = handler(self._context, item)
                results.append(result)
        self._context[step.step_id] = results

    def _check_dependencies(self, step: WorkflowStep) -> bool:
        return all(dep in self._executed for dep in step.depends_on)

    @property
    def executed_steps(self) -> List[str]:
        return self._executed.copy()

    @property
    def context(self) -> Dict[str, Any]:
        return self._context.copy()
