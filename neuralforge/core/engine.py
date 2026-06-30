"""Pipeline execution engine — the heart of NeuralForge.

Executes a DAG of agents with:
    - Parallel execution of independent nodes
    - Retry with exponential backoff
    - Middleware chain (cache, rate-limit, circuit-breaker)
    - Cancellation support
    - Full observability via tracer and metrics
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from neuralforge.agents.base import BaseAgent
from neuralforge.agents.registry import AgentRegistry
from neuralforge.core.context import PipelineContext
from neuralforge.core.dag import DAG, DAGNode, NodeStatus
from neuralforge.core.state import PipelineState, StateMachine
from neuralforge.middleware.circuit_breaker import CircuitBreaker
from neuralforge.middleware.rate_limiter import TokenBucketRateLimiter
from neuralforge.observability.logger import StructuredLogger
from neuralforge.observability.metrics import MetricsCollector
from neuralforge.observability.tracer import Tracer


class ExecutionPolicy(enum.Enum):
    """Controls how the engine handles node failures."""

    FAIL_FAST = "fail_fast"  # Cancel entire pipeline on first failure
    BEST_EFFORT = "best_effort"  # Continue other branches, fail at end
    IGNORE_FAILURES = "ignore"  # Skip failed nodes, treat as success


@dataclass
class EngineConfig:
    """Tuning knobs for the pipeline engine."""

    max_parallel: int = 8
    execution_policy: ExecutionPolicy = ExecutionPolicy.BEST_EFFORT
    global_timeout: float = 3600.0  # 1 hour default
    enable_cache: bool = True
    enable_tracing: bool = True
    rate_limit_rps: float = 100.0
    circuit_breaker_threshold: int = 5
    on_node_start: Optional[Callable[[DAGNode], None]] = None
    on_node_complete: Optional[Callable[[DAGNode], None]] = None
    on_node_error: Optional[Callable[[DAGNode, Exception], None]] = None
    on_pipeline_complete: Optional[Callable[[PipelineContext], None]] = None


class PipelineEngine:
    """Execute pipelines defined as DAGs with full middleware and observability.

    Usage:
        engine = PipelineEngine(registry=my_registry)
        dag = build_my_dag()
        ctx = PipelineContext(pipeline_name="example")
        result = await engine.execute(dag, ctx)
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.registry = registry or AgentRegistry()
        self.config = config or EngineConfig()
        self.state_machine = StateMachine()
        self.tracer = Tracer() if self.config.enable_tracing else None
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger("neuralforge.engine")
        self._rate_limiter = TokenBucketRateLimiter(self.config.rate_limit_rps)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold
        )
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self, dag: DAG, context: PipelineContext
    ) -> PipelineContext:
        """Execute a full DAG pipeline and return the populated context."""
        self._semaphore = asyncio.Semaphore(self.config.max_parallel)
        self.state_machine = StateMachine()

        # Validate
        self.state_machine.transition(PipelineState.VALIDATING)
        dag.validate()

        self.state_machine.transition(PipelineState.RUNNING)
        context.log_event("pipeline_start", data={"dag_nodes": len(dag.nodes)})
        self.metrics.increment("pipeline.started")

        try:
            await asyncio.wait_for(
                self._run_dag(dag, context),
                timeout=self.config.global_timeout,
            )

            # Check for any failures
            failed = [n for n in dag.nodes if n.status == NodeStatus.FAILED]
            if failed and self.config.execution_policy != ExecutionPolicy.IGNORE_FAILURES:
                self.state_machine.transition(PipelineState.FAILED)
                self.metrics.increment("pipeline.failed")
            else:
                self.state_machine.transition(PipelineState.SUCCESS)
                self.metrics.increment("pipeline.success")

        except asyncio.TimeoutError:
            self.state_machine.transition(PipelineState.TIMEOUT)
            self.metrics.increment("pipeline.timeout")
            await context.log_event("pipeline_timeout")

        except asyncio.CancelledError:
            self.state_machine.transition(PipelineState.CANCELLED)
            self.metrics.increment("pipeline.cancelled")
            await context.log_event("pipeline_cancelled")

        elapsed = time.monotonic() - context.started_at
        self.metrics.observe("pipeline.duration", elapsed)
        await context.log_event(
            "pipeline_end", data={"state": self.state_machine.state.value, "elapsed": elapsed}
        )

        if self.config.on_pipeline_complete:
            self.config.on_pipeline_complete(context)

        return context

    def cancel(self, context: PipelineContext) -> None:
        """Request graceful cancellation of a running pipeline."""
        context.cancel()
        self.logger.warn("Pipeline cancellation requested", run_id=context.run_id)

    # ------------------------------------------------------------------
    # Internal execution loop
    # ------------------------------------------------------------------

    async def _run_dag(self, dag: DAG, ctx: PipelineContext) -> None:
        """Execute DAG nodes level by level with parallel dispatch."""
        completed: set[str] = set()
        failed_nodes: set[str] = set()

        while True:
            ready = dag.ready_nodes(completed | failed_nodes)
            if not ready:
                break

            if ctx.is_cancelled:
                for node in ready:
                    node.status = NodeStatus.SKIPPED
                break

            tasks = [
                self._execute_node(node, ctx, completed, failed_nodes)
                for node in ready
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Check fail-fast policy
            if (
                self.config.execution_policy == ExecutionPolicy.FAIL_FAST
                and failed_nodes
            ):
                # Mark remaining as skipped
                for node in dag.nodes:
                    if node.status == NodeStatus.PENDING:
                        node.status = NodeStatus.SKIPPED
                break

    async def _execute_node(
        self,
        node: DAGNode,
        ctx: PipelineContext,
        completed: set[str],
        failed_nodes: set[str],
    ) -> None:
        """Execute a single node with retry, timeout, and middleware."""
        async with self._semaphore:
            # Check condition
            if node.condition and not node.condition(ctx):
                node.status = NodeStatus.SKIPPED
                completed.add(node.node_id)
                await ctx.log_event("node_skipped", node_id=node.node_id)
                return

            # Acquire rate limit token
            await self._rate_limiter.acquire()

            # Circuit breaker check
            if not self._circuit_breaker.allow():
                node.status = NodeStatus.FAILED
                node.error = RuntimeError("Circuit breaker open")
                failed_nodes.add(node.node_id)
                return

            if self.config.on_node_start:
                self.config.on_node_start(node)

            max_retries = node.retry_policy.get("max_retries", 3)
            backoff_base = node.retry_policy.get("backoff_base", 2)

            for attempt in range(max_retries + 1):
                node.attempt = attempt
                node.status = NodeStatus.RUNNING if attempt == 0 else NodeStatus.RETRYING
                await ctx.log_event("node_start", node_id=node.node_id, data={"attempt": attempt})
                self.metrics.increment("node.started", tags={"node": node.node_id})
                span = (
                    self.tracer.start_span(f"node:{node.node_id}") if self.tracer else None
                )

                try:
                    agent = self.registry.get(node.agent)
                    dep_results = ctx.get_dependency_results(node.depends_on)

                    if node.timeout > 0:
                        result = await asyncio.wait_for(
                            agent.execute(node.config, dep_results, ctx),
                            timeout=node.timeout,
                        )
                    else:
                        result = await agent.execute(node.config, dep_results, ctx)

                    node.result = result
                    node.status = NodeStatus.SUCCESS
                    ctx.set_node_result(node.node_id, result)
                    completed.add(node.node_id)
                    self._circuit_breaker.record_success()

                    self.metrics.increment("node.success", tags={"node": node.node_id})
                    if span:
                        span.set_attribute("status", "ok")
                        self.tracer.end_span(span)

                    if self.config.on_node_complete:
                        self.config.on_node_complete(node)

                    await ctx.log_event("node_complete", node_id=node.node_id)
                    return

                except Exception as exc:
                    node.error = exc
                    self.metrics.increment("node.error", tags={"node": node.node_id})

                    if span:
                        span.set_attribute("error", str(exc))
                        self.tracer.end_span(span)

                    if attempt < max_retries:
                        delay = backoff_base ** attempt
                        self.logger.warn(
                            "Node failed, retrying",
                            node=node.node_id,
                            attempt=attempt,
                            delay=delay,
                            error=str(exc),
                        )
                        await ctx.log_event(
                            "node_retry",
                            node_id=node.node_id,
                            data={"attempt": attempt, "error": str(exc)},
                        )
                        await asyncio.sleep(delay)
                    else:
                        node.status = NodeStatus.FAILED
                        failed_nodes.add(node.node_id)
                        self._circuit_breaker.record_failure()
                        self.logger.error(
                            "Node failed after retries",
                            node=node.node_id,
                            attempts=attempt + 1,
                            error=str(exc),
                        )
                        if self.config.on_node_error:
                            self.config.on_node_error(node, exc)
                        await ctx.log_event(
                            "node_failed",
                            node_id=node.node_id,
                            data={"error": str(exc)},
                        )
