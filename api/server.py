"""
FastAPI server for NeuralForge monitoring and control.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


if HAS_FASTAPI:
    class TaskRequest(BaseModel):
        agent_name: str
        payload: Dict[str, Any]
        priority: int = 0
        use_cache: bool = True

    class TaskResponse(BaseModel):
        task_id: str
        status: str
        result: Optional[Any] = None
        error: Optional[str] = None

    class AgentInfo(BaseModel):
        name: str
        description: str
        history_count: int


def create_app(engine=None, tracer=None, metrics=None):
    """Create FastAPI application for NeuralForge monitoring."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="NeuralForge API",
        description="AI Agent Orchestration Framework - Monitoring & Control API",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "name": "NeuralForge",
            "version": "2.0.0",
            "status": "running",
            "agents": len(engine.registered_agents) if engine else 0,
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time(),
        }

    @app.get("/agents", response_model=List[AgentInfo])
    async def list_agents():
        if not engine:
            raise HTTPException(503, "Engine not initialized")
        return [
            AgentInfo(
                name=name,
                description=engine.get_agent(name).config.description,
                history_count=len(engine.get_agent(name).history),
            )
            for name in engine.registered_agents
        ]

    @app.post("/tasks", response_model=TaskResponse)
    async def submit_task(request: TaskRequest):
        if not engine:
            raise HTTPException(503, "Engine not initialized")
        try:
            task = await engine.submit_task(
                agent_name=request.agent_name,
                payload=request.payload,
                priority=request.priority,
                use_cache=request.use_cache,
            )
            return TaskResponse(
                task_id=task.task_id,
                status=task.status.value,
                result=task.result,
                error=task.error,
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str):
        if not engine:
            raise HTTPException(503, "Engine not initialized")
        task = engine.get_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return TaskResponse(
            task_id=task.task_id,
            status=task.status.value,
            result=task.result,
            error=task.error,
        )

    @app.get("/stats")
    async def get_stats():
        stats = {}
        if engine:
            stats["engine"] = engine.stats
        if metrics:
            stats["metrics"] = metrics.get_all_metrics()
        if tracer:
            stats["tracing"] = tracer.get_stats()
        return stats

    @app.get("/metrics")
    async def get_metrics():
        if not metrics:
            raise HTTPException(503, "Metrics not configured")
        return metrics.get_all_metrics()

    @app.get("/traces")
    async def get_traces():
        if not tracer:
            raise HTTPException(503, "Tracer not configured")
        return [span.to_dict() for span in tracer.get_all_spans()]

    return app
