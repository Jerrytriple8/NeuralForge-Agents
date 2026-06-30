"""NeuralForge - AI Pipeline Orchestration Framework."""

__version__ = "0.1.0"
__author__ = "NeuralForge Contributors"

from neuralforge.core.engine import PipelineEngine
from neuralforge.core.dag import DAG, DAGNode
from neuralforge.agents.base import BaseAgent
from neuralforge.agents.registry import AgentRegistry

__all__ = [
    "PipelineEngine",
    "DAG",
    "DAGNode",
    "BaseAgent",
    "AgentRegistry",
]
