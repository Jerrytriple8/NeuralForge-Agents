# NeuralForge

**AI Pipeline Orchestration Framework** — DAG-based execution engine with intelligent agents, middleware stack, and real-time observability.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    NeuralForge CLI                    │
├──────────┬──────────┬──────────┬────────────────────┤
│  DAG     │  Agent   │ Pipeline │  Observability     │
│  Engine  │  System  │ Loader   │  (tracing/metrics) │
├──────────┴──────────┴──────────┴────────────────────┤
│               Middleware Stack                        │
│  (cache | rate limiter | retry | circuit breaker)    │
├──────────────────────────────────────────────────────┤
│            Web Dashboard (FastAPI + WS)               │
└──────────────────────────────────────────────────────┘
```

## Features

- **DAG Execution Engine** — Topology-sorted, parallel node execution with dependency resolution
- **Intelligent Agents** — Plugin-based agent system with capabilities (research, code gen, review)
- **Pipeline DSL** — YAML-based pipeline definitions with validation
- **Middleware Stack** — Cache (LRU/TTL), rate limiter, retry with backoff, circuit breaker
- **Observability** — Distributed tracing, Prometheus-style metrics, structured JSON logging
- **Web Dashboard** — Real-time monitoring via FastAPI + WebSocket
- **CLI** — Full-featured CLI with Typer (run, validate, templates, scaffold, serve)

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Run a pipeline
neuralforge run examples/basic_pipeline.yaml --context '{"topic": "async Python"}'

# Validate without executing
neuralforge validate examples/basic_pipeline.yaml

# List templates
neuralforge templates

# Generate from template
neuralforge scaffold research_and_code -o my_pipeline.yaml

# Start monitoring dashboard
neuralforge serve --port 8080
```

## Pipeline YAML Format

```yaml
name: my_pipeline
version: "1.0"
nodes:
  - id: research
    agent: researcher
    config:
      query: "Research {topic}"
      sources: [google, arxiv]

  - id: generate
    agent: coder
    depends_on: [research]
    config:
      task: "Generate code from research"
      language: python

  - id: review
    agent: reviewer
    depends_on: [generate]
    config:
      language: python
      severity_threshold: warning
```

## Custom Agents

```python
from neuralforge.agents.base import BaseAgent, AgentCapability

class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "my_agent"

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [AgentCapability.CUSTOM]

    async def execute(self, config, dependencies, context):
        return {"result": "processed"}
```

## Built-in Agents

| Agent       | Capabilities                           |
|-------------|----------------------------------------|
| Researcher  | Web search, data synthesis             |
| Coder       | Code generation (Python, JS, Go, Rust) |
| Reviewer    | Static analysis, pattern detection     |

## Middleware

```python
from neuralforge.middleware import LRUCache, TokenBucketRateLimiter, RetryPolicy, CircuitBreaker

# LRU Cache with TTL
cache = LRUCache(capacity=1000)

# Rate Limiter
limiter = TokenBucketRateLimiter(rate=100, burst=50)

# Retry with exponential backoff
policy = RetryPolicy(max_retries=3, base_delay=1.0)

# Circuit Breaker
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
```

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
