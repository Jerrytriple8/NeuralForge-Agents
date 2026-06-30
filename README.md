# NeuralForge

**AI Agent Orchestration Framework**

NeuralForge is a comprehensive framework for building, orchestrating, and monitoring AI agent pipelines with built-in neural network capabilities.

## Features

- **Multi-Agent System**: Research, Code, and Critic agents working together
- **Neural Network Engine**: From-scratch implementation with multiple optimizers
- **Pipeline Orchestration**: Complex workflow management with dependencies
- **Observability**: Distributed tracing, metrics, and structured logging
- **REST API**: FastAPI-based monitoring and control interface
- **CLI Tools**: Command-line interface for quick operations

## Quick Start

```bash
# Install
pip install -e .

# Run CLI
neuralforge version
neuralforge agents
neuralforge train --epochs 100

# Start API server
neuralforge serve --port 8000
```

## Architecture

```
neuralforge/
├── core/           # Neural network, engine, memory
├── agents/         # AI agent implementations
├── pipeline/       # Workflow orchestration
├── observability/  # Tracing, metrics, logging
├── api/            # FastAPI server
├── cli/            # Command-line interface
├── tests/          # Test suite
├── configs/        # Configuration files
└── examples/       # Usage examples
```

## Usage Examples

### Neural Network Training

```python
from core import NeuralNetwork
from core.neural_net import LayerConfig
from core.optimizer import Adam
import numpy as np

# Create network
nn = NeuralNetwork()
nn.add_layer(LayerConfig(input_size=2, output_size=8, activation="relu"))
nn.add_layer(LayerConfig(input_size=8, output_size=1, activation="sigmoid"))
nn.set_loss("mse")
nn.set_optimizer(Adam(lr=0.01))

# Train
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
nn.fit(X, y, epochs=500, verbose=True)
```

### Multi-Agent Pipeline

```python
import asyncio
from core import NeuralEngine
from agents import ResearchAgent, CoderAgent, CriticAgent

async def main():
    engine = NeuralEngine()
    engine.register_agent("researcher", ResearchAgent())
    engine.register_agent("coder", CoderAgent())
    engine.register_agent("critic", CriticAgent())

    # Run research
    task = await engine.submit_task("researcher", {
        "task": "analyze",
        "data": "Your text here"
    })
    await asyncio.sleep(1)
    result = engine.get_task(task.task_id)
    print(result.result.output)

asyncio.run(main())
```

### Pipeline Workflow

```python
from pipeline import Workflow

wf = Workflow("analysis")
wf.transform("load", lambda ctx: load_data())
wf.filter("clean", lambda x: x is not None, depends_on=["load"])
wf.transform("process", lambda ctx: process(ctx["clean"]), depends_on=["clean"])

result = await wf.execute()
```

## Agents

| Agent | Purpose | Capabilities |
|-------|---------|--------------|
| Researcher | Information gathering | Analysis, extraction, summarization |
| Coder | Code generation | Generation, refactoring, debugging |
| Critic | Quality assessment | Scoring, comparison, bias detection |

## Neural Network

- **Activations**: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Swish
- **Losses**: MSE, Cross-Entropy, Binary Cross-Entropy
- **Optimizers**: Adam, SGD (with momentum), AdaGrad, RMSProp
- **Features**: Dropout, Batch Normalization, He initialization

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/agents` | List agents |
| POST | `/tasks` | Submit task |
| GET | `/tasks/{id}` | Get task status |
| GET | `/stats` | Engine statistics |
| GET | `/metrics` | Metrics data |
| GET | `/traces` | Trace data |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_engine.py -v

# Run with coverage
pytest tests/ --cov=neuralforge
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
engine:
  max_concurrent_tasks: 10
  timeout: 300.0
  enable_caching: true

neural_network:
  default_optimizer: adam
  default_lr: 0.001
```

## License

MIT License - see LICENSE file

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request
