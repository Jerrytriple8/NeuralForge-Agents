"""
NeuralForge CLI - Command-line interface for agent orchestration.
"""

import asyncio
import json
import sys
import time
from typing import Optional

try:
    import typer
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False

from core import NeuralEngine, NeuralNetwork, Layer
from core.neural_net import LayerConfig
from core.optimizer import Adam
from agents import ResearchAgent, CoderAgent, CriticAgent
from agents.base import AgentConfig


if HAS_TYPER:
    app = typer.Typer(
        name="neuralforge",
        help="NeuralForge - AI Agent Orchestration Framework",
        add_completion=False,
    )

    @app.command()
    def version():
        """Show NeuralForge version."""
        typer.echo("NeuralForge v2.0.0")

    @app.command()
    def agents():
        """List available agents."""
        available = [
            ("researcher", "Analyzes tasks and gathers information"),
            ("coder", "Generates, refactors, and debugs code"),
            ("critic", "Reviews and scores outputs"),
        ]
        typer.echo("Available Agents:")
        for name, desc in available:
            typer.echo(f"  {name:15} - {desc}")

    @app.command()
    def run(
        agent_name: str = typer.Argument(..., help="Agent to run"),
        task: str = typer.Option("analyze", help="Task type"),
        data: str = typer.Option("", help="Input data"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ):
        """Run an agent on a task."""
        async def _run():
            engine = NeuralEngine()

            agent_map = {
                "researcher": ResearchAgent(),
                "coder": CoderAgent(),
                "critic": CriticAgent(),
            }

            if agent_name not in agent_map:
                typer.echo(f"Unknown agent: {agent_name}")
                raise typer.Exit(1)

            engine.register_agent(agent_name, agent_map[agent_name])
            result = await engine.submit_task(
                agent_name=agent_name,
                payload={"task": task, "data": data},
            )

            # Wait for completion
            await asyncio.sleep(1)
            task_result = engine.get_task(result.task_id)

            if task_result and task_result.status.value == "completed":
                typer.echo(json.dumps(task_result.result, indent=2, default=str))
            elif task_result:
                typer.echo(f"Status: {task_result.status.value}")
                if task_result.error:
                    typer.echo(f"Error: {task_result.error}")
            else:
                typer.echo("Task not found")

            await engine.shutdown()

        asyncio.run(_run())

    @app.command()
    def train(
        epochs: int = typer.Option(100, help="Training epochs"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        batch_size: int = typer.Option(32, help="Batch size"),
    ):
        """Train a demo neural network."""
        import numpy as np

        typer.echo(f"Training neural network: epochs={epochs}, lr={lr}, batch_size={batch_size}")

        # Create XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)

        # Build network
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=8, activation="relu"))
        nn.add_layer(LayerConfig(input_size=8, output_size=4, activation="relu"))
        nn.add_layer(LayerConfig(input_size=4, output_size=1, activation="sigmoid"))
        nn.set_loss("mse")
        nn.set_optimizer(Adam(lr=lr))

        typer.echo(nn.summary())
        typer.echo("\nTraining...")

        history = nn.fit(X, y, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

        typer.echo(f"\nFinal loss: {history[-1]['loss']:.6f}")

        # Test predictions
        predictions = nn.predict(X)
        typer.echo("\nPredictions:")
        for i in range(4):
            typer.echo(f"  {X[i]} -> {predictions[i][0]:.4f} (target: {y[i][0]})")

    @app.command()
    def serve(
        host: str = typer.Option("0.0.0.0", help="Host"),
        port: int = typer.Option(8000, help="Port"),
    ):
        """Start the API server."""
        try:
            import uvicorn
            from api import create_app

            engine = NeuralEngine()
            for name, agent_cls in [
                ("researcher", ResearchAgent()),
                ("coder", CoderAgent()),
                ("critic", CriticAgent()),
            ]:
                engine.register_agent(name, agent_cls)

            app_instance = create_app(engine=engine)
            typer.echo(f"Starting NeuralForge API on {host}:{port}")
            uvicorn.run(app_instance, host=host, port=port)
        except ImportError:
            typer.echo("Install uvicorn: pip install uvicorn")
            raise typer.Exit(1)


    def cli():
        app()
else:
    def cli():
        print("Typer not installed. Install with: pip install typer")
        sys.exit(1)


if __name__ == "__main__":
    cli()
