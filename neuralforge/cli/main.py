"""NeuralForge CLI — powered by Typer."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

from neuralforge.core.engine import PipelineEngine
from neuralforge.core.context import PipelineState
from neuralforge.pipeline.loader import PipelineLoader
from neuralforge.pipeline.validator import PipelineValidator
from neuralforge.pipeline.templates import list_templates, get_template, TEMPLATES


def _get_app():
    """Lazy-create the Typer app."""
    if typer is None:
        print("Error: typer is required for CLI. Install: pip install 'neuralforge[cli]'")
        sys.exit(1)

    app = typer.Typer(
        name="neuralforge",
        help="NeuralForge - AI Pipeline Orchestration Framework",
        add_completion=False,
    )

    @app.command()
    def run(
        pipeline: str = typer.Argument(..., help="Path to pipeline YAML file"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, don't execute"),
        max_parallel: int = typer.Option(4, "--parallel", "-p", help="Max parallel workers"),
        context_file: Optional[str] = typer.Option(None, "--context", "-c", help="JSON context file"),
    ):
        """Run a pipeline from a YAML definition."""
        typer.echo(f"Loading pipeline: {pipeline}")
        try:
            dag, metadata = PipelineLoader.from_yaml(pipeline)
        except Exception as e:
            typer.echo(f"Error loading pipeline: {e}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Pipeline: {metadata['name']} v{metadata['version']}")
        typer.echo(f"Nodes: {len(dag.nodes)}")

        if dry_run:
            typer.echo("Dry run - validation passed!")
            raise typer.Exit(0)

        # Load context if provided
        initial_context = {}
        if context_file:
            with open(context_file) as f:
                initial_context = json.load(f)

        # Execute
        engine = PipelineEngine(max_parallel=max_parallel)
        typer.echo("Executing pipeline...")
        result = asyncio.run(engine.execute(dag, initial_context))

        # Print results
        typer.echo("\n" + "=" * 60)
        typer.echo("RESULTS")
        typer.echo("=" * 60)
        for node_id, node_result in result.items():
            if isinstance(node_result, Exception):
                typer.echo(f"  {node_id}: FAILED - {node_result}")
            elif isinstance(node_result, dict):
                typer.echo(f"  {node_id}: OK")
                for k, v in node_result.items():
                    typer.echo(f"    {k}: {str(v)[:80]}")
            else:
                typer.echo(f"  {node_id}: {str(node_result)[:100]}")

    @app.command()
    def validate(
        pipeline: str = typer.Argument(..., help="Path to pipeline YAML file"),
    ):
        """Validate a pipeline definition."""
        typer.echo(f"Validating: {pipeline}")
        try:
            dag, metadata = PipelineLoader.from_yaml(pipeline)
            typer.echo(f"Pipeline: {metadata['name']} v{metadata['version']}")
            typer.echo(f"Nodes: {len(dag.nodes)}")
            typer.echo("Validation passed!")
        except Exception as e:
            typer.echo(f"Validation failed: {e}", err=True)
            raise typer.Exit(1)

    @app.command()
    def templates():
        """List available pipeline templates."""
        typer.echo("Available templates:")
        for name in list_templates():
            tmpl = get_template(name)
            typer.echo(f"  {name}: {tmpl.get('description', '')}")
            typer.echo(f"    Nodes: {len(tmpl.get('nodes', []))}")

    @app.command()
    def scaffold(
        template: str = typer.Argument(..., help="Template name"),
        output: str = typer.Option("pipeline.yaml", "--output", "-o", help="Output file"),
    ):
        """Generate a pipeline YAML from a template."""
        try:
            tmpl = get_template(template)
        except KeyError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)

        yaml_str = PipelineLoader.to_yaml(
            PipelineLoader.from_dict(tmpl)[0],
            {"name": tmpl["name"], "version": tmpl["version"]},
        )
        Path(output).write_text(yaml_str)
        typer.echo(f"Generated: {output}")

    @app.command()
    def serve(
        host: str = typer.Option("0.0.0.0", help="Bind host"),
        port: int = typer.Option(8080, help="Bind port"),
    ):
        """Start the monitoring dashboard."""
        try:
            import uvicorn
            from neuralforge.web.app import create_dashboard_app
            app = create_dashboard_app()
            typer.echo(f"Dashboard: http://{host}:{port}")
            uvicorn.run(app, host=host, port=port)
        except ImportError:
            typer.echo("Error: uvicorn required. Install: pip install 'neuralforge[web]'")
            raise typer.Exit(1)

    return app


def main():
    """CLI entry point."""
    app = _get_app()
    app()


if __name__ == "__main__":
    main()
