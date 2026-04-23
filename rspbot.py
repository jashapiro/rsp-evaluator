#!/usr/bin/env python3
"""
Research Sharing Plan Evalubot
Evaluates a target document against a policy and rubric using an LLM.
"""

from pathlib import Path
from typing import Optional

import typer

from src.config import DEFAULT_BACKEND, DEFAULT_POLICY_PATH, DEFAULT_RUBRIC_PATH, resolve_model
from src.evaluator import evaluate_document
from src.llm import setup_llm

app = typer.Typer()


@app.command()
def evaluate(
    target_file: Path = typer.Argument(
        ..., help="Path to the document file to analyze (PDF or Word)"
    ),
    policy: Path = typer.Option(
        DEFAULT_POLICY_PATH, "--policy", "-p", help="Path to the policy document",
    ),
    rubric: Path = typer.Option(
        DEFAULT_RUBRIC_PATH, "--rubric", "-r", help="Path to the rubric document",
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model to use for analysis"
    ),
    backend: str = typer.Option(
        DEFAULT_BACKEND, "--backend", "-b", help="LLM backend to use: 'ollama' or 'mlx'"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """Evaluate a document against a policy and rubric."""
    # Validate paths
    target_path = Path(target_file)
    if not target_path.exists():
        raise typer.BadParameter(f"Target file not found: {target_file}")

    policy_path = Path(policy)
    if not policy_path.exists():
        if not policy_path.is_absolute():
            policy_path = Path.cwd() / policy
        if not policy_path.exists():
            raise typer.BadParameter(f"Policy file not found: {policy}")

    rubric_path = Path(rubric)
    if not rubric_path.exists():
        if not rubric_path.is_absolute():
            rubric_path = Path.cwd() / rubric
        if not rubric_path.exists():
            raise typer.BadParameter(f"Rubric file not found: {rubric}")

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    try:
        llm = setup_llm(resolve_model(model_name, backend), backend, verbose)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Starting...", total=None)
            final_result = ""

            for event in evaluate_document(target_path, policy_path, rubric_path, llm, verbose):
                if event["type"] == "status":
                    progress.update(task_id, description=event["message"])
                    if event.get("elapsed"):
                        console.print(f"[green]✓[/green] {event['stage'].title()} completed in {event['elapsed']:.2f}s")
                elif event["type"] == "result":
                    final_result = event["content"]
                    console.print(f"[bold green]Total time: {event['total_elapsed']:.2f}s[/bold green]")

        header = f"# Evaluation of: {target_path.name}\n\n"
        if output_file:
            Path(output_file).write_text(header + final_result)
            if verbose:
                console.print(f"Evaluation written to: {output_file}")
        else:
            console.print(header + final_result)

    except Exception as e:
        console.print(f"[bold red]Error evaluating document:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def summarize(
    target_file: Path = typer.Argument(
        ..., help="Path to the document file to analyze (PDF or Word)"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model to use for analysis"
    ),
    backend: str = typer.Option(
        DEFAULT_BACKEND, "--backend", "-b", help="LLM backend to use: 'ollama' or 'mlx'"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """Summarize the research plan from a document."""
    from rich.console import Console
    from src.evaluator import summarize_research_plan

    console = Console()
    try:
        llm = setup_llm(resolve_model(model_name, backend), backend, verbose)
        summary = summarize_research_plan(target_file, llm, verbose)
        if output_file:
            output_file.write_text(summary)
            console.print(f"Summary written to {output_file}")
        else:
            console.print(summary)
    except Exception as e:
        console.print(f"[bold red]Error summarizing document:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def extract(
    target_file: Path = typer.Argument(
        ..., help="Path to the document file to analyze (PDF or Word)"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model to use for analysis"
    ),
    backend: str = typer.Option(
        DEFAULT_BACKEND, "--backend", "-b", help="LLM backend to use: 'ollama' or 'mlx'"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """Extract the resource sharing plan from a document."""
    from rich.console import Console
    from src.evaluator import extract_sharing_plan

    console = Console()
    try:
        llm = setup_llm(resolve_model(model_name, backend), backend, verbose)
        sharing_plan = extract_sharing_plan(target_file, llm, verbose)
        if output_file:
            output_file.write_text(sharing_plan)
            console.print(f"Sharing plan written to {output_file}")
        else:
            console.print(sharing_plan)
    except Exception as e:
        console.print(f"[bold red]Error extracting sharing plan:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the web interface."""
    import uvicorn
    print(f"Starting web interface at http://{host}:{port}")
    uvicorn.run("src.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
