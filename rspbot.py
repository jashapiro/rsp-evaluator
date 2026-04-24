#!/usr/bin/env python3
"""
Research Sharing Plan Evalubot
Evaluates a target document against a policy and rubric using an LLM.
"""

import os

from pathlib import Path
from typing import Optional

# Must be set before any import that triggers transformers (langchain_community does so
# as a side effect when loading document loaders).
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import typer

from src.config import DEFAULT_BACKEND, DEFAULT_POLICY_PATH, DEFAULT_RUBRIC_PATH, resolve_model
from src.evaluator import evaluate_document
from src.llm import setup_llm

app = typer.Typer()

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc"}


def _resolve_path(path: Path, label: str) -> Path:
    """Return path as-is if it exists, otherwise try relative to cwd."""
    if path.exists():
        return path
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
    raise typer.BadParameter(f"{label} not found: {path}")


@app.command()
def evaluate(
    target: Path = typer.Argument(
        ..., help="Path to a document (PDF or Word) or a directory of documents to analyze"
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
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file (single input) or directory (directory input). "
             "Defaults to stdout for single files, or '<target>_evaluations/' for directories.",
    ),
):
    """Evaluate a document or directory of documents against a policy and rubric."""
    from rich.console import Console

    console = Console()

    if not target.exists():
        raise typer.BadParameter(f"Target not found: {target}")

    policy_path = _resolve_path(policy, "Policy file")
    rubric_path = _resolve_path(rubric, "Rubric file")

    try:
        llm = setup_llm(resolve_model(model_name, backend), backend, verbose)
    except Exception as e:
        console.print(f"[bold red]Error initializing LLM:[/bold red] {e}")
        raise typer.Exit(code=1)

    if target.is_dir():
        input_files = sorted(
            f for f in target.iterdir() if f.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not input_files:
            raise typer.BadParameter(f"No PDF or Word documents found in: {target}")

        output_dir = output or target.parent / f"{target.name}_evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, input_file in enumerate(input_files, 1):
            console.print(f"\n[bold]({i}/{len(input_files)}) Evaluating: {input_file.name}[/bold]")
            try:
                _run_evaluation(input_file, policy_path, rubric_path, llm, verbose,
                                output_dir / f"{input_file.stem}_evaluation.md", console)
            except Exception as e:
                console.print(f"[bold red]  Error evaluating document:[/bold red] {e}")

        console.print(f"\n[bold green]Done. Results written to: {output_dir}[/bold green]")

    else:
        if target.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise typer.BadParameter(f"Unsupported file type: {target.suffix}")

        try:
            _run_evaluation(target, policy_path, rubric_path, llm, verbose, output, console)
        except Exception as e:
            console.print(f"[bold red]Error evaluating document:[/bold red] {e}")
            raise typer.Exit(code=1)


def _run_evaluation(
    target_path: Path,
    policy_path: Path,
    rubric_path: Path,
    llm,
    verbose: bool,
    output_file: Optional[Path],
    console,
) -> None:
    """Run a single evaluation and write or print the result."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

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
                    console.print(f"  [green]✓[/green] {event['stage'].title()} completed in {event['elapsed']:.2f}s")
            elif event["type"] == "result":
                final_result = event["content"]
                console.print(f"  [bold green]Total time: {event['total_elapsed']:.2f}s[/bold green]")

    content = f"# Evaluation of: {target_path.name}\n\n{final_result}"
    if output_file:
        output_file.write_text(content)
        if verbose:
            console.print(f"  Evaluation written to: {output_file}")
    else:
        console.print(content)


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
