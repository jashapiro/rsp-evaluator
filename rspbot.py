#!/usr/bin/env python3
"""
Research Sharing Plan Evalubot
Evaluates a target document against a policy and rubric using an LLM.
"""

from pathlib import Path

import typer

from src.config import DEFAULT_MODEL_NAME, DEFAULT_POLICY_PATH, DEFAULT_RUBRIC_PATH
from src.evaluator import evaluate_document

app = typer.Typer()


@app.command()
def eval(
    target_file: Path = typer.Argument(
        ..., help="Path to the document file to analyze (PDF or Word)"
    ),
    policy: Path = typer.Option(
        DEFAULT_POLICY_PATH,
        "--policy",
        "-p",
        help="Path to the policy document",
    ),
    rubric: Path = typer.Option(
        DEFAULT_RUBRIC_PATH,
        "--rubric",
        "-r",
        help="Path to the rubric document",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODEL_NAME, "--model", "-m", help="Ollama model to use for analysis"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """
    Evaluate a document against a policy and rubric.
    """

    # Validate paths
    target_path = Path(target_file)
    if not target_path.exists():
        raise typer.BadParameter(f"Target file not found: {target_file}")

    policy_path = Path(policy)
    if not policy_path.exists():
        # Try relative to cwd if not found
        if not policy_path.is_absolute():
            policy_path = Path.cwd() / policy

        if not policy_path.exists():
            raise typer.BadParameter(f"Policy file not found: {policy}")

    rubric_path = Path(rubric)
    if not rubric_path.exists():
        # Try relative to cwd if not found
        if not rubric_path.is_absolute():
            rubric_path = Path.cwd() / rubric

        if not rubric_path.exists():
            raise typer.BadParameter(f"Rubric file not found: {rubric}")

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Starting...", total=None)

            eval_generator = evaluate_document(
                target_path,
                policy_path,
                rubric_path,
                model_name=model_name,
                verbose=verbose,
            )

            final_result = ""

            for event in eval_generator:
                if event["type"] == "status":
                    progress.update(task_id, description=f"{event['message']}")
                    if event.get("elapsed"):
                        console.print(
                            f"[green]✓[/green] {event['stage'].title()} completed in {event['elapsed']:.2f}s"
                        )
                elif event["type"] == "result":
                    final_result = event["content"]
                    console.print(
                        f"[bold green]Total time: {event['total_elapsed']:.2f}s[/bold green]"
                    )

        # Format output
        output = f"\n{'=' * 70}\nDOCUMENT EVALUATION REPORT\n{'=' * 70}\n{final_result}\n{'=' * 70}\n"

        # Write to file or print to stdout
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(final_result)
            if verbose:
                console.print(f"Evaluation written to: {output_file}")
        else:
            console.print(output)

    except Exception as e:
        console.print(f"[bold red]Error evaluating document:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def summarize(
    target_file: Path = typer.Argument(
        ..., help="Path to the document file to analyze (PDF or Word)"
    ),
    model_name: str = typer.Option(
        DEFAULT_MODEL_NAME, "--model", "-m", help="Ollama model to use for analysis"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """
    Summarize the research plan from a document.
    """
    from rich.console import Console

    from src.evaluator import summarize_research_plan

    console = Console()

    try:
        summary = summarize_research_plan(
            target_file, model_name=model_name, verbose=verbose
        )
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
    model_name: str = typer.Option(
        DEFAULT_MODEL_NAME, "--model", "-m", help="Ollama model to use for analysis"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """
    Extract the resource sharing plan from a document.
    """
    from rich.console import Console

    from src.evaluator import extract_sharing_plan

    console = Console()

    try:
        sharing_plan = extract_sharing_plan(
            target_file, model_name=model_name, verbose=verbose
        )
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
    """
    Start the web interface.
    """
    import uvicorn

    print(f"Starting web interface at http://{host}:{port}")
    uvicorn.run("src.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
