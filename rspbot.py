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

    try:
        evaluation = evaluate_document(
            target_path,
            policy_path,
            rubric_path,
            model_name=model_name,
            verbose=verbose,
        )

        # Format output
        output = f"\n{'=' * 70}\nDOCUMENT EVALUATION REPORT\n{'=' * 70}\n{evaluation}\n{'=' * 70}\n"

        # Write to file or print to stdout
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(evaluation)
            if verbose:
                print(f"Evaluation written to: {output_file}")
        else:
            print(output)

    except Exception as e:
        print(f"Error evaluating document: {e}")
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
