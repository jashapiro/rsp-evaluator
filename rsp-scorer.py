#!/usr/bin/env python3
"""
RSP Scorer
Evaluates a target document against a policy and rubric using an LLM.
"""

from pathlib import Path

import typer

from src.evaluator import evaluate_document

app = typer.Typer()


@app.command()
def main(
    target_file: Path = typer.Argument(
        ..., help="Path to the document file to analyze (PDF or Word)"
    ),
    policy: Path = typer.Option(
        "reference/alsf_resource_sharing_policy.pdf",
        "--policy",
        "-p",
        help="Path to the policy document",
    ),
    rubric: str = typer.Option(
        "reference/RSP-Rubric-4_11_23.docx",
        "--rubric",
        "-r",
        help="Path to the rubric document",
    ),
    model_name: str = typer.Option(
        "llama3.2", "--model", "-m", help="Ollama model to use for analysis"
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


if __name__ == "__main__":
    app()
