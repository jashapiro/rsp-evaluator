#!/usr/bin/env python3
"""
Model Testing Script for RSP Evaluator

Tests various Ollama models on the RSP evaluation tasks:
1. Summarizing research plans
2. Extracting resource sharing plans
3. Full document evaluation

Models are selected based on their suitability for:
- Document understanding and summarization
- Information extraction
- Following complex instructions with structured output
- Long context handling
"""

import argparse
import json
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Path to the recommended models configuration
RECOMMENDED_MODELS_FILE = Path(__file__).parent / "recommended_models.toml"


def load_recommended_models() -> list[tuple[str, str]]:
    """Load recommended models from the TOML configuration file."""
    if not RECOMMENDED_MODELS_FILE.exists():
        print(f"Warning: {RECOMMENDED_MODELS_FILE} not found")
        return []

    with open(RECOMMENDED_MODELS_FILE, "rb") as f:
        config = tomllib.load(f)

    return [(m["name"], m["description"]) for m in config.get("models", [])]


@dataclass
class TestResult:
    """Result of a single model test."""
    model: str
    task: str
    success: bool
    elapsed_time: float
    output_file: Optional[Path] = None
    error: Optional[str] = None


def get_installed_models() -> list[str]:
    """Get list of currently installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except subprocess.CalledProcessError:
        print("Error: Could not get model list. Is Ollama running?")
        return []


def get_recommended_models_to_test(
    installed_models: list[str],
    recommended_models: list[tuple[str, str]]
) -> list[str]:
    """Get recommended models that are already installed."""
    models_to_test = []
    for model, _ in recommended_models:
        # Check if model is installed (exact match or base name match)
        base_name = model.split(":")[0]
        for installed in installed_models:
            if model == installed or base_name == installed.split(":")[0]:
                # Use the installed version name
                models_to_test.append(installed)
                break
    return models_to_test


def pull_model(model: str) -> bool:
    """Pull a model from Ollama."""
    print(f"  Pulling {model}...")
    try:
        # Don't capture output so user can see download progress
        result = subprocess.run(
            ["ollama", "pull", model],
            check=True
        )
        print(f"  Successfully pulled {model}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to pull {model}")
        return False


def ensure_model_available(model: str, installed_models: list[str]) -> bool:
    """Ensure a model is available, pulling if necessary."""
    # Check if already installed (exact match or base name match)
    for installed in installed_models:
        if model == installed or model.split(":")[0] == installed.split(":")[0]:
            return True

    # Try to pull the model
    return pull_model(model)


def run_summarize(
    target_file: Path,
    model: str,
    output_dir: Path,
    verbose: bool = False
) -> TestResult:
    """Run the summarize task."""
    output_file = output_dir / "summarize_output.md"
    start_time = time.time()

    try:
        cmd = [
            "pixi", "run", "python", "rspbot.py",
            "summarize", str(target_file),
            "--model", model,
            "--output", str(output_file)
        ]
        if verbose:
            cmd.append("--verbose")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0 and output_file.exists():
            return TestResult(
                model=model,
                task="summarize",
                success=True,
                elapsed_time=elapsed,
                output_file=output_file
            )
        else:
            return TestResult(
                model=model,
                task="summarize",
                success=False,
                elapsed_time=elapsed,
                error=result.stderr or result.stdout
            )
    except subprocess.TimeoutExpired:
        return TestResult(
            model=model,
            task="summarize",
            success=False,
            elapsed_time=600,
            error="Timeout after 10 minutes"
        )
    except Exception as e:
        return TestResult(
            model=model,
            task="summarize",
            success=False,
            elapsed_time=time.time() - start_time,
            error=str(e)
        )


def run_extract(
    target_file: Path,
    model: str,
    output_dir: Path,
    verbose: bool = False
) -> TestResult:
    """Run the extract task."""
    output_file = output_dir / "extract_output.md"
    start_time = time.time()

    try:
        cmd = [
            "pixi", "run", "python", "rspbot.py",
            "extract", str(target_file),
            "--model", model,
            "--output", str(output_file)
        ]
        if verbose:
            cmd.append("--verbose")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        elapsed = time.time() - start_time

        if result.returncode == 0 and output_file.exists():
            return TestResult(
                model=model,
                task="extract",
                success=True,
                elapsed_time=elapsed,
                output_file=output_file
            )
        else:
            return TestResult(
                model=model,
                task="extract",
                success=False,
                elapsed_time=elapsed,
                error=result.stderr or result.stdout
            )
    except subprocess.TimeoutExpired:
        return TestResult(
            model=model,
            task="extract",
            success=False,
            elapsed_time=600,
            error="Timeout after 10 minutes"
        )
    except Exception as e:
        return TestResult(
            model=model,
            task="extract",
            success=False,
            elapsed_time=time.time() - start_time,
            error=str(e)
        )


def run_eval(
    target_file: Path,
    model: str,
    output_dir: Path,
    policy_path: Path,
    rubric_path: Path,
    verbose: bool = False
) -> TestResult:
    """Run the full evaluation task."""
    output_file = output_dir / "evaluation_output.md"
    start_time = time.time()

    try:
        cmd = [
            "pixi", "run", "python", "rspbot.py",
            "eval", str(target_file),
            "--policy", str(policy_path),
            "--rubric", str(rubric_path),
            "--model", model,
            "--output", str(output_file)
        ]
        if verbose:
            cmd.append("--verbose")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200  # 20 minute timeout for full eval
        )

        elapsed = time.time() - start_time

        if result.returncode == 0 and output_file.exists():
            return TestResult(
                model=model,
                task="eval",
                success=True,
                elapsed_time=elapsed,
                output_file=output_file
            )
        else:
            return TestResult(
                model=model,
                task="eval",
                success=False,
                elapsed_time=elapsed,
                error=result.stderr or result.stdout
            )
    except subprocess.TimeoutExpired:
        return TestResult(
            model=model,
            task="eval",
            success=False,
            elapsed_time=1200,
            error="Timeout after 20 minutes"
        )
    except Exception as e:
        return TestResult(
            model=model,
            task="eval",
            success=False,
            elapsed_time=time.time() - start_time,
            error=str(e)
        )


def test_model(
    model: str,
    target_file: Path,
    policy_path: Path,
    rubric_path: Path,
    output_base: Path,
    verbose: bool = False,
    skip_eval: bool = False
) -> list[TestResult]:
    """Run all tests for a single model."""
    # Create model output directory (sanitize model name for filesystem)
    model_dir_name = model.replace(":", "_").replace("/", "_")
    output_dir = output_base / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\n{'=' * 60}")
    print(f"Testing model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Test 1: Summarize
    print("\n[1/3] Running summarize task...")
    result = run_summarize(target_file, model, output_dir, verbose)
    results.append(result)
    if result.success:
        print(f"  SUCCESS in {result.elapsed_time:.1f}s")
    else:
        print(f"  FAILED: {result.error[:100] if result.error else 'Unknown error'}")

    # Test 2: Extract
    print("\n[2/3] Running extract task...")
    result = run_extract(target_file, model, output_dir, verbose)
    results.append(result)
    if result.success:
        print(f"  SUCCESS in {result.elapsed_time:.1f}s")
    else:
        print(f"  FAILED: {result.error[:100] if result.error else 'Unknown error'}")

    # Test 3: Full evaluation
    if not skip_eval:
        print("\n[3/3] Running full evaluation task...")
        result = run_eval(target_file, model, output_dir, policy_path, rubric_path, verbose)
        results.append(result)
        if result.success:
            print(f"  SUCCESS in {result.elapsed_time:.1f}s")
        else:
            print(f"  FAILED: {result.error[:100] if result.error else 'Unknown error'}")
    else:
        print("\n[3/3] Skipping full evaluation (--skip-eval)")

    # Save metadata
    metadata = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "target_file": str(target_file),
        "results": [
            {
                "task": r.task,
                "success": r.success,
                "elapsed_time": r.elapsed_time,
                "output_file": str(r.output_file) if r.output_file else None,
                "error": r.error
            }
            for r in results
        ]
    }

    metadata_file = output_dir / "test_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return results


def print_summary(all_results: dict[str, list[TestResult]]):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<25} {'Summarize':<15} {'Extract':<15} {'Eval':<15}")
    print("-" * 70)

    for model, results in all_results.items():
        row = f"{model:<25}"
        for result in results:
            if result.success:
                status = f"OK ({result.elapsed_time:.0f}s)"
            else:
                status = "FAILED"
            row += f" {status:<15}"
        print(row)

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test various Ollama models on RSP evaluation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test installed recommended models
  python test_models.py

  # Download missing models and test all recommended
  python test_models.py --download

  # Test specific models
  python test_models.py --models llama3.2:latest qwen2.5:7b

  # Quick test (skip full evaluation)
  python test_models.py --skip-eval

  # List recommended models and their install status
  python test_models.py --list-models
"""
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: recommended models that are installed)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download any missing recommended models before testing"
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("grants/ALSF Redacted Grant for AI RSP Demo.docx"),
        help="Target document to evaluate"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("reference/alsf_resource_sharing_policy.pdf"),
        help="Policy document path"
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("reference/RSP-Rubric-4_11_23.docx"),
        help="Rubric document path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs"),
        help="Base directory for output files"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the full evaluation task (faster testing)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List installed and recommended models, then exit"
    )

    args = parser.parse_args()

    # Load recommended models from config
    recommended_models = load_recommended_models()
    if not recommended_models:
        print("Error: No recommended models configured")
        sys.exit(1)

    # Get installed models
    installed_models = get_installed_models()

    if args.list_models:
        print("Recommended models for RSP evaluation:")
        for model, description in recommended_models:
            is_installed = any(
                model.split(":")[0] == m.split(":")[0] for m in installed_models
            )
            status = "[INSTALLED]" if is_installed else "[NOT INSTALLED]"
            print(f"  {status:<15} {model:<20} {description}")
        return

    # Validate paths
    if not args.target.exists():
        print(f"Error: Target file not found: {args.target}")
        sys.exit(1)
    if not args.policy.exists():
        print(f"Error: Policy file not found: {args.policy}")
        sys.exit(1)
    if not args.rubric.exists():
        print(f"Error: Rubric file not found: {args.rubric}")
        sys.exit(1)

    # Determine which models to test
    if args.models:
        models_to_test = args.models
    else:
        models_to_test = get_recommended_models_to_test(installed_models, recommended_models)

    # Download missing models if requested
    if args.download:
        print("Checking for missing recommended models...")
        for model, desc in recommended_models:
            if not ensure_model_available(model, installed_models):
                print(f"  Could not download {model}")
            elif model not in models_to_test:
                models_to_test.append(model)
        # Refresh installed models list
        installed_models = get_installed_models()
        models_to_test = get_recommended_models_to_test(installed_models, recommended_models)

    if not models_to_test:
        print("No recommended models are installed.")
        print("Use --download to download them, or install manually with:")
        print("  ollama pull <model_name>")
        print("\nRecommended models:")
        for model, desc in recommended_models:
            print(f"  - {model}: {desc}")
        sys.exit(1)

    print(f"\nModels to test: {', '.join(models_to_test)}")
    print(f"Target document: {args.target}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Verify all models are available
    available_models = []
    for model in models_to_test:
        if ensure_model_available(model, installed_models):
            available_models.append(model)

    if not available_models:
        print("No models available to test!")
        sys.exit(1)

    # Run tests
    all_results = {}
    for model in available_models:
        results = test_model(
            model=model,
            target_file=args.target,
            policy_path=args.policy,
            rubric_path=args.rubric,
            output_base=args.output_dir,
            verbose=args.verbose,
            skip_eval=args.skip_eval
        )
        all_results[model] = results

    # Print summary
    print_summary(all_results)

    # Save overall summary
    summary_file = args.output_dir / "test_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "target_file": str(args.target),
        "models_tested": list(all_results.keys()),
        "results": {
            model: [
                {
                    "task": r.task,
                    "success": r.success,
                    "elapsed_time": r.elapsed_time
                }
                for r in results
            ]
            for model, results in all_results.items()
        }
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
