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
import re
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

# Reference file for extraction comparison
REFERENCE_EXTRACT_FILE = Path(__file__).parent / "model_outputs" / "extracted_sharing_plan.md"


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace for comparison."""
    # Remove extra whitespace and normalize line endings
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compare_extraction(extracted_file: Path, reference_file: Path) -> dict:
    """Compare extracted text to reference, excluding whitespace differences.

    Returns a dict with:
    - match: bool indicating if content matches (ignoring whitespace)
    - similarity: float 0-1 indicating character-level similarity
    - diff_chars: number of different characters
    - extracted_len: length of normalized extracted text
    - reference_len: length of normalized reference text
    """
    if not extracted_file.exists():
        return {"match": False, "error": "Extracted file not found"}
    if not reference_file.exists():
        return {"match": False, "error": "Reference file not found"}

    extracted = normalize_text(extracted_file.read_text())
    reference = normalize_text(reference_file.read_text())

    # Exact match after normalization
    match = extracted == reference

    # Calculate character-level similarity using longest common subsequence ratio
    # Simpler approach: count matching characters at each position
    max_len = max(len(extracted), len(reference))
    if max_len == 0:
        return {"match": True, "similarity": 1.0, "diff_chars": 0,
                "extracted_len": 0, "reference_len": 0}

    # Use difflib for better similarity calculation
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, extracted, reference)
    similarity = matcher.ratio()

    # Count actual character differences
    diff_chars = abs(len(extracted) - len(reference))
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != 'equal':
            diff_chars += max(i2 - i1, j2 - j1)

    return {
        "match": match,
        "similarity": round(similarity, 4),
        "diff_chars": diff_chars,
        "extracted_len": len(extracted),
        "reference_len": len(reference)
    }


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
    extraction_comparison: Optional[dict] = None  # For extract task only


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
        # Check if model is installed
        # If recommended model has a specific tag (e.g., qwen2.5:32b), require exact match
        # If no tag specified, match by base name
        if ":" in model:
            # Exact tag specified - require exact match
            recommended_base, recommended_tag = model.split(":", 1)
            for installed in installed_models:
                installed_base = installed.split(":")[0]
                installed_tag = installed.split(":", 1)[1] if ":" in installed else "latest"
                if recommended_base == installed_base and recommended_tag == installed_tag:
                    models_to_test.append(installed)
                    break
        else:
            # No tag specified - match any version of this model
            for installed in installed_models:
                if model == installed.split(":")[0]:
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
    # Check if already installed with exact tag match
    if ":" in model:
        model_base, model_tag = model.split(":", 1)
        for installed in installed_models:
            installed_base = installed.split(":")[0]
            installed_tag = installed.split(":", 1)[1] if ":" in installed else "latest"
            if model_base == installed_base and model_tag == installed_tag:
                return True
    else:
        # No tag specified - match any version
        for installed in installed_models:
            if model == installed.split(":")[0]:
                return True

    # Try to pull the model
    return pull_model(model)


def get_output_filename(base_name: str, run: Optional[int] = None) -> str:
    """Get output filename with optional run suffix.

    Args:
        base_name: Base filename like 'summarize_output.md'
        run: Optional run number (1, 2, 3, etc.)

    Returns:
        Filename like 'summarize_output_run1.md' or 'summarize_output.md' if run is None
    """
    if run is None:
        return base_name
    stem = base_name.rsplit('.', 1)[0]
    ext = base_name.rsplit('.', 1)[1] if '.' in base_name else ''
    return f"{stem}_run{run}.{ext}" if ext else f"{stem}_run{run}"


def check_completed_tasks(output_dir: Path, skip_eval: bool = False, run: Optional[int] = None) -> dict[str, bool]:
    """Check which tasks have already been completed for a model.

    Returns a dict with task names as keys and completion status as values.
    A task is considered complete if its output file exists and has content.
    """
    tasks = {
        "summarize": output_dir / get_output_filename("summarize_output.md", run),
        "extract": output_dir / get_output_filename("extract_output.md", run),
    }
    if not skip_eval:
        tasks["eval"] = output_dir / get_output_filename("evaluation_output.md", run)

    completed = {}
    for task, output_file in tasks.items():
        # Check if file exists and has content (more than just whitespace)
        if output_file.exists():
            content = output_file.read_text().strip()
            completed[task] = len(content) > 0
        else:
            completed[task] = False

    return completed


def run_summarize(
    target_file: Path,
    model: str,
    output_dir: Path,
    verbose: bool = False,
    run: Optional[int] = None
) -> TestResult:
    """Run the summarize task."""
    output_file = output_dir / get_output_filename("summarize_output.md", run)
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
    verbose: bool = False,
    run: Optional[int] = None
) -> TestResult:
    """Run the extract task."""
    output_file = output_dir / get_output_filename("extract_output.md", run)
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
            # Compare extraction to reference
            comparison = compare_extraction(output_file, REFERENCE_EXTRACT_FILE)
            return TestResult(
                model=model,
                task="extract",
                success=True,
                elapsed_time=elapsed,
                output_file=output_file,
                extraction_comparison=comparison
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
    verbose: bool = False,
    run: Optional[int] = None
) -> TestResult:
    """Run the full evaluation task."""
    output_file = output_dir / get_output_filename("evaluation_output.md", run)
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
    skip_eval: bool = False,
    force: bool = False,
    run: Optional[int] = None
) -> tuple[list[TestResult], bool]:
    """Run all tests for a single model.

    Returns a tuple of (results, was_skipped) where was_skipped is True if
    all tasks were already completed.
    """
    # Create model output directory (sanitize model name for filesystem)
    model_dir_name = model.replace(":", "_").replace("/", "_")
    output_dir = output_base / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for already completed tasks
    completed = check_completed_tasks(output_dir, skip_eval, run)

    run_label = f" (run {run})" if run else ""

    # If all tasks are completed and not forcing, skip this model
    if not force and all(completed.values()) and len(completed) > 0:
        print(f"\n{'=' * 60}")
        print(f"SKIPPING model: {model}{run_label} (all tasks already completed)")
        print(f"  Use --force to re-run tests")
        print(f"{'=' * 60}")
        return [], True

    results = []

    print(f"\n{'=' * 60}")
    print(f"Testing model: {model}{run_label}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Test 1: Summarize
    if not force and completed.get("summarize", False):
        print("\n[1/3] Summarize task already completed, skipping...")
    else:
        print("\n[1/3] Running summarize task...")
        result = run_summarize(target_file, model, output_dir, verbose, run)
        results.append(result)
        if result.success:
            print(f"  SUCCESS in {result.elapsed_time:.1f}s")
        else:
            print(f"  FAILED: {result.error[:100] if result.error else 'Unknown error'}")

    # Test 2: Extract
    if not force and completed.get("extract", False):
        print("\n[2/3] Extract task already completed, skipping...")
    else:
        print("\n[2/3] Running extract task...")
        result = run_extract(target_file, model, output_dir, verbose, run)
        results.append(result)
        if result.success:
            sim = result.extraction_comparison.get('similarity', 0) if result.extraction_comparison else 0
            match = result.extraction_comparison.get('match', False) if result.extraction_comparison else False
            match_str = "EXACT MATCH" if match else f"similarity={sim:.1%}"
            print(f"  SUCCESS in {result.elapsed_time:.1f}s ({match_str})")
        else:
            print(f"  FAILED: {result.error[:100] if result.error else 'Unknown error'}")

    # Test 3: Full evaluation
    if not skip_eval:
        if not force and completed.get("eval", False):
            print("\n[3/3] Evaluation task already completed, skipping...")
        else:
            print("\n[3/3] Running full evaluation task...")
            result = run_eval(target_file, model, output_dir, policy_path, rubric_path, verbose, run)
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
        "run": run,
        "timestamp": datetime.now().isoformat(),
        "target_file": str(target_file),
        "results": [
            {
                "task": r.task,
                "success": r.success,
                "elapsed_time": r.elapsed_time,
                "output_file": str(r.output_file) if r.output_file else None,
                "error": r.error,
                "extraction_comparison": r.extraction_comparison
            }
            for r in results
        ]
    }

    metadata_suffix = f"_run{run}" if run else ""
    metadata_file = output_dir / f"test_metadata{metadata_suffix}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return results, False


def print_summary(all_results: dict[str, list[TestResult]], skipped_models: list[str], run: Optional[int] = None):
    """Print a summary table of all results."""
    run_label = f" (Run {run})" if run else ""
    print("\n" + "=" * 100)
    print(f"SUMMARY{run_label}")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<25} {'Summarize':<12} {'Extract':<12} {'Similarity':<12} {'Eval':<12}")
    print("-" * 90)

    for model, results in all_results.items():
        row = f"{model:<25}"
        extract_sim = ""
        for result in results:
            if result.success:
                status = f"OK ({result.elapsed_time:.0f}s)"
            else:
                status = "FAILED"

            if result.task == "extract" and result.extraction_comparison:
                comp = result.extraction_comparison
                if comp.get("match"):
                    extract_sim = "100% MATCH"
                else:
                    extract_sim = f"{comp.get('similarity', 0):.1%}"

            if result.task == "summarize":
                row += f" {status:<12}"
            elif result.task == "extract":
                row += f" {status:<12} {extract_sim:<12}"
            elif result.task == "eval":
                row += f" {status:<12}"
        print(row)

    if skipped_models:
        print(f"\nSkipped (already completed): {', '.join(skipped_models)}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Test various Ollama models on RSP evaluation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test installed recommended models (skips already completed)
  python test_models.py

  # Force re-run all tests even if completed
  python test_models.py --force

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
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-run tests even if outputs already exist"
    )
    parser.add_argument(
        "--run",
        "-r",
        type=int,
        default=None,
        help="Run number for multiple test runs (e.g., --run 1, --run 2)"
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

    run_label = f" (Run {args.run})" if args.run else ""
    print(f"\nModels to test: {', '.join(models_to_test)}")
    print(f"Target document: {args.target}")
    print(f"Output directory: {args.output_dir}")
    if args.run:
        print(f"Run number: {args.run}")

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
    skipped_models = []
    for model in available_models:
        results, was_skipped = test_model(
            model=model,
            target_file=args.target,
            policy_path=args.policy,
            rubric_path=args.rubric,
            output_base=args.output_dir,
            verbose=args.verbose,
            skip_eval=args.skip_eval,
            force=args.force,
            run=args.run
        )
        if was_skipped:
            skipped_models.append(model)
        else:
            all_results[model] = results

    # Print summary
    print_summary(all_results, skipped_models, args.run)

    # Save overall summary
    summary_suffix = f"_run{args.run}" if args.run else ""
    summary_file = args.output_dir / f"test_summary{summary_suffix}.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "run": args.run,
        "target_file": str(args.target),
        "models_tested": list(all_results.keys()),
        "results": {
            model: [
                {
                    "task": r.task,
                    "success": r.success,
                    "elapsed_time": r.elapsed_time,
                    "extraction_comparison": r.extraction_comparison
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
