#!/bin/bash
#
# Model Testing Script Runner for RSP Evaluator
#
# This script runs the model testing suite using pixi.
#
# Usage:
#   ./run_model_tests.sh                    # Test installed recommended models
#   ./run_model_tests.sh --quick            # Quick test (skip full evaluation)
#   ./run_model_tests.sh --download         # Download missing models first
#   ./run_model_tests.sh --list             # List recommended models
#   ./run_model_tests.sh --help             # Show all options
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if pixi is available
if ! command -v pixi &> /dev/null; then
    echo "Error: pixi is not installed or not in PATH"
    echo "Install pixi from: https://pixi.sh"
    exit 1
fi

# Check if Ollama is running
if ! ollama list &> /dev/null; then
    echo "Error: Ollama is not running or not installed"
    echo "Start Ollama with: ollama serve"
    exit 1
fi

# Parse arguments
ARGS=()
for arg in "$@"; do
    case $arg in
        --quick)
            ARGS+=("--skip-eval")
            ;;
        --list)
            ARGS+=("--list-models")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

echo "Running model tests..."
echo "======================"
pixi run python test_models.py "${ARGS[@]}"
