# RSP Evaluator

A tool to evaluate a Research Sharing Plan against a policy and rubric using a local LLM.

## Prerequisites

- [Pixi](https://pixi.sh/) package manager — handles all software dependencies.
- **MLX backend** (default on Apple Silicon Mac): no additional software needed.
- **Ollama backend** (default on Linux/Windows): [Ollama](https://ollama.com/) must be installed and running. The server needs to be running whenever you use the tool — look for the Ollama icon in the menu bar (Mac) or system tray (Windows). Mac users can also install Ollama and use it with `--backend ollama` if preferred.

## Setup

1. **Install dependencies**:
    ```bash
    pixi install
    ```

2. **Start Ollama** (if using the Ollama backend):
    ```bash
    ollama serve
    ```
    Models are downloaded automatically on first use — no manual `ollama pull` required.

That's it. MLX models are also downloaded automatically from HuggingFace Hub on first use and cached locally.

## Usage

The RSP Evaluator can be used either through its command-line interface (CLI) or through a web interface.

### Command-Line Interface (CLI)

Run any command with `pixi run python rspbot.py <command>`.

#### `eval`

Evaluate a document against a policy and rubric.

```bash
pixi run python rspbot.py eval /path/to/document.pdf [OPTIONS]
```

**Options:**
- `--policy` / `-p`: Path to the policy document (default: `reference/alsf_resource_sharing_policy.pdf`)
- `--rubric` / `-r`: Path to the rubric document (default: `reference/RSP-Rubric-4_11_23.docx`)
- `--model` / `-m`: Model to use (default: backend-appropriate, see below)
- `--backend` / `-b`: LLM backend — `ollama` or `mlx` (default: `mlx` on Apple Silicon, `ollama` elsewhere)
- `--verbose` / `-v`: Enable verbose output
- `--output` / `-o`: Output file path (prints to stdout if not specified)

**Example:**
```bash
pixi run python rspbot.py eval my_grant_proposal.pdf -o evaluation_report.md
```

#### `summarize`

Summarize the research plan from a document.

```bash
pixi run python rspbot.py summarize /path/to/document.pdf [OPTIONS]
```

**Options:** `--model`, `--backend`, `--verbose`, `--output` (same as above)

#### `extract`

Extract the resource sharing plan from a document.

```bash
pixi run python rspbot.py extract /path/to/document.pdf [OPTIONS]
```

**Options:** `--model`, `--backend`, `--verbose`, `--output` (same as above)

### Web Interface

```bash
pixi run python rspbot.py serve
```

Opens a web server at `http://127.0.0.1:8000`.

## Models

### MLX (Apple Silicon default)

Default model: `mlx-community/Qwen3.6-35B-A3B-4bit`

Models are specified as HuggingFace repository IDs and downloaded automatically on first use. Browse available models at [huggingface.co/mlx-community](https://huggingface.co/mlx-community).

```bash
pixi run python rspbot.py eval doc.pdf --backend mlx --model mlx-community/Qwen3-14B-4bit
```

### Ollama (Linux/Windows default)

Default model: `qwen3.6:35b`

Models are pulled from Ollama automatically on first use. Browse available models at [ollama.com/library](https://ollama.com/library).

```bash
pixi run python rspbot.py eval doc.pdf --backend ollama --model qwen2.5:7b
```

## Project Structure

```
rsp-evaluator/
├── rspbot.py                  # Main CLI script
├── prompts/                   # Prompt templates
│   ├── evaluation_prompt.txt
│   ├── extract_sharing_plan.txt
│   └── summarize_research_plan.txt
├── src/                       # Source code
│   ├── config.py
│   ├── evaluator.py
│   ├── llm.py
│   ├── loader.py
│   └── web.py
├── tests/                     # Test suite
├── pixi.toml                  # Dependency and task configuration
└── README.md                  # This file
```
