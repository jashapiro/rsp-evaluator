# RSP Evaluator

A tool to evaluate a Research Sharing Plan against a policy and rubric using a local LLM.

## Prerequisites

To get started, first download and install the following two tools:

- [Pixi](https://pixi.sh/) package manager
  - This will handle software dependencies and allow some shortcuts.
  - Follow the instructions on the [Pixi website](https://pixi.sh/) to install it.
- [Ollama](https://ollama.com/)
  - This is the engine that will run the local LLM models.
  It needs to be running for the evaluator to work.
  - Follow the instructions on the [Ollama website](https://ollama.com/) to install it.
  
  
## Setup

1.  **Install dependencies**:
    ```bash
    pixi install
    ```

2.  **Set up and start Ollama with the default model**:
    ```bash
    pixi run llm-setup
    ```

    This will start the Ollama server in the background and download the `granite3.3` model. If you want to use a different model, you can pull it manually:
    ```bash
    ollama pull <model-name>
    ```

    **Note**: The Ollama server needs to be running for the tool to work. If you restart your computer, you may need to start it again. Look for the Ollama icon in the menu bar (Mac) or the system tray (Windows).

## Usage

The RSP Evaluator can be used either through its command-line interface (CLI) or through a web interface.

### Command-Line Interface (CLI)

The CLI provides commands to evaluate a document, summarize the research plan, and extract the resource sharing plan.

To run any of the CLI commands, you can use `pixi run python rspbot.py <command>`.

#### `eval`

Evaluate a document against a policy and rubric.

**Usage:**
```bash
pixi run python rspbot.py eval /path/to/document.pdf [OPTIONS]
```

**Options:**
*   `--policy` / `-p`: Path to the policy document (default: `reference/alsf_resource_sharing_policy.pdf`)
*   `--rubric` / `-r`: Path to the rubric document (default: `reference/RSP-Rubric-4_11_23.docx`)
*   `--model` / `-m`: Ollama model to use for analysis (default: `granite3.3`)
*   `--verbose` / `-v`: Enable verbose output
*   `--output` / `-o`: Output file path (optional, prints to stdout if not specified)

**Example:**
```bash
pixi run python rspbot.py eval my_grant_proposal.pdf -o evaluation_report.md
```

#### `summarize`

Summarize the research plan from a document.

**Usage:**
```bash
pixi run python rspbot.py summarize /path/to/document.pdf [OPTIONS]
```

**Options:**
*   `--model` / `-m`: Ollama model to use for analysis (default: `granite3.3`)
*   `--verbose` / `-v`: Enable verbose output
*   `--output` / `-o`: Output file path (optional, prints to stdout if not specified)

**Example:**
```bash
pixi run python rspbot.py summarize my_grant_proposal.pdf -o summary.txt
```

#### `extract`

Extract the resource sharing plan from a document.

**Usage:**
```bash
pixi run python rspbot.py extract /path/to/document.pdf [OPTIONS]
```

**Options:**
*   `--model` / `-m`: Ollama model to use for analysis (default: `granite3.3`)
*   `--verbose` / `-v`: Enable verbose output
*   `--output` / `-o`: Output file path (optional, prints to stdout if not specified)

**Example:**
```bash
pixi run python rspbot.py extract my_grant_proposal.pdf -o sharing_plan.txt
```

### Web Interface

The web interface provides a user-friendly way to upload a document and get an evaluation.

**To start the web interface, run:**
```bash
pixi run python rspbot.py serve
```
This will start a web server at `http://127.0.0.1:8000`. You can then open this URL in your browser to use the web interface.

## Project Structure

```
rsp-evaluator/
├── rspbot.py                  # Main CLI script
├── prompts/                   # Prompt templates
│   ├── evaluation_prompt.txt
│   ├── extract_sharing_plan.txt
│   └── summarize_research_plan.txt
├── src/                       # Source code
│   ├── evaluator.py
│   ├── llm.py
│   ├── loader.py
│   └── web.py
├── tests/                     # Test suite
├── pixi.toml                  # Dependency and task configuration
└── README.md                  # This file
```
