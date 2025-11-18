# RSP Evaluator

A PDF analysis tool that uses LangChain and Ollama to generate comprehensive summaries of PDF documents.

## Features

- Extract and analyze text from PDF files
- Generate AI-powered summaries using local LLMs via Ollama
- Configurable output options (stdout or file)
- Verbose mode for detailed progress information
- Customizable prompts stored in separate files

## Prerequisites

- [Pixi](https://pixi.sh/) package manager
- [Ollama](https://ollama.ai/) (installed via pixi)

## Setup

1. **Install dependencies**:
   ```bash
   pixi install
   ```

2. **Set up and start Ollama with the default model**:
   ```bash
   pixi run llm-setup
   ```

   This will start the Ollama server in the background and download the `llama3.2` model. If you want to use a different model, you can pull it manually:
   ```bash
   ollama pull <model-name>
   ```

   **Note**: The Ollama server needs to be running for the PDF analyzer to work. If you restart your computer, you may need to run `ollama serve &` to start it again.

## Usage

### Basic Usage

Analyze a PDF and print the summary to stdout:

```bash
pixi run python langchain_test.py /path/to/document.pdf
```

### Options

- **`--model` / `-m`**: Specify the Ollama model to use (default: `llama3.2`)
  ```bash
  pixi run python langchain_test.py document.pdf -m llama3.1
  ```

- **`--verbose` / `-v`**: Enable verbose output to see progress messages
  ```bash
  pixi run python langchain_test.py document.pdf -v
  ```

- **`--output` / `-o`**: Write the summary to a file instead of stdout
  ```bash
  pixi run python langchain_test.py document.pdf -o summary.txt
  ```

### Combined Options

```bash
# Verbose mode with file output
pixi run python langchain_test.py document.pdf -v -o summary.txt

# Use a different model and save to file
pixi run python langchain_test.py document.pdf -m llama3.1 -o output.txt
```

## Project Structure

```
rsp-evaluator/
├── langchain_test.py          # Main PDF analyzer script
├── prompts/
│   └── pdf_summary.txt        # Customizable prompt template
├── tests/
│   └── test_langchain_pdf.py  # Test suite
├── pixi.toml                  # Dependency and task configuration
└── README.md                  # This file
```

## How It Works

1. **Load PDF**: Extracts text from all pages using PyMuPDF
2. **Split Text**: Breaks the document into manageable chunks (2000 chars each with 200 char overlap)
3. **Combine Content**: Joins all chunks for processing
4. **Generate Summary**: Uses Ollama with LangChain to analyze the content and generate a comprehensive summary

