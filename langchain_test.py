#!/usr/bin/env python3
"""
LangChain PDF Reader with Ollama
Reads a PDF file and answers the question "What is this PDF about?"
"""

import sys
from pathlib import Path
from typing import List

import typer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

app = typer.Typer()


def load_pdf(pdf_path: str, verbose: bool = False) -> List:
    """Load and extract text from a PDF file."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if verbose:
        print(f"Loading PDF: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    if verbose:
        print(f"Loaded {len(documents)} pages from PDF")
    return documents


def split_text(documents: List, chunk_size: int = 2000, chunk_overlap: int = 200, verbose: bool = False) -> List:
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    texts = text_splitter.split_documents(documents)
    if verbose:
        print(f"Split into {len(texts)} chunks")
    return texts


def setup_ollama_llm(model_name: str = "llama3.2", verbose: bool = False) -> OllamaLLM:
    """Initialize Ollama LLM."""
    if verbose:
        print(f"Setting up Ollama with model: {model_name}")

    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=0.1,  # Low temperature for more focused responses
        )

        # Test the connection
        test_response = llm.invoke("Hello")
        if verbose:
            print(f"Ollama connection successful. Test response: {test_response[:50]}...")
        return llm

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running and the model is available.")
        print(f"Try running: ollama pull {model_name}")
        sys.exit(1)


def create_summary_chain(llm: OllamaLLM) -> Runnable:
    """Create a chain for summarizing PDF content."""

    # Load prompt template from file
    prompt_path = Path(__file__).parent / "prompts" / "pdf_summary.txt"
    prompt_template = prompt_path.read_text()

    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_template
    )

    return prompt | llm


def analyze_pdf(pdf_path: str, model_name: str = "llama3.2", verbose: bool = False) -> str:
    """Main function to analyze a PDF and answer what it's about."""

    # Load and process PDF
    documents = load_pdf(pdf_path, verbose=verbose)
    text_chunks = split_text(documents, verbose=verbose)

    # Combine all text chunks
    combined_text = "\n\n".join([chunk.page_content for chunk in text_chunks])
    if verbose:
        print(f"Processing {len(text_chunks)} chunks from the PDF")

    # Set up LLM and create summary
    llm = setup_ollama_llm(model_name, verbose=verbose)
    summary_chain = create_summary_chain(llm)

    if verbose:
        print("\nAnalyzing PDF content...")
        print("=" * 50)

    # Generate summary
    result = summary_chain.invoke({"text": combined_text})

    return result


@app.command()
def main(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to analyze"),
    model_name: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model to use for analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output_file: str = typer.Option(None, "--output", "-o", help="Output file path (optional, prints to stdout if not specified)")
):
    """
    Analyze a PDF file and answer "What is this PDF about?"

    Uses LangChain with Ollama to provide a comprehensive summary of the PDF content.
    """

    # Typer should prevent None, but double-check for safety
    if not pdf_path:
        raise typer.BadParameter("PDF path is required")

    # Check if PDF path is relative, make it absolute
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.is_absolute():
        pdf_path = str(Path.cwd() / pdf_path_obj)

    try:
        summary = analyze_pdf(pdf_path, model_name, verbose=verbose)

        # Format output
        output = f"\n{'=' * 70}\nWHAT IS THIS PDF ABOUT?\n{'=' * 70}\n{summary}\n{'=' * 70}\n"

        # Write to file or print to stdout
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(summary)
            if verbose:
                print(f"Summary written to: {output_file}")
        else:
            print(output)

    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()