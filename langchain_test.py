#!/usr/bin/env python3
"""
LangChain Document Reader with Ollama
Reads a PDF or Word document and answers the question "What is this document about?"
"""

import sys
from pathlib import Path
from typing import List

import typer
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = typer.Typer()


def load_pdf(pdf_path: str, verbose: bool = False) -> List[Document]:
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


def load_docx(docx_path: str, verbose: bool = False) -> List[Document]:
    """Load and extract text from a Word document (.docx)."""
    if not Path(docx_path).exists():
        raise FileNotFoundError(f"Word document not found: {docx_path}")

    if verbose:
        print(f"Loading Word document: {docx_path}")
    loader = Docx2txtLoader(docx_path)
    documents = loader.load()

    if verbose:
        print(f"Loaded {len(documents)} document(s) from Word file")
    return documents


def load_document(file_path: str, verbose: bool = False) -> List[Document]:
    """Load a document (PDF or Word) based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        return load_pdf(file_path, verbose=verbose)
    elif suffix in ['.docx', '.doc']:
        return load_docx(file_path, verbose=verbose)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types: .pdf, .docx, .doc")



def split_text(
    documents: List[Document],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    verbose: bool = False,
) -> List[Document]:
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
            print(
                f"Ollama connection successful. Test response: {test_response[:50]}..."
            )
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

    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    return prompt | llm


def analyze_document(
    file_path: str, model_name: str = "llama3.2", verbose: bool = False
) -> str:
    """Main function to analyze a document (PDF or Word) and answer what it's about."""

    # Load and process document
    documents = load_document(file_path, verbose=verbose)
    text_chunks = split_text(documents, verbose=verbose)

    # Combine all text chunks
    combined_text = "\n\n".join([chunk.page_content for chunk in text_chunks])
    if verbose:
        print(f"Processing {len(text_chunks)} chunks from the document")

    # Set up LLM and create summary
    llm = setup_ollama_llm(model_name, verbose=verbose)
    summary_chain = create_summary_chain(llm)

    if verbose:
        print("\nAnalyzing document content...")
        print("=" * 50)

    # Generate summary
    result = summary_chain.invoke({"text": combined_text})

    return result


@app.command()
def main(
    file_path: str = typer.Argument(..., help="Path to the document file to analyze (PDF or Word)"),
    model_name: str = typer.Option(
        "llama3.2", "--model", "-m", help="Ollama model to use for analysis"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (optional, prints to stdout if not specified)",
    ),
):
    """
    Analyze a document file (PDF or Word) and answer "What is this document about?"

    Supported formats: .pdf, .docx, .doc

    Uses LangChain with Ollama to provide a comprehensive summary of the document content.
    """

    # Typer should prevent None, but double-check for safety
    if not file_path:
        raise typer.BadParameter("File path is required")

    # Check if file path is relative, make it absolute
    file_path_obj = Path(file_path)
    if not file_path_obj.is_absolute():
        file_path = str(Path.cwd() / file_path_obj)

    try:
        summary = analyze_document(file_path, model_name, verbose=verbose)

        # Format output
        output = f"\n{'=' * 70}\nWHAT IS THIS DOCUMENT ABOUT?\n{'=' * 70}\n{summary}\n{'=' * 70}\n"

        # Write to file or print to stdout
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(summary)
            if verbose:
                print(f"Summary written to: {output_file}")
        else:
            print(output)

    except Exception as e:
        print(f"Error analyzing document: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
