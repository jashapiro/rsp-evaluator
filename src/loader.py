from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path: str | Path, verbose: bool = False) -> List[Document]:
    """Load and extract text from a PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if verbose:
        print(f"Loading PDF: {path}")
    loader = PyMuPDFLoader(str(path))
    documents = loader.load()

    if verbose:
        print(f"Loaded {len(documents)} pages from PDF")
    return documents


def load_docx(docx_path: str | Path, verbose: bool = False) -> List[Document]:
    """Load and extract text from a Word document (.docx)."""
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"Word document not found: {path}")

    if verbose:
        print(f"Loading Word document: {path}")
    loader = Docx2txtLoader(str(path))
    documents = loader.load()

    if verbose:
        print(f"Loaded {len(documents)} document(s) from Word file")
    return documents


def load_document(file_path: str | Path, verbose: bool = False) -> List[Document]:
    """Load a document (PDF or Word) based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path, verbose=verbose)
    elif suffix in [".docx", ".doc"]:
        return load_docx(path, verbose=verbose)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported types: .pdf, .docx, .doc"
        )


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
