#!/usr/bin/env python3
"""
Tests for langchain-test.py PDF processing functions
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_test import (
    load_pdf,
    split_text,
    create_summary_chain,
    setup_ollama_llm,
    analyze_pdf,
)


class TestLoadPdf:
    """Tests for load_pdf function"""

    def test_load_pdf_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file"""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            load_pdf("/nonexistent/file.pdf")

    @patch("langchain_test.PyMuPDFLoader")
    def test_load_pdf_success(self, mock_loader_class):
        """Test successful PDF loading"""
        # Create a temporary PDF file
        test_pdf = Path("/tmp/test.pdf")
        test_pdf.touch()

        try:
            # Mock the loader
            mock_loader = Mock()
            mock_loader.load.return_value = [Mock(page_content="Page 1"), Mock(page_content="Page 2")]
            mock_loader_class.return_value = mock_loader

            # Call function
            result = load_pdf(str(test_pdf))

            # Assertions
            assert len(result) == 2
            mock_loader_class.assert_called_once_with(str(test_pdf))
            mock_loader.load.assert_called_once()
        finally:
            test_pdf.unlink()


class TestSplitText:
    """Tests for split_text function"""

    def test_split_text_with_documents(self):
        """Test text splitting with mock documents"""
        # Create mock documents with proper metadata
        mock_docs = [
            Mock(page_content="a" * 3000, metadata={}),  # Long text to ensure splitting
            Mock(page_content="b" * 3000, metadata={}),
        ]

        # Call function
        result = split_text(mock_docs, chunk_size=1000, chunk_overlap=100)

        # Assertions
        assert isinstance(result, list)
        assert len(result) > 2  # Should be split into multiple chunks

    def test_split_text_custom_parameters(self):
        """Test text splitting with custom chunk size and overlap"""
        mock_docs = [Mock(page_content="x" * 5000, metadata={})]

        result = split_text(mock_docs, chunk_size=500, chunk_overlap=50)

        assert isinstance(result, list)
        assert len(result) >= 10  # 5000 chars with 500 chunk size should produce many chunks


class TestSetupOllamaLLM:
    """Tests for setup_ollama_llm function"""

    @patch("langchain_test.OllamaLLM")
    def test_setup_ollama_llm_success(self, mock_ollama_class):
        """Test successful Ollama LLM setup"""
        # Mock the LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Hello response"
        mock_ollama_class.return_value = mock_llm

        # Call function
        result = setup_ollama_llm("llama3.2")

        # Assertions
        assert result == mock_llm
        mock_ollama_class.assert_called_once_with(
            model="llama3.2",
            temperature=0.1,
        )
        mock_llm.invoke.assert_called_once_with("Hello")

    @patch("langchain_test.OllamaLLM")
    @patch("langchain_test.sys.exit")
    def test_setup_ollama_llm_connection_error(self, mock_exit, mock_ollama_class):
        """Test Ollama LLM setup with connection error"""
        # Mock connection error
        mock_ollama_class.side_effect = Exception("Connection refused")

        # Call function
        setup_ollama_llm("llama3.2")

        # Assertions
        mock_exit.assert_called_once_with(1)


class TestCreateSummaryChain:
    """Tests for create_summary_chain function"""

    @patch("langchain_test.Path")
    def test_create_summary_chain(self, mock_path_class):
        """Test summary chain creation"""
        # Mock the prompt file path
        mock_path = Mock()
        mock_path.read_text.return_value = "Test prompt {text}"
        mock_path_class.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path

        # Mock LLM
        mock_llm = Mock()

        # Call function
        result = create_summary_chain(mock_llm)

        # Assertions
        assert result is not None
        mock_path.read_text.assert_called_once()

    def test_create_summary_chain_with_real_prompt(self):
        """Test summary chain creation with real prompt file"""
        # Check if prompt file exists
        prompt_path = Path(__file__).parent.parent / "prompts" / "pdf_summary.txt"

        if not prompt_path.exists():
            pytest.skip("Prompt file not found")

        # Mock LLM
        mock_llm = Mock()

        # Call function
        result = create_summary_chain(mock_llm)

        # Assertions
        assert result is not None


class TestAnalyzePdf:
    """Tests for analyze_pdf function"""

    @patch("langchain_test.setup_ollama_llm")
    @patch("langchain_test.split_text")
    @patch("langchain_test.load_pdf")
    def test_analyze_pdf_small_document(self, mock_load, mock_split, mock_setup_llm):
        """Test analyze_pdf with a small document (3 chunks)"""
        # Mock the PDF loading
        mock_load.return_value = [Mock(page_content="test content", metadata={})]

        # Mock text splitting - small PDF with 3 chunks
        mock_chunks = [
            Mock(page_content="chunk 1", metadata={}),
            Mock(page_content="chunk 2", metadata={}),
            Mock(page_content="chunk 3", metadata={}),
        ]
        mock_split.return_value = mock_chunks

        # Mock LLM and chain
        mock_llm = Mock()
        mock_setup_llm.return_value = mock_llm

        with patch("langchain_test.create_summary_chain") as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = "This is a summary"
            mock_create_chain.return_value = mock_chain

            # Create temporary test PDF
            test_pdf = Path("/tmp/test_small.pdf")
            test_pdf.touch()

            try:
                result = analyze_pdf(str(test_pdf), "llama3.2")

                # Assertions
                assert result == "This is a summary"
                mock_load.assert_called_once_with(str(test_pdf), verbose=False)
                mock_split.assert_called_once()
                mock_setup_llm.assert_called_once_with("llama3.2", verbose=False)
                mock_create_chain.assert_called_once_with(mock_llm)

                # Verify all chunks were used (small PDF)
                invoke_call = mock_chain.invoke.call_args
                combined_text = invoke_call[0][0]["text"]
                assert "chunk 1" in combined_text
                assert "chunk 2" in combined_text
                assert "chunk 3" in combined_text
            finally:
                test_pdf.unlink()

    @patch("langchain_test.setup_ollama_llm")
    @patch("langchain_test.split_text")
    @patch("langchain_test.load_pdf")
    def test_analyze_pdf_large_document(self, mock_load, mock_split, mock_setup_llm):
        """Test analyze_pdf with a large document (10 chunks)"""
        # Mock the PDF loading
        mock_load.return_value = [Mock(page_content="test content", metadata={})]

        # Mock text splitting - large PDF with 10 chunks
        mock_chunks = [
            Mock(page_content=f"chunk {i}", metadata={}) for i in range(10)
        ]
        mock_split.return_value = mock_chunks

        # Mock LLM and chain
        mock_llm = Mock()
        mock_setup_llm.return_value = mock_llm

        with patch("langchain_test.create_summary_chain") as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = "This is a summary of large PDF"
            mock_create_chain.return_value = mock_chain

            # Create temporary test PDF
            test_pdf = Path("/tmp/test_large.pdf")
            test_pdf.touch()

            try:
                result = analyze_pdf(str(test_pdf), "llama3.2")

                # Assertions
                assert result == "This is a summary of large PDF"

                # Verify all chunks were used (no limit)
                invoke_call = mock_chain.invoke.call_args
                combined_text = invoke_call[0][0]["text"]
                assert "chunk 0" in combined_text
                assert "chunk 4" in combined_text
                assert "chunk 9" in combined_text  # All chunks should be included
            finally:
                test_pdf.unlink()


class TestMainFunction:
    """Tests for the main CLI function"""

    def test_main_path_handling(self):
        """Test that main function handles path conversion correctly"""
        from langchain_test import Path as LangChainPath

        # Test absolute path remains absolute
        abs_path = "/tmp/test.pdf"
        pdf_path_obj = LangChainPath(abs_path)
        assert pdf_path_obj.is_absolute()
        result_path = str(pdf_path_obj)
        assert result_path == abs_path

        # Test relative path becomes absolute
        rel_path = "test.pdf"
        pdf_path_obj = LangChainPath(rel_path)
        assert not pdf_path_obj.is_absolute()
        result_path = str(LangChainPath.cwd() / pdf_path_obj)
        assert LangChainPath(result_path).is_absolute()

    def test_path_resolution_relative_to_absolute(self):
        """Test that relative paths are converted to absolute"""
        from langchain_test import Path as LangChainPath

        # Test the logic used in main()
        pdf_path = "test.pdf"
        pdf_path_obj = LangChainPath(pdf_path)

        if not pdf_path_obj.is_absolute():
            absolute_path = str(LangChainPath.cwd() / pdf_path_obj)
            assert LangChainPath(absolute_path).is_absolute()


class TestPathUsage:
    """Tests to verify pathlib usage instead of os.path"""

    def test_imports(self):
        """Verify that the module uses pathlib instead of os.path"""
        import langchain_test as module

        # Check that pathlib.Path is imported
        assert hasattr(module, "Path")

        # Check that os is not imported (we removed it)
        assert not hasattr(module, "os")

    def test_path_operations_use_pathlib(self):
        """Test that Path operations work correctly"""
        from langchain_test import Path as LangChainPath

        # Test basic Path operations
        test_path = LangChainPath("/tmp/test.pdf")
        assert test_path.name == "test.pdf"
        assert test_path.suffix == ".pdf"
        assert test_path.parent == LangChainPath("/tmp")
