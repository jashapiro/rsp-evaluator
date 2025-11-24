from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.evaluator import evaluate_document


class TestEvaluateDocument:
    @patch("src.evaluator.load_document")
    @patch("src.evaluator.setup_ollama_llm")
    @patch("src.evaluator.PromptTemplate")
    def test_evaluate_document_flow(self, mock_prompt, mock_setup_llm, mock_load_doc):
        # Mock dependencies
        mock_load_doc.return_value = [Mock(page_content="test content")]

        mock_llm = Mock()
        mock_setup_llm.return_value = mock_llm

        mock_chain = Mock()
        mock_chain.invoke.return_value = "mock result"
        # Mock the chain creation (PromptTemplate | llm)
        # Since we can't easily mock the | operator return value directly on the class,
        # we'll assume the code works if we mock the chain execution.
        # Actually, the code does `summary_prompt | llm`.
        # We need to mock what PromptTemplate returns so that `|` works.

        mock_prompt_instance = Mock()
        mock_prompt.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = Mock(return_value=mock_chain)

        # Create dummy paths
        target = Path("target.pdf")
        policy = Path("policy.pdf")
        rubric = Path("rubric.pdf")

        # Run the generator
        generator = evaluate_document(target, policy, rubric)

        events = list(generator)

        # Verify events
        status_events = [e for e in events if e["type"] == "status"]
        result_events = [e for e in events if e["type"] == "result"]

        assert (
            len(status_events) >= 4
        )  # loading, loading done, summarizing, extracting, evaluating, evaluating done
        assert len(result_events) == 1
        assert result_events[0]["content"] == "mock result"

        # Verify stages
        stages = [e["stage"] for e in status_events]
        assert "loading" in stages
        assert "summarizing" in stages
        assert "extracting" in stages
        assert "evaluating" in stages
