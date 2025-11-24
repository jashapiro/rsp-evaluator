import time
from pathlib import Path
from typing import Any, Dict, Generator, Union

from langchain_core.prompts import PromptTemplate

from src.llm import setup_ollama_llm
from src.loader import load_document


def summarize_research_plan(
    target_path: Path,
    model_name: str = "llama3.2",
    verbose: bool = False,
) -> str:
    """
    Summarize the research plan from a target document.
    """
    target_docs = load_document(target_path, verbose=verbose)
    target_text = "\n\n".join([d.page_content for d in target_docs])

    prompts_dir = Path(__file__).parent.parent / "prompts"
    summary_prompt_template = (prompts_dir / "summarize_research_plan.txt").read_text()

    llm = setup_ollama_llm(model_name, verbose=verbose)

    summary_prompt = PromptTemplate(
        input_variables=["target_document"],
        template=summary_prompt_template,
    )
    summary_chain = summary_prompt | llm
    research_plan_summary = summary_chain.invoke({"target_document": target_text})
    return research_plan_summary


def extract_sharing_plan(
    target_path: Path,
    model_name: str = "llama3.2",
    verbose: bool = False,
) -> str:
    """
    Extract the resource sharing plan from a target document.
    """
    target_docs = load_document(target_path, verbose=verbose)
    target_text = "\n\n".join([d.page_content for d in target_docs])

    prompts_dir = Path(__file__).parent.parent / "prompts"
    extraction_prompt_template = (prompts_dir / "extract_sharing_plan.txt").read_text()

    llm = setup_ollama_llm(model_name, verbose=verbose)

    extraction_prompt = PromptTemplate(
        input_variables=["target_document"],
        template=extraction_prompt_template,
    )
    extraction_chain = extraction_prompt | llm
    resource_sharing_plan = extraction_chain.invoke({"target_document": target_text})
    return resource_sharing_plan


def evaluate_document(
    target_path: Path,
    policy_path: Path,
    rubric_path: Path,
    model_name: str = "llama3.2",
    verbose: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    Evaluate a target document against a policy and rubric.
    Yields status updates and finally the result.
    """

    # Helper to yield status
    def yield_status(stage: str, message: str, start_time: float) -> Dict[str, Any]:
        elapsed = time.time() - start_time
        return {
            "type": "status",
            "stage": stage,
            "message": message,
            "elapsed": elapsed,
        }

    total_start_time = time.time()

    # Load documents
    yield {
        "type": "status",
        "stage": "loading",
        "message": "Loading documents...",
        "elapsed": 0.0,
    }

    step_start = time.time()
    policy_docs = load_document(policy_path, verbose=verbose)
    rubric_docs = load_document(rubric_path, verbose=verbose)

    # Extract text content
    policy_text = "\n\n".join([d.page_content for d in policy_docs])
    rubric_text = "\n\n".join([d.page_content for d in rubric_docs])

    yield yield_status("loading", "Documents loaded", step_start)

    # Load prompt templates
    prompts_dir = Path(__file__).parent.parent / "prompts"
    eval_prompt_template = (prompts_dir / "evaluation_prompt.txt").read_text()

    llm = setup_ollama_llm(model_name, verbose=verbose)

    # Step 1: Summarize Research Plan
    step_start = time.time()
    yield {
        "type": "status",
        "stage": "summarizing",
        "message": "Summarizing Research Plan...",
        "elapsed": 0.0,
    }

    if verbose:
        print("Summarizing Research Plan...")

    research_plan_summary = summarize_research_plan(target_path, model_name, verbose)

    yield yield_status("summarizing", "Research Plan summarized", step_start)

    # Step 2: Extract Resource Sharing Plan
    step_start = time.time()
    yield {
        "type": "status",
        "stage": "extracting",
        "message": "Extracting Resource Sharing Plan...",
        "elapsed": 0.0,
    }

    if verbose:
        print("Extracting Resource Sharing Plan...")

    resource_sharing_plan = extract_sharing_plan(target_path, model_name, verbose)

    yield yield_status("extracting", "Resource Sharing Plan extracted", step_start)

    # Step 3: Evaluate
    step_start = time.time()
    yield {
        "type": "status",
        "stage": "evaluating",
        "message": "Evaluating against Rubric...",
        "elapsed": 0.0,
    }

    if verbose:
        print("Evaluating against Rubric...")

    eval_prompt = PromptTemplate(
        input_variables=[
            "policy",
            "rubric",
            "research_plan_summary",
            "resource_sharing_plan",
        ],
        template=eval_prompt_template,
    )

    eval_chain = eval_prompt | llm

    if verbose:
        print("\nStarting evaluation...")
        print("=" * 50)

    result = eval_chain.invoke(
        {
            "policy": policy_text,
            "rubric": rubric_text,
            "research_plan_summary": research_plan_summary,
            "resource_sharing_plan": resource_sharing_plan,
        }
    )

    yield yield_status("evaluating", "Evaluation complete", step_start)

    yield {
        "type": "result",
        "content": result,
        "total_elapsed": time.time() - total_start_time,
    }
