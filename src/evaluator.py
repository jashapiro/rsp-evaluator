import time
from pathlib import Path
from typing import Any, Dict, Generator

from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate

from src.loader import load_document


def summarize_research_plan(target_path: Path, llm: BaseLLM, verbose: bool = False) -> str:
    """Summarize the research plan from a target document."""
    target_text = "\n\n".join(
        d.page_content for d in load_document(target_path, verbose=verbose)
    )
    prompts_dir = Path(__file__).parent.parent / "prompts"
    template = (prompts_dir / "summarize_research_plan.txt").read_text()
    return (PromptTemplate(input_variables=["target_document"], template=template) | llm).invoke(
        {"target_document": target_text}
    )


def extract_sharing_plan(target_path: Path, llm: BaseLLM, verbose: bool = False) -> str:
    """Extract the resource sharing plan from a target document."""
    target_text = "\n\n".join(
        d.page_content for d in load_document(target_path, verbose=verbose)
    )
    prompts_dir = Path(__file__).parent.parent / "prompts"
    template = (prompts_dir / "extract_sharing_plan.txt").read_text()
    return (PromptTemplate(input_variables=["target_document"], template=template) | llm).invoke(
        {"target_document": target_text}
    )


def evaluate_document(
    target_path: Path,
    policy_path: Path,
    rubric_path: Path,
    llm: BaseLLM,
    verbose: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    Evaluate a target document against a policy and rubric.
    Yields status updates and finally the result.
    """

    def yield_status(stage: str, message: str, start_time: float) -> Dict[str, Any]:
        return {"type": "status", "stage": stage, "message": message, "elapsed": time.time() - start_time}

    total_start_time = time.time()

    yield {"type": "status", "stage": "loading", "message": "Loading documents...", "elapsed": 0.0}

    step_start = time.time()
    policy_text = "\n\n".join(d.page_content for d in load_document(policy_path, verbose=verbose))
    rubric_text = "\n\n".join(d.page_content for d in load_document(rubric_path, verbose=verbose))
    yield yield_status("loading", "Documents loaded", step_start)

    prompts_dir = Path(__file__).parent.parent / "prompts"
    eval_template = (prompts_dir / "evaluation_prompt.txt").read_text()

    # Step 1: Summarize Research Plan
    step_start = time.time()
    yield {"type": "status", "stage": "summarizing", "message": "Summarizing Research Plan...", "elapsed": 0.0}
    if verbose:
        print("Summarizing Research Plan...")
    research_plan_summary = summarize_research_plan(target_path, llm, verbose)
    yield yield_status("summarizing", "Research Plan summarized", step_start)

    # Step 2: Extract Resource Sharing Plan
    step_start = time.time()
    yield {"type": "status", "stage": "extracting", "message": "Extracting Resource Sharing Plan...", "elapsed": 0.0}
    if verbose:
        print("Extracting Resource Sharing Plan...")
    resource_sharing_plan = extract_sharing_plan(target_path, llm, verbose)
    yield yield_status("extracting", "Resource Sharing Plan extracted", step_start)

    # Step 3: Evaluate
    step_start = time.time()
    yield {"type": "status", "stage": "evaluating", "message": "Evaluating against Rubric...", "elapsed": 0.0}
    if verbose:
        print("Evaluating against Rubric...")

    result = (
        PromptTemplate(
            input_variables=["policy", "rubric", "research_plan_summary", "resource_sharing_plan"],
            template=eval_template,
        )
        | llm
    ).invoke({
        "policy": policy_text,
        "rubric": rubric_text,
        "research_plan_summary": research_plan_summary,
        "resource_sharing_plan": resource_sharing_plan,
    })

    yield yield_status("evaluating", "Evaluation complete", step_start)
    yield {"type": "result", "content": result, "total_elapsed": time.time() - total_start_time}
