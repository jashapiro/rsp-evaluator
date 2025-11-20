from pathlib import Path

from langchain_core.prompts import PromptTemplate

from src.loader import load_document


def evaluate_document(
    target_path: Path,
    policy_path: Path,
    rubric_path: Path,
    model_name: str = "llama3.2",
    verbose: bool = False,
) -> str:
    """Evaluate a target document against a policy and rubric."""

    from src.llm import setup_ollama_llm

    # Load documents
    target_docs = load_document(target_path, verbose=verbose)
    policy_docs = load_document(policy_path, verbose=verbose)
    rubric_docs = load_document(rubric_path, verbose=verbose)

    # Extract text content
    # For simplicity, we'll just join all text. For very large docs, we might need better chunking/retrieval.
    target_text = "\n\n".join([d.page_content for d in target_docs])
    policy_text = "\n\n".join([d.page_content for d in policy_docs])
    rubric_text = "\n\n".join([d.page_content for d in rubric_docs])

    # Load prompt templates
    prompts_dir = Path(__file__).parent.parent / "prompts"

    summary_prompt_template = (prompts_dir / "summarize_research_plan.txt").read_text()
    extraction_prompt_template = (prompts_dir / "extract_sharing_plan.txt").read_text()
    eval_prompt_template = (prompts_dir / "evaluation_prompt.txt").read_text()

    llm = setup_ollama_llm(model_name, verbose=verbose)

    # Step 1: Summarize Research Plan
    if verbose:
        print("Summarizing Research Plan...")

    summary_prompt = PromptTemplate(
        input_variables=["target_document"],
        template=summary_prompt_template,
    )
    summary_chain = summary_prompt | llm
    research_plan_summary = summary_chain.invoke({"target_document": target_text})

    # Step 2: Extract Resource Sharing Plan
    if verbose:
        print("Extracting Resource Sharing Plan...")

    extraction_prompt = PromptTemplate(
        input_variables=["target_document"],
        template=extraction_prompt_template,
    )
    extraction_chain = extraction_prompt | llm
    resource_sharing_plan = extraction_chain.invoke({"target_document": target_text})

    # Step 3: Evaluate
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

    return result
