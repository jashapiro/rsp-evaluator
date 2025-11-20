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

    # Load prompt template
    prompt_path = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"
    prompt_template = prompt_path.read_text()

    prompt = PromptTemplate(
        input_variables=["policy", "rubric", "target_document"],
        template=prompt_template,
    )

    llm = setup_ollama_llm(model_name, verbose=verbose)

    chain = prompt | llm

    if verbose:
        print("\nStarting evaluation...")
        print("=" * 50)

    result = chain.invoke(
        {"policy": policy_text, "rubric": rubric_text, "target_document": target_text}
    )

    return result
