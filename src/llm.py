import sys
from langchain_ollama import OllamaLLM

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
