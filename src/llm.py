import subprocess
import sys

from langchain_core.language_models.llms import BaseLLM
from langchain_ollama import OllamaLLM


def setup_llm(model_name: str, backend: str = "ollama", verbose: bool = False) -> BaseLLM:
    """Dispatcher to set up the appropriate LLM backend."""
    if backend == "ollama":
        return setup_ollama_llm(model_name, verbose)
    elif backend == "mlx":
        return setup_mlx_llm(model_name, verbose)
    else:
        print(f"Unknown backend: {backend}. Use 'ollama' or 'mlx'.")
        sys.exit(1)


def setup_ollama_llm(model_name: str = "llama3.2", verbose: bool = False) -> OllamaLLM:
    """Initialize Ollama LLM, pulling the model automatically if not already downloaded."""
    if verbose:
        print(f"Setting up Ollama with model: {model_name}")

    llm = OllamaLLM(model=model_name, temperature=0.1)

    try:
        test_response = llm.invoke("Hello")
        if verbose:
            print(f"Ollama connection successful. Test response: {test_response[:50]}...")
        return llm

    except Exception as e:
        if "not found" in str(e).lower():
            print(f"Model '{model_name}' not found locally. Pulling from Ollama (this may take a while)...")
            result = subprocess.run(["ollama", "pull", model_name])
            if result.returncode != 0:
                print(f"Failed to pull model '{model_name}'. Check that Ollama is running and the model name is correct.")
                sys.exit(1)
            print(f"Model '{model_name}' ready.")
            return llm
        else:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running.")
            sys.exit(1)


def setup_mlx_llm(model_name: str, verbose: bool = False) -> BaseLLM:
    """Initialize MLX LLM (Apple Silicon only). Downloads model from HuggingFace Hub on first use."""
    try:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from langchain_community.llms.mlx_pipeline import MLXPipeline
    except ImportError:
        print("mlx-lm is not installed. It should be installed automatically on osx-arm64 via pixi.")
        sys.exit(1)

    from huggingface_hub import snapshot_download

    # Download with progress if not already cached
    try:
        snapshot_download(model_name, local_files_only=True)
        if verbose:
            print(f"Loading MLX model '{model_name}' from cache...")
    except Exception:
        print(f"Downloading MLX model '{model_name}' from HuggingFace Hub (this may take a while)...")
        snapshot_download(model_name)
        print(f"Model '{model_name}' ready.")

    return MLXPipeline.from_model_id(
        model_name,
        pipeline_kwargs={"temp": 0.1, "max_tokens": 4096},
    )
