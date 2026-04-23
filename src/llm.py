import subprocess
import sys
from typing import Any, List, Optional

from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_ollama import OllamaLLM
from pydantic import PrivateAttr


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


class MLXLLM(BaseLLM):
    """LangChain-compatible LLM wrapper for mlx-lm (Apple Silicon only)."""

    model_name: str
    temperature: float = 0.1
    max_tokens: int = 16384

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        from mlx_lm import load
        self._model, self._tokenizer = load(self.model_name)

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        import re
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=self.temperature)
        generations = []
        for prompt in prompts:
            # Instruction-tuned models need the chat template applied
            if hasattr(self._tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                formatted = prompt
            text = generate(
                self._model,
                self._tokenizer,
                prompt=formatted,
                max_tokens=self.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            # Strip <think>...</think> reasoning blocks (Qwen3 and similar models)
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "mlx"


def setup_mlx_llm(model_name: str, verbose: bool = False) -> MLXLLM:
    """Initialize MLX LLM (Apple Silicon only). Downloads model from HuggingFace Hub on first use."""
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("mlx-lm is not installed. It should be installed automatically on osx-arm64 via pixi.")
        sys.exit(1)

    from huggingface_hub import disable_progress_bars, enable_progress_bars, snapshot_download

    # Check cache silently; only show progress if a real download is needed
    disable_progress_bars()
    cached = True
    try:
        snapshot_download(model_name, local_files_only=True)
    except Exception:
        cached = False
    finally:
        enable_progress_bars()

    if not cached:
        print(f"Downloading MLX model '{model_name}' from HuggingFace Hub (this may take a while)...")
        snapshot_download(model_name)
        print(f"Model '{model_name}' ready.")
    elif verbose:
        print(f"Loading MLX model '{model_name}' from cache...")

    return MLXLLM(model_name=model_name)
