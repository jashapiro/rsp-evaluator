import platform
import sys
from pathlib import Path
from typing import Optional

DEFAULT_POLICY_PATH = Path("reference/alsf_resource_sharing_policy.pdf")
DEFAULT_RUBRIC_PATH = Path("reference/RSP-Rubric-4_11_23.docx")
DEFAULT_MODEL_NAME = "qwen3.6:35b"
DEFAULT_MLX_MODEL_NAME = "mlx-community/Qwen3.6-35B-A3B-4bit"


def get_default_backend() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "ollama"


DEFAULT_BACKEND = get_default_backend()


def resolve_model(model_name: Optional[str], backend: str) -> str:
    """Return model_name if provided, otherwise the default for the given backend."""
    if model_name is not None:
        return model_name
    if backend == "mlx":
        return DEFAULT_MLX_MODEL_NAME
    return DEFAULT_MODEL_NAME
