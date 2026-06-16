import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is in requirements
    pass

from huggingface_hub import hf_hub_download

DEFAULT_MODEL_DIR = "models"


def get_model_dir() -> Path:
    raw = os.environ.get("LLAMA_MODEL_DIR", DEFAULT_MODEL_DIR)
    return Path(raw).expanduser().resolve()


def get_llama_model() -> str:
    custom = os.environ.get("LLAMA_MODEL_PATH")
    if custom:
        model_path = Path(custom).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"LLAMA_MODEL_PATH does not point to an existing GGUF file: {model_path}"
            )
        return str(model_path)

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
        filename="llama-2-7b-chat.Q4_K_M.gguf",
        local_dir=str(model_dir),
    )

    print("Model saved to:", model_path)
    return model_path


if __name__ == "__main__":
    get_llama_model()
