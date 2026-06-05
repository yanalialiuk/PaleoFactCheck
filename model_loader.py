from pathlib import Path

from huggingface_hub import hf_hub_download
import os


def get_llama_model() -> str:
    custom = os.environ.get("LLAMA_MODEL_PATH")
    if custom:
        model_path = Path(custom).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"LLAMA_MODEL_PATH does not point to an existing GGUF file: {model_path}"
            )
        return str(model_path)

    model_dir = Path("models")
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
