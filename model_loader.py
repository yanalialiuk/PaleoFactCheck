from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_DIR = Path("models")
GGUF_FILENAME = "llama-2-7b-chat.Q4_K_M.gguf"
HF_REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"


def get_llama_model() -> str:
    """Return local path to the Llama-2 chat GGUF, downloading once if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached = MODEL_DIR / GGUF_FILENAME
    if cached.is_file():
        return str(cached)

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=GGUF_FILENAME,
        local_dir=str(MODEL_DIR),
    )
    print("Модель сохранена в:", model_path)
    return model_path


if __name__ == "__main__":
    get_llama_model()
