from huggingface_hub import hf_hub_download
import os

def get_llama_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = hf_hub_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
        filename="llama-2-7b-chat.Q4_K_M.gguf",
        local_dir=model_dir
    )

    print("Модель сохранена в:", model_path)
    return model_path

if __name__ == "__main__":
    get_llama_model()
