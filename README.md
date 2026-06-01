# PaleoFactCheck

Local RAG pipeline for checking paleontology claims against a Chroma document store and a Llama-2 GGUF model.

## Requirements

- Python 3.10+
- Dependencies: `sentence-transformers`, `chromadb`, `llama-cpp-python`, `fastapi`, `langchain`, `PyPDF2`, `python-docx`

## Setup

1. Clone the repository and create a virtual environment.
2. Place source documents under `Data/` (PDF, DOCX, or TXT).
3. Download the GGUF model (first run of `model_loader.py` will fetch it via Hugging Face):

```bash
python model_loader.py
```

4. Build the vector index:

```bash
python -c "from data_processing.data_builder import build_dataset; build_dataset()"
```

## Usage

**CLI fact check**

```bash
python main.py
```

**HTTP API**

```bash
uvicorn FastAPI:app --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "Тираннозавр был травоядным."}'
```

## Project layout

| Path | Role |
|------|------|
| `fact_check.py` | CLI inference against Chroma + Llama |
| `FastAPI.py` | OpenAI-style `/ask` HTTP wrapper |
| `data_processing/` | Ingestion, chunking, Chroma persistence |
| `model_loader.py` | GGUF download helper |
