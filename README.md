# PaleoFactCheck

RAG fact-checking for paleontology claims (Chroma + Llama-2 GGUF).

## Setup

```bash
pip install -r requirements.txt
python model_loader.py
python -c "from data_processing.data_builder import build_dataset; build_dataset()"
```

## CLI

```bash
python main.py "Тираннозавр был хищником."
```

## API

```bash
uvicorn FastAPI:app --reload
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "Анкилозавр был травоядным."}'
```
