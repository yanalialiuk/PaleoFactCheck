# PaleoFactCheck

Local RAG stack for paleontology fact-checking: Chroma retrieval + Llama-2 GGUF via `llama.cpp`.

## Quick start

```bash
pip install -r requirements.txt
python model_loader.py
python -c "from data_processing.data_builder import build_dataset; build_dataset()"
python main.py "Тираннозавр был хищником."
uvicorn FastAPI:app --reload
```

## API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Анкилозавр был травоядным."}'
```
