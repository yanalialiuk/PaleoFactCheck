from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_processing.data_layer import get_chroma_collection
from model_loader import get_llama_model
from langchain.llms import LlamaCpp

class RAGPipeline:
    def __init__(self):
        self.collection = get_chroma_collection()
        self.model = get_llama_model()
        self.llm = LlamaCpp(model_path=self.model)

    def run(self, query, top_k: int = 3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        context = " ".join(results["documents"][0])
        prompt = f"Утверждение: {query}\nКонтекст: {context}\nОтвет: []"
        answer = self.llm(prompt)
        return answer


rag = RAGPipeline()
app = FastAPI(title="PaleoFactCheck API", version="0.1.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Claim to verify against the knowledge base")
    top_k: int = 500


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    answer = rag.run(query, request.top_k)
    return QueryResponse(answer=answer)

