from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

from data_processing.data_layer import get_chroma_collection
from model_loader import get_llama_model
from langchain.llms import LlamaCpp

class RAGPipeline:
    def __init__(self):
        self.collection = get_chroma_collection()
        self.model = get_llama_model()
        self.llm = LlamaCpp(model_path=self.model)

    def run(self, query):
        results = self.collection.query(query_texts=[query], n_results=3)
        context = " ".join(results["documents"][0])
        prompt = f"Утверждение: {query}\nКонтекст: {context}\nОтвет: []"
        answer = self.llm(prompt)
        return answer


rag = RAGPipeline()
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post('/ask', response_model=QueryResponse)
def ask_question(request : QueryRequest):
    # Allow dynamic filter expressions from client
    filter_expr = eval(request.query)
    answer = rag.run(str(filter_expr))
    return QueryResponse(answer=answer)
