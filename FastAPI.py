from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fact_check import DEFAULT_TOP_K, run_fact_check

app = FastAPI(title="PaleoFactCheck API")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Paleontology claim to verify")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50, description="Number of Chroma chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    verdict: str
    sources: list[str]
    claim: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        result = run_fact_check(request.query, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not result.claim:
        raise HTTPException(status_code=400, detail="Claim must not be empty after normalization.")

    return QueryResponse(
        answer=result.as_text(),
        verdict=result.verdict,
        sources=result.sources,
        claim=result.claim,
    )
