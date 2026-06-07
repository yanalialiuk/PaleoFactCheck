from dataclasses import dataclass

from data_processing.retrieval import RetrievalResult, retrieve_claim_context
from fact_check_parsing import normalize_claim, parse_verdict
from model_loader import get_llama_model
from llama_cpp import Llama

DEFAULT_TOP_K = 3

_llm: Llama | None = None


def get_llm() -> Llama:
    """Load the GGUF model on first fact-check to keep imports and /health cheap."""
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=get_llama_model(),
            n_ctx=4096,
            n_threads=8,
        )
    return _llm


@dataclass(frozen=True)
class FactCheckResult:
    claim: str
    verdict: str
    raw_answer: str
    sources: list[str]
    context_excerpt: str

    def as_text(self) -> str:
        if self.sources:
            source_line = f"Sources: {', '.join(self.sources)}"
            return f"{self.verdict}\n{source_line}"
        return self.verdict


def build_fact_check_prompt(query: str, context: str) -> str:
    return f"""
            [INST] <<SYS>>
            You are a scientific fact-checking assistant.
            Verify only this claim: "{query}"
            Reply briefly with one of: True, False, or Insufficient information.
            Do not add other claims or extra commentary.
            <</SYS>>

            Context:
            {context}

            [/INST]
            """


def run_fact_check(query: str, top_k: int = DEFAULT_TOP_K) -> FactCheckResult:
    claim = normalize_claim(query)
    if not claim:
        return FactCheckResult(
            claim="",
            verdict="Insufficient information",
            raw_answer="",
            sources=[],
            context_excerpt="",
        )

    retrieval = retrieve_claim_context(claim, top_k=top_k)
    prompt = build_fact_check_prompt(claim, retrieval.context)
    response = get_llm()(
        prompt,
        max_tokens=200,
        stop=["</s>", "User:"],
    )

    raw_answer = response["choices"][0]["text"].strip()
    if not raw_answer:
        raw_answer = "Unable to verify the claim."

    verdict = parse_verdict(raw_answer)
    excerpt = retrieval.context[:500]

    return FactCheckResult(
        claim=claim,
        verdict=verdict,
        raw_answer=raw_answer,
        sources=retrieval.sources,
        context_excerpt=excerpt,
    )


def fact_check(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Check a claim against the Chroma knowledge base and return a short LLM verdict."""
    return run_fact_check(query, top_k=top_k).as_text()
