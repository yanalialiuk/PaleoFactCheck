import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fact_check_parsing import normalize_claim, parse_verdict
from model_loader import get_llama_model

DEFAULT_TOP_K = 3

_llm: Any = None
_llm_lock = threading.Lock()
_llm_loader: Optional[Callable[[], Any]] = None


def configure_llm_loader(loader: Optional[Callable[[], Any]]) -> None:
    """Override the LLM loader (used by tests to avoid loading a real model)."""
    global _llm_loader
    _llm_loader = loader


def get_llm() -> Any:
    """Load the GGUF model on first fact-check to keep imports and /health cheap.

    Thread-safe: concurrent first callers will only build one model instance.
    """
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is None:
            if _llm_loader is not None:
                _llm = _llm_loader()
            else:
                from llama_cpp import Llama  # lazy import: avoids hard dep at import time

                _llm = Llama(
                    model_path=get_llama_model(),
                    n_ctx=4096,
                    n_threads=8,
                )
    return _llm


def reset_llm_cache() -> None:
    """Clear the cached LLM (used by tests)."""
    global _llm
    with _llm_lock:
        _llm = None


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

    from data_processing.retrieval import retrieve_claim_context  # lazy: keeps /health and tests cheap

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
