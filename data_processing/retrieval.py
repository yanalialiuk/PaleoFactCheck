from dataclasses import dataclass

import torch

from .data_layer import get_chroma_collection, get_embedding


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    distance: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]

    @property
    def context(self) -> str:
        if not self.chunks:
            return "No relevant information available."
        return "\n".join(chunk.text for chunk in self.chunks)

    @property
    def sources(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for chunk in self.chunks:
            if chunk.source not in seen:
                seen.add(chunk.source)
                ordered.append(chunk.source)
        return ordered


def retrieve_claim_context(query: str, top_k: int = 3) -> RetrievalResult:
    """Embed the claim and fetch the closest Chroma chunks with source metadata."""
    normalized = (query or "").strip()
    if not normalized:
        return RetrievalResult(query="", chunks=[])

    collection = get_chroma_collection()
    with torch.no_grad():
        query_emb = get_embedding(normalized)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    chunks: list[RetrievedChunk] = []
    for index, text in enumerate(documents):
        if not text:
            continue
        meta = metadatas[index] if index < len(metadatas) else {}
        source = str(meta.get("source", "unknown"))
        distance = distances[index] if index < len(distances) else None
        chunks.append(RetrievedChunk(text=text, source=source, distance=distance))

    return RetrievalResult(query=normalized, chunks=chunks)
