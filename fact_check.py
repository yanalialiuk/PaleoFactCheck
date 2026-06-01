from data_processing.data_layer import get_chroma_collection
from model_loader import get_llama_model
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import torch

DEFAULT_TOP_K = 3

embedder = SentenceTransformer("all-MiniLM-L6-v2")

llm = Llama(
    model_path=get_llama_model(),
    n_ctx=4096,
    n_threads=8,
)

collection = get_chroma_collection()


def fact_check(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Check a claim against the Chroma knowledge base and return a short LLM verdict."""
    query = (query or "").strip()
    if not query:
        return "Insufficient information."

    # embed the query
    with torch.no_grad():
        query_emb = embedder.encode(query).tolist()

    # top-k similar documents
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        context = "No relevant information available."
    else:
        context = "\n".join(docs)

    
 
    prompt = f"""
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

    # generate answer
    response = llm(
        prompt,
        max_tokens=200,
        stop=["</s>", "User:"]
    )

    answer = response["choices"][0]["text"].strip()
    if not answer:
        answer = "Unable to verify the claim."

    return answer
