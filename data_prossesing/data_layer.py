from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer



CHROMA_DIR = "chroma_db"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Разбивает текст на чанки
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Эмбеддинги текста
def get_embedding(text: str) -> list[float]:
    return embedder.encode(text).tolist()

# Сохраняем в ChromaDB
def save_to_chroma(chunks: list[str], source_name: str) -> None:
    client = chromadb.PersistentClient(CHROMA_DIR)
    collection = client.get_or_create_collection(name="paleo_facts")

    for i, chunk in enumerate(tqdm(chunks, desc=f"Saving chunks from {source_name}")):
        embedding = get_embedding(chunk)
        collection.add(
            ids=[f"{source_name}_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": source_name}]
        )
  
def get_chroma_collection():
    client = chromadb.PersistentClient(CHROMA_DIR)
    return client.get_or_create_collection(name="paleo_facts")