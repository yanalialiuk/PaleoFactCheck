import os
import fitz  
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer


PDF_DIR = "Data/PDFs"
CHROMA_DIR = "chroma_db"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Извлекает текст из PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

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


def build_dataset() -> None:

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("Нет файлов в папке 'PDFs'")
        return

    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        print(f"Обработка {pdf_file}...")

        text = extract_text_from_pdf(path)
        chunks = split_text(text)
        save_to_chroma(chunks, pdf_file)

    


def get_chroma_collection():
    client = chromadb.PersistentClient(CHROMA_DIR)
    return client.get_or_create_collection(name="paleo_facts")


