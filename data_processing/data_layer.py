from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib



CHROMA_DIR = "chroma_db"
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def text_to_id(text, source_name, index):
    s = f"{source_name}_{index}_{text}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# Разбивает текст на чанки
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Эмбеддинги текста
def get_embedding(text):
    return embedder.encode(text).tolist()


def _chunk_id(source_name, chunk_text):
    
    h = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:16]
    safe_source = source_name.replace("\\", "_").replace("/", "_")
    return f"{safe_source}:{h}"

def _existing_ids(col, ids):
    
    have = set()
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i+BATCH]
        res = col.get(ids=batch) 
        have.update(res.get("ids", []))
    return have

# Сохраняем в ChromaDB
def save_to_chroma(chunks: list[str], source_name: str) -> None:
    client = chromadb.PersistentClient(CHROMA_DIR)
    col = client.get_or_create_collection(name="paleo_facts")

    
    ids = [_chunk_id(source_name, ch) for ch in chunks]
    metas = [{"source": source_name} for _ in chunks]

    
    have = _existing_ids(col, ids)

    
    new_idx = [i for i, cid in enumerate(ids) if cid not in have]
    if not new_idx:
        print(f"⏩ Все чанки из '{source_name}' уже в базе")
        return

    
    add_ids = []
    add_docs = []
    add_metas = []
    add_embs = []

    for i in tqdm(new_idx, desc=f"Adding NEW chunks from {source_name}"):
        ch = chunks[i]
        add_ids.append(ids[i])
        add_docs.append(ch)
        add_metas.append(metas[i])
        add_embs.append(get_embedding(ch))

    
    col.add(ids=add_ids, documents=add_docs, embeddings=add_embs, metadatas=add_metas)
  
def get_chroma_collection():
    client = chromadb.PersistentClient(CHROMA_DIR)
    return client.get_or_create_collection(name="paleo_facts")