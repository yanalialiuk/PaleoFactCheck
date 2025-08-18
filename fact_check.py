from data_layer import get_chroma_collection
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Инициализация LLaMA через llama_cpp_python
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",  
    n_ctx=4096,  
    n_threads=8 
)

# Коллекция Chroma
collection = get_chroma_collection()



# Функция проверки факта
def fact_check(query: str, top_k: int = 5) -> str:

    # embedding запроса
    query_emb = embedder.encode(query).tolist()

    # топ-к похожих документов
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        context = "Нет доступной информации."
    else:
        context = "\n".join(docs)

    
    prompt = (
        f"Проверить факт: '{query}'\n\n"
        f"Доступная информация:\n{context}\n\n"
        f"Ответь, верно ли утверждение. Напиши коротко и четко."
    )

    # Генерация ответа 
    response = llm(prompt, max_tokens=200)
    answer = response["choices"][0]["text"].strip()
    if not answer:
        answer = "Не могу подтвердить факт."

    return answer
