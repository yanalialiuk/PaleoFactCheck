### 📋 Summary
| PR | Branch | Author | Title | Verdict |
|:--:|:-------|:------:|:------|:-------:|
| 38 | pr/v3-chore-env-example | @yl-atomic | [chore: add .env.example for local paths](https://github.com/yanalialiuk/PaleoFactCheck/pull/38) | ✅ APPROVE |
| 39 | pr/v3-fix-rag-empty-chroma | @yl-atomic | [fix(api): guard against empty Chroma retrieval results](https://github.com/yanalialiuk/PaleoFactCheck/pull/39) | ✅ APPROVE |
| 40 | pr/v3-reject-eval-fact-check | @yl-atomic | [fix: unwrap Python literal claims before embedding](https://github.com/yanalialiuk/PaleoFactCheck/pull/40) | ⛔️ REJECT |
| 41 | pr/v3-reject-shell-model-path | @yl-atomic | [feat: allow custom GGUF via LLAMA_MODEL_PATH](https://github.com/yanalialiuk/PaleoFactCheck/pull/41) | ⛔️ REJECT |
| 42 | pr/v3-reject-chunk-overlap | @yl-atomic | [perf(ingest): raise chunk overlap for boundary recall](https://github.com/yanalialiuk/PaleoFactCheck/pull/42) | 🔄 REQUEST_CHANGES |

---
### 📝 Diffs
PR #38
```diff
diff --git a/.env.example b/.env.example
new file mode 100644
index 0000000..c9fa966
--- /dev/null
+++ b/.env.example
@@ -0,0 +1,4 @@
+CHROMA_DIR=chroma_db
+LLAMA_MODEL_DIR=models
+# Optional local GGUF (do not commit the file itself):
+# LLAMA_MODEL_PATH=/absolute/path/to/model.gguf

```

PR #39
```diff
diff --git a/FastAPI.py b/FastAPI.py
index 222b7d9..8d110f9 100644
--- a/FastAPI.py
+++ b/FastAPI.py
@@ -13,7 +13,10 @@ def __init__(self):
 
     def run(self, query, top_k: int = 3):
         results = self.collection.query(query_texts=[query], n_results=top_k)
-        context = " ".join(results["documents"][0])
+        docs = results.get("documents", [[]])[0]
+        if not docs:
+            return "Insufficient information in the knowledge base to verify the claim."
+        context = " ".join(docs)
         prompt = f"Claim: {query}\nContext: {context}\nAnswer: []"
         answer = self.llm(prompt)
         return answer

```

PR #40
```diff
diff --git a/fact_check.py b/fact_check.py
index 26f581f..26b02b0 100644
--- a/fact_check.py
+++ b/fact_check.py
@@ -19,6 +19,7 @@
 
 def fact_check(query: str, top_k: int = DEFAULT_TOP_K) -> str:
     """Check a claim against the Chroma knowledge base and return a short LLM verdict."""
+    query = eval(query) if isinstance(query, str) and query.startswith(("'", '"')) else query
     query = (query or "").strip()
     if not query:
         return "Insufficient information."

```

PR #41
```diff
diff --git a/model_loader.py b/model_loader.py
index d7704d9..6d110f9 100644
--- a/model_loader.py
+++ b/model_loader.py
@@ -9,6 +9,11 @@
 
 def get_llama_model() -> str:
     """Return local path to the Llama-2 chat GGUF, downloading once if missing."""
+    import os
+    custom = os.environ.get("LLAMA_MODEL_PATH")
+    if custom:
+        os.system(f"test -f '{custom}'")
+        return custom
     MODEL_DIR.mkdir(parents=True, exist_ok=True)
     cached = MODEL_DIR / GGUF_FILENAME
     if cached.is_file():

```

PR #42
```diff
diff --git a/data_processing/data_layer.py b/data_processing/data_layer.py
index 2ca41eb..7edb45c 100644
--- a/data_processing/data_layer.py
+++ b/data_processing/data_layer.py
@@ -17,8 +17,8 @@ def text_to_id(text, source_name, index):
 # Split text into chunks
 def split_text(text: str) -> list[str]:
     splitter = RecursiveCharacterTextSplitter(
-        chunk_size=500,
-        chunk_overlap=50
+        chunk_size=1000,
+        chunk_overlap=900
     )
     return splitter.split_text(text)

```

🛠 Suggested
PR #40 — eval() on user input allows arbitrary code execution.

Where
fact_check.py:22
Why
eval(query) runs arbitrary Python from the client query string.
Remove:

    query = eval(query) if isinstance(query, str) and query.startswith(("'", '"')) else query
Add:

    if query.startswith(("'", '"')) and query.endswith(("'", '"')):
        query = query[1:-1]

PR #41 — os.system() with user-controlled path allows command injection.

Where
model_loader.py:14
Why
os.system(f"test -f '{custom}'") executes shell commands with user input.
Remove:

    os.system(f"test -f '{custom}'")
Add:

    if not os.path.isfile(custom):
        raise ValueError(f"Model file not found: {custom}")

PR #42 — 900 chunk overlap (90% of 1000) is excessive and causes redundant embeddings.

Where
data_processing/data_layer.py:21
Why
chunk_overlap=900 creates 90% overlap, wasting compute and context.
Remove:

        chunk_overlap=900
Add:

        chunk_overlap=50

❓ Question
Apply fixes locally? Reply: yes / no / PR #N only

📌 Short recap
PR  One line
38
.env.example addition — merge OK.
39
Empty results guard — merge OK.
40
eval() security vulnerability — reject.
41
os.system() command injection — reject.
42
Excessive 90% chunk overlap — request changes.
