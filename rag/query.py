import os
import re
import sys
import json
import time
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb
import ollama
import rag.config as config
import ollama_utils

# ============================================================
# DEBUG HELPERS
# ============================================================
def _line(char="─", n=70):
    print(char * n)

def _debug_block(label: str, content: str):
    _line("·")
    print(f"[DEBUG] {label}:")
    print(content if content.strip() else "(kosong)")

def _debug_retrieved_chunks(query: str, results: dict):
    ids = results["ids"][0]
    if not ids:
        print("[DEBUG] Tidak ada chunk yang ter-retrieve.")
        return
    for rank, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
        preview = doc[:60].replace("\n", " ").strip()
        print(f'[DEBUG] #{rank} dist={dist:.4f} | {meta.get("file_name")} | "{preview}..."')

# ============================================================
# DATABASE RIWAYAT CHAT
# ============================================================
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id)")
    conn.commit()
    return conn

def _clear_history(conn: sqlite3.Connection, user_id: str):
    conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()

def _get_history(conn: sqlite3.Connection, user_id: str):
    cur = conn.execute("SELECT role, content, created_at FROM chat_history WHERE user_id = ? ORDER BY id ASC", (user_id,))
    rows = cur.fetchall()
    if not rows: return []
    last_ts = rows[-1][2]
    idle_seconds = time.time() - last_ts
    if idle_seconds > config.HISTORY_TIMEOUT_SECONDS:
        _clear_history(conn, user_id)
        return []
    return [{"role": r[0], "content": r[1]} for r in rows]

def _save_message(conn: sqlite3.Connection, user_id: str, role: str, content: str):
    conn.execute("INSERT INTO chat_history (user_id, role, content, created_at) VALUES (?, ?, ?, ?)", (user_id, role, content, time.time()))
    conn.commit()

def reset_history(user_id: str):
    conn = _get_db()
    try: _clear_history(conn, user_id)
    finally: conn.close()

# ============================================================
# CHITCHAT DETECTION
# ============================================================
_chitchat_data = None

def _load_chitchat() -> dict:
    if not os.path.isfile(config.CHITCHAT_PATH): return {}
    with open(config.CHITCHAT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_chitchat(message: str):
    global _chitchat_data
    if _chitchat_data is None: _chitchat_data = _load_chitchat()
    text = re.sub(r"[^\w\s]", "", message.lower().strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return None, None
    for category, keywords in _chitchat_data.items():
        for kw in keywords:
            if re.sub(r"[^\w\s]", "", str(kw).lower().strip()) == text:
                return category, str(kw)
    return None, None

# ============================================================
# RETRIEVAL (RAG)
# ============================================================
_chroma_client = None
_collection = None

def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = _chroma_client.get_collection(config.COLLECTION_NAME)
    return _collection

def retrieve_context(query: str, top_k: int = None, fetch_k: int = None):
    """
    Alur retrieval (over-fetch -> filter jarak -> truncate):
      1. Ambil kandidat lebih banyak dari yang akhirnya dipakai (fetch_k > top_k),
         supaya proses filter di bawah masih punya cukup pilihan.
      2. Buang kandidat yang jaraknya di atas config.DISTANCE_THRESHOLD (dianggap tidak relevan) ->
         kalau semua kandidat terbuang, context_str akhirnya kosong dan fallback di system prompt aktif.
      3. Urutkan sisanya berdasarkan jarak, lalu ambil top_k teratas.
    """
    top_k = top_k or config.TOP_K
    fetch_k = fetch_k or max(getattr(config, "RETRIEVE_FETCH_K", top_k * 2), top_k)

    collection = _get_collection()

    # Jangan minta n_results lebih banyak dari jumlah chunk yang benar-benar ada
    # di koleksi -> menghindari warning/error dari ChromaDB kalau database masih kecil.
    try:
        collection_count = collection.count()
    except Exception:
        collection_count = None
    if collection_count == 0:
        print("[DEBUG] Koleksi ChromaDB kosong -> tidak ada dokumen untuk di-retrieve.")
        return []
    if collection_count is not None:
        fetch_k = min(fetch_k, collection_count)

    query_embedding = ollama.embeddings(model=config.EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=fetch_k)
    _debug_retrieved_chunks(query, results)

    candidates = [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

    # 2. filter jarak (kalau threshold-nya diaktifkan di config)
    threshold = getattr(config, "DISTANCE_THRESHOLD", None)
    if threshold is not None:
        candidates = [c for c in candidates if c["distance"] <= threshold]

    # 3. urutkan berdasarkan jarak, ambil top_k
    candidates.sort(key=lambda c: c["distance"])
    selected = candidates[:top_k]

    if not selected:
        print("[DEBUG] Tidak ada chunk relevan setelah filter -> context kosong (fallback akan aktif).")

    return selected

def build_context_str(chunks: list) -> str:
    if not chunks: return ""
    return "\n\n---\n\n".join([f"[Sumber: {c['metadata'].get('file_name')}]\n{c['text']}" for c in chunks])

# ============================================================
# PERTANYAAN MANDIRI (STANDALONE QUESTION)
# ============================================================
def _generate_standalone_question(history: list, user_message: str) -> str:
    """
    Kalau ada riwayat chat, minta LLM menyusun ulang pesan pengguna
    (yang bisa jadi cuma pertanyaan lanjutan/singkat) menjadi satu
    pertanyaan mandiri yang lengkap dan berdiri sendiri.
    Pertanyaan mandiri ini yang dipakai untuk retrieval ke ChromaDB,
    biar hasil pencarian dokumen lebih akurat sesuai konteks obrolan.
    """
    if not history:
        return user_message

    chat_history_str = "\n".join(
        f"{h['role']}: {h['content']}" for h in history[-(config.MAX_HISTORY_TURNS * 2):]
    )
    prompt = config.STANDALONE_QUESTION_PROMPT.format(
        chat_history=chat_history_str,
        question=user_message,
    )
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": config.LLM_TEMPERATURE},
    )
    return response["message"]["content"].strip()

# ============================================================
# LLM CALL
# ============================================================
def _call_llm(system_prompt: str, history: list, user_message: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-(config.MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_message})
    response = ollama.chat(model=config.LLM_MODEL, messages=messages, options={"temperature": config.LLM_TEMPERATURE})
    return response["message"]["content"]

# ============================================================
# MAIN PIPELINE
# ============================================================
def generate_reply(user_id: str, user_message: str) -> str:
    conn = _get_db()
    try:
        _line("=")
        print(f"[DEBUG] user_id = {user_id}")
        _debug_block("Pertanyaan (asli dari user)", user_message)

        history = _get_history(conn, user_id)
        category, _ = detect_chitchat(user_message)

        if category:
            print(f"[DEBUG] Terdeteksi CHITCHAT -> kategori: {category}")
            system_prompt = config.CHITCHAT_SYSTEM_PROMPT
        else:
            standalone_question = _generate_standalone_question(history, user_message)
            _debug_block("Pertanyaan mandiri (hasil LLM)", standalone_question)
            try:
                chunks = retrieve_context(standalone_question)
                context_str = build_context_str(chunks)
                system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(context_str=context_str)
            except Exception as e:
                system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(context_str="")
                print(f"[DEBUG] RAG belum di-ingest atau error: {e}")

        answer = _call_llm(system_prompt, history, user_message)
        _debug_block("Jawaban", answer)
        _line("=")

        _save_message(conn, user_id, "user", user_message)
        _save_message(conn, user_id, "assistant", answer)
        return answer
    finally:
        conn.close()

if __name__ == "__main__":
    ollama_utils.ensure_ready()
    print("Mode CLI diaktifkan. Ketik pesan (ketik 'exit' untuk keluar):")
    while True:
        try:
            msg = input("[User]: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.strip().lower() == "exit":
            break
        if not msg.strip():
            continue
        print("[Minci]:", generate_reply("cli-user", msg))
