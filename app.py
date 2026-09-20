import os
import sys
import time
import sqlite3
import requests
from fastapi import BackgroundTasks, Body, FastAPI, Request
from fastapi.responses import PlainTextResponse

import rag.config as config

from rag.query import generate_reply

app = FastAPI(title="Minci WA Webhook")

print(f"[STARTUP] WA_GRAPH_URL yang dipakai: {config.WA_GRAPH_URL}")
print(f"[STARTUP] WA_ACCESS_TOKEN ada: {bool(config.WA_ACCESS_TOKEN)} | WA_PHONE_NUMBER_ID ada: {bool(config.WA_PHONE_NUMBER_ID)}")

DEDUP_TTL_SECONDS = 24 * 60 * 60

# ============================================================
# DEBUG HELPERS
# ============================================================
def _line(char="─", n=70):
    print(char * n)

def _debug_block(label: str, content: str):
    print(f"[DEBUG][APP] {label}:")
    print(content if content.strip() else "(kosong)")

def _get_dedup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_wa_messages (
            message_id TEXT PRIMARY KEY,
            processed_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn

def _already_processed(message_id: str) -> bool:
    conn = _get_dedup_db()
    try:
        conn.execute("DELETE FROM processed_wa_messages WHERE processed_at < ?", (time.time() - DEDUP_TTL_SECONDS,))
        try:
            conn.execute("INSERT INTO processed_wa_messages (message_id, processed_at) VALUES (?, ?)", (message_id, time.time()))
            conn.commit()
            return False
        except sqlite3.IntegrityError:
            return True
    finally:
        conn.close()

def _send_whatsapp_message(to: str, text: str):
    if not config.WA_ACCESS_TOKEN or not config.WA_PHONE_NUMBER_ID:
        print(f"[DRY RUN] WA_ACCESS_TOKEN/WA_PHONE_NUMBER_ID kosong -> balasan TIDAK dikirim ke WA, cuma diprint di sini.")
        print(f"[DRY RUN] Balasan ke {to}: {text}")
        return
    headers = {"Authorization": f"Bearer {config.WA_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    try:
        print(f"\n[WA] Mengirim balasan ke {to} lewat {config.WA_GRAPH_URL}")
        resp = requests.post(config.WA_GRAPH_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code >= 400:
            print(f"[!] Gagal kirim WA ke {to}: {resp.status_code} {resp.text}")
        else:
            print(f"[WA] Balasan berhasil terkirim ke {to}\n")
    except requests.RequestException as e:
        print(f"[!] Error koneksi WA: {e}")

def _process_and_reply(user_id: str, user_text: str):
    _line("=")
    print(f"[DEBUG][APP] Pesan masuk dari WhatsApp | user_id = {user_id}")
    _debug_block("Pertanyaan (dari WA)", user_text)
    _line("-")

    reply_text = generate_reply(user_id=user_id, user_message=user_text)

    _debug_block("Jawaban final (akan dikirim ke WA)", reply_text)
    _line("=")

    _send_whatsapp_message(user_id, reply_text)

@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == config.WA_VERIFY_TOKEN:
        return challenge
    return PlainTextResponse("Verifikasi gagal", status_code=403)

@app.post("/webhook")
def receive_message(payload: dict = Body(default={}), background_tasks: BackgroundTasks = None):
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        if "messages" not in value: return {"status": "ignored"}
        message = value["messages"][0]
        message_id = message.get("id")
        user_id = message["from"]
        msg_type = message.get("type")
    except (KeyError, IndexError, TypeError):
        return {"status": "ignored"}

    if message_id and _already_processed(message_id): return {"status": "duplicate"}
    if msg_type != "text":
        print(f"[DEBUG][APP] Pesan dari {user_id} bertipe '{msg_type}' (bukan teks) -> diabaikan.")
        background_tasks.add_task(_send_whatsapp_message, user_id, "Maaf kak, Minci saat ini baru bisa membaca pesan teks ya 🙏")
        return {"status": "unsupported_type"}

    user_text = message["text"]["body"]
    background_tasks.add_task(_process_and_reply, user_id, user_text)
    return {"status": "received"}

@app.get("/")
def health_check():
    return {"status": "Minci webhook aktif"}
