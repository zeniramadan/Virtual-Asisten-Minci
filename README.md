# Minci (versi lokal)

Chatbot RAG untuk WhatsApp (Ollama + ChromaDB + FastAPI), hasil konversi dari notebook Colab `Minci.ipynb`
supaya bisa dijalankan di laptop.

## Struktur folder

Semua data disimpan **di sebelah file kode** (bukan lagi di Google Drive).
Lokasinya bisa diubah dengan environment variable `MINCI_BASE_DIR`.

```
Minci_local/
├── .env                 <- salin dari .env.example, isi token
├── ollama_utils.py      <- cek/nyalakan Ollama + cek model
├── app.py               <- webhook FastAPI untuk WhatsApp
├── run_server.py        <- jalankan app.py (+ ngrok)
├── requirements.txt
├── documents/           <- taruh PMB.docx, KRS.docx, BIAYA.pdf, KALENDER.pdf, dll
├── dataset/             <- chitchat.json, ground_truth.json, stress_cases.json
└── rag/
   ├── config.py        <- pengaturan (path, model, prompt, dll)
   ├── ingest.py        <- baca dokumen -> chunk -> embedding -> ChromaDB
   ├── query.py         <- pipeline RAG + riwayat chat (bisa dipakai mode CLI)
   ├── evaluate_rag.py  <- evaluasi RAG (Gemini sebagai judge)
    ├── dataset/         <- database vektor ChromaDB (dibuat oleh ingest.py)
    ├── database/        <- chat_history.db (SQLite)
    └── eval_reports/    <- hasil evaluasi
```

## Langkah pertama kali

1. **Python 3.10 – 3.12**, lalu buat virtual environment dan install library:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```
2. **Install Ollama** dari https://ollama.com/download, lalu unduh model:
   ```
   ollama pull bge-m3
   ollama pull llama3.2
   ```
   `LLM_MODEL` default-nya `minci` (model custom). Kalau belum dibuat di Ollama lokal, buat dengan
   `ollama create minci -f Modelfile`, **atau** isi `LLM_MODEL=llama3.2` di `.env`.
   (System prompt sudah dikirim dari `query.py` di setiap pesan.)
3. **Salin data dari Google Drive** (`MyDrive/Minci_STTC`) ke folder ini:
   `documents/` dan `dataset/`, serta file `.env`. Folder `rag/` tidak perlu disalin, dibuat ulang oleh langkah 5.
   Kalau belum punya `.env`: salin `.env.example` jadi `.env` lalu isi.
4. Cek isi `.env` (token WhatsApp, ngrok, Gemini).
5. **Ingest dokumen** (harus diulang setiap dokumen di `documents/` berubah):
   ```
   python -m rag.ingest
   ```
   Pertama kali jalan butuh internet (library mengunduh data tokenizer).

## Menjalankan

| Perintah                          | Fungsi                                                                 |
| --------------------------------- | ---------------------------------------------------------------------- |
| `python -m rag.query`             | Ngobrol dengan Minci lewat terminal (untuk tes, tanpa WhatsApp)        |
| `python run_server.py`            | Webhook + tunnel ngrok. URL publik dicetak di terminal (`.../webhook`) |
| `python run_server.py --no-ngrok` | Webhook lokal saja                                                     |
| `python -m rag.evaluate_rag`      | Evaluasi retrieval, faithfulness, dan stress test                      |

Hentikan server (Ctrl+C) sebelum menjalankan ulang `ingest.py`, supaya folder database tidak terkunci (terutama di Windows).

## Yang berubah dari versi Colab

- Path Google Drive diganti folder proyek lokal.
- Tidak ada lagi `drive.mount`, `%%writefile`, `!pip`, `!apt-get`, dan `OLLAMA_MODELS` (Ollama lokal menyimpan modelnya sendiri).
- Ollama tidak di-`serve` manual: dicek dulu, dinyalakan otomatis hanya kalau belum jalan.
- Cell 6 (ngrok + uvicorn) menjadi `run_server.py`, tanpa `nest_asyncio` dan tanpa `pkill` (agar jalan juga di Windows).
- Server bind ke `127.0.0.1` (ngrok tetap bisa mengaksesnya). Pakai `--host 0.0.0.0` kalau perlu diakses dari perangkat lain.
- Nama model bisa diubah lewat `.env` (`LLM_MODEL`, `EMBED_MODEL`).
- Template `.env` lama menempelkan baris `NGROK_AUTH_TOKEN`, `GEMINI_API_KEY`, `GEMINI_MODEL` jadi satu baris
  (kurang `\n`). Di `.env.example` sudah diperbaiki.
