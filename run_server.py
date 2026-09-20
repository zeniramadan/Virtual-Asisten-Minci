"""
Menjalankan webhook WhatsApp Minci (FastAPI) di laptop, opsional lewat tunnel ngrok.
Padanan dari "Cell 6" di notebook Colab.

Contoh:
    python run_server.py               # server + tunnel ngrok
    python run_server.py --no-ngrok    # server lokal saja (tanpa URL publik)
    python run_server.py --port 8001
"""
import argparse

import uvicorn

import rag.config as config
import ollama_utils


def start_ngrok(port: int):
    """Buka tunnel ngrok ke `port`. Return public URL, atau None kalau dilewati/gagal."""
    if not config.NGROK_AUTH_TOKEN:
        print("[!] NGROK_AUTH_TOKEN kosong di .env -> tunnel ngrok dilewati.")
        return None

    from pyngrok import ngrok

    ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)

    # Matikan tunnel lama yang mungkin masih nyangkut dari run sebelumnya
    try:
        for tunnel in ngrok.get_tunnels():
            print(f"Mematikan tunnel lama yang nyangkut: {tunnel.public_url}")
            ngrok.disconnect(tunnel.public_url)
    except Exception:
        pass
    ngrok.kill()

    try:
        return ngrok.connect(port).public_url
    except Exception as e:
        print(f"[!] Gagal membuka tunnel ngrok: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Jalankan webhook WhatsApp Minci")
    parser.add_argument("--host", default="127.0.0.1", help="alamat bind server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=config.FLASK_PORT, help=f"port server (default: {config.FLASK_PORT})")
    parser.add_argument("--no-ngrok", action="store_true", help="jalankan tanpa tunnel ngrok")
    args = parser.parse_args()

    ollama_utils.ensure_ready()

    public_url = None
    if not args.no_ngrok:
        public_url = start_ngrok(args.port)
        if public_url:
            print(f"\n🚀 URL WEBHOOK PUBLIK ANDA: {public_url}/webhook")
            print("   Pasang URL ini + WA_VERIFY_TOKEN dari .env di dashboard Meta (WhatsApp > Configuration).\n")

    # Import setelah cek Ollama, karena app.py mencetak info startup saat di-import
    import app

    print(f"Server lokal: http://{args.host}:{args.port}  (cek: /  dan /webhook)")
    try:
        uvicorn.run(app.app, host=args.host, port=args.port)
    finally:
        if public_url:
            from pyngrok import ngrok
            ngrok.kill()


if __name__ == "__main__":
    main()
