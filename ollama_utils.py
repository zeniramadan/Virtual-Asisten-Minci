"""
Helper untuk memastikan server Ollama lokal berjalan dan model yang dibutuhkan sudah ada.

Di Colab, `ollama serve` dijalankan manual lewat subprocess. Di laptop, Ollama biasanya
sudah jalan otomatis (aplikasi desktop / service), jadi di sini cukup dicek dulu dan baru
dinyalakan kalau memang belum aktif.
"""
import atexit
import os
import shutil
import subprocess
import sys
import time

import ollama
import requests

import config

_started_process = None


def _base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def is_ollama_running(timeout: float = 2.0) -> bool:
    try:
        return requests.get(_base_url(), timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _stop_started_process():
    if _started_process is not None and _started_process.poll() is None:
        _started_process.terminate()


def ensure_ollama_running(wait_seconds: int = 30) -> bool:
    """Pastikan server Ollama aktif. Kalau belum, coba nyalakan `ollama serve`."""
    global _started_process

    if is_ollama_running():
        return True

    if shutil.which("ollama") is None:
        print("[!] Ollama tidak ditemukan di PATH. Install dulu dari https://ollama.com/download")
        return False

    print("Server Ollama belum aktif -> menyalakan `ollama serve` ...")
    _started_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_stop_started_process)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_ollama_running():
            print("Ollama siap!")
            return True
        time.sleep(0.5)

    print("[!] Server Ollama tidak merespons setelah dinyalakan.")
    return False


def _installed_model_names() -> set:
    resp = ollama.list()
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])

    names = set()
    for m in models or []:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.add(name)
    return names


def _has_model(installed: set, wanted: str) -> bool:
    wanted = wanted.strip()
    if wanted in installed:
        return True
    return ":" not in wanted and f"{wanted}:latest" in installed


def ensure_ready(required: list = None, exit_on_error: bool = True) -> bool:
    """
    Cek server Ollama + model yang dibutuhkan.
    Default: model embedding dan model LLM dari config.
    """
    required = required or [config.EMBED_MODEL, config.LLM_MODEL]

    if not ensure_ollama_running():
        if exit_on_error:
            sys.exit(1)
        return False

    installed = _installed_model_names()
    missing = [m for m in required if not _has_model(installed, m)]
    if not missing:
        return True

    print(f"[!] Model belum ada di Ollama lokal: {', '.join(missing)}")
    for m in missing:
        if m == config.LLM_MODEL and m == "minci":
            print(f"    - '{m}' adalah model custom. Buat dulu dengan Modelfile-nya:")
            print(f"        ollama create {m} -f Modelfile")
            print("      atau pakai model lain lewat .env, mis. LLM_MODEL=llama3.2 (lalu: ollama pull llama3.2)")
        else:
            print(f"    - unduh dengan: ollama pull {m}")

    if exit_on_error:
        sys.exit(1)
    return False
