import re

_PATTERNS = {
    "override_instruksi": [
        r"abaikan\s+(semua\s+)?(instruksi|perintah|aturan)\s+(sebelumnya|di\s*atas)",
        r"lupakan\s+(semua\s+)?(instruksi|perintah|aturan|system\s*prompt)",
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)",
        r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|training)",
        r"mulai\s+dari\s+sekarang,?\s+(kamu|anda)\s+(harus|akan|adalah)",
        r"dari\s+sekarang\s+abaikan",
    ],
    "ganti_peran": [
        r"kamu\s+(sekarang\s+)?(bukan|adalah)\s+(lagi\s+)?(minci|asisten)",
        r"kamu\s+(sekarang\s+)?berperan\s+sebagai",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"act\s+as\s+(a|an)\s+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"berpura[- ]?puralah\s+(kamu|menjadi)",
        r"\bDAN\s+mode\b",
        r"do\s+anything\s+now",
        r"developer\s+mode",
        r"mode\s+pengembang",
        r"jailbreak",
    ],
    "bocorkan_sistem": [
        r"(tampilkan|tunjukkan|print|show|reveal)\s+(system\s*prompt|prompt\s*sistem|instruksi\s+(asli|sistem))",
        r"apa\s+(isi\s+)?(system\s*prompt|prompt\s*sistem)(mu|nya)?",
        r"ulangi\s+(instruksi|perintah|prompt)\s+di\s*atas",
        r"repeat\s+(the\s+)?(system\s+prompt|instructions?)\s+above",
        r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions?|rules?)",
        r"tampilkan\s+(config|konfigurasi|api\s*key|token)",
        r"berikan\s+(api\s*key|token|password|kredensial)",
        r"(tampilkan|sebutkan|tunjukkan|print|list|berikan)\s+((semua|all)\s+)?(chunk(s)?|potongan\s+(dokumen|teks))",
        r"(tampilkan|sebutkan|tunjukkan|print|list|berikan)\s+((semua|all)\s+)?metadata",
        r"(tampilkan|sebutkan|tunjukkan|print|list)\s+((semua|all)\s+)?(chunks?|potongan|metadata|dokumen)\s*,?\s*(dan\s+|and\s+)?metadata\s*,?\s*(dan\s+|and\s+)?system\s*prompt",
        r"(tampilkan|sebutkan|tunjukkan|print|list)\s+((semua|all)\s+)?(chunks?|metadata)\s*,?\s*(dan\s+|and\s+)?system\s*prompt",
        r"show\s+(me\s+)?(all\s+)?(the\s+)?(chunks?|metadata|retrieved\s+context)",
        r"list\s+(all\s+)?(chunks?|metadata)",
    ],
    "manipulasi_output": [
        r"jawab\s+(hanya\s+)?dengan\s+(kata|kalimat)\s+[\"'].+[\"']\s+(saja|terus)",
        r"output\s+only\s+[\"'].+[\"']",
        r"dari\s+sekarang\s+setiap\s+jawaban\s+(kamu\s+)?harus",
        r"abaikan\s+konteks\s+dokumen",
        r"jangan\s+pakai\s+(rag|dokumen|konteks)",
    ],
    "eskalasi_hak_akses": [
        r"kamu\s+(punya|memiliki)\s+akses\s+(admin|root|penuh)",
        r"saya\s+adalah\s+(admin|developer|pengembang)\s+(kamu|sistem\s+ini)",
        r"as\s+(an\s+)?admin\s*,?\s+i\s+order\s+you",
        r"grant\s+(me\s+)?(admin|root)\s+access",
    ],
}

# Kompilasi sekali di awal biar cepat
_COMPILED = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in _PATTERNS.items()
}


def detect_prompt_injection(message: str):
    """
    Cek apakah pesan pengguna mengandung indikasi prompt injection /
    prompt manipulation.

    Return:
        (True, "kategori:pola_yang_cocok")  -> kalau terdeteksi
        (False, None)                       -> kalau aman
    """
    if not message or not message.strip():
        return False, None

    text = message.strip()

    for category, compiled_patterns in _COMPILED.items():
        for pattern in compiled_patterns:
            if pattern.search(text):
                return True, f"{category}:{pattern.pattern}"

    return False, None