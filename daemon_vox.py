import json
import logging
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import torch
import torchaudio
from TTS.api import TTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] daemon_vox - %(message)s",
)
log = logging.getLogger("daemon_vox")

# --- Optymalizacje CUDA ---
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Sciezki projektu ---
PROJECT_DIR = Path(r"K:\DAEMON_PROJECT")
SAMPLES_DIR = PROJECT_DIR / "voice_samples"
CACHE_PATH = PROJECT_DIR / "daemon_voice_cache.pth"
OUTPUT_DIR = PROJECT_DIR / "tests"
SOCKET_HOST = "127.0.0.1"
SOCKET_PORT = 59721
LANGUAGE = "pl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYNTH_PARAMS = dict(
    temperature=0.75,
    length_penalty=1.0,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    speed=0.95,
    enable_text_splitting=True,
)


def zbierz_probki() -> list[str]:
    pliki = sorted(SAMPLES_DIR.glob("*.wav"))
    if not pliki:
        raise FileNotFoundError(f"Brak plikow .wav w {SAMPLES_DIR}")
    log.info("Probki: %s", [p.name for p in pliki])
    return [str(p) for p in pliki]


def zaladuj_model() -> TTS:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Ladowanie modelu XTTS v2 na %s...", device)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model
    model.eval()
    try:
        model.half()
        log.info("Tryb FP16 aktywny")
    except Exception as e:
        log.warning("FP16 niedostepne: %s", e)
    return tts


def zbuduj_cache_glosu(tts: TTS, probki: list[str]) -> None:
    log.info("Obliczam embeddingi glosu ze wszystkich probek...")
    model = tts.synthesizer.tts_model
    model.float()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=probki
    )
    try:
        model.half()
    except Exception:
        pass
    torch.save(
        {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding},
        CACHE_PATH,
    )
    log.info("Cache zapisany: %s", CACHE_PATH)


def zaladuj_lub_zbuduj_cache(tts: TTS, probki: list[str]):
    if CACHE_PATH.exists():
        log.info("Cache: %s", CACHE_PATH.name)
        dane = torch.load(
            CACHE_PATH, map_location=tts.synthesizer.tts_model.device, weights_only=True
        )
        return dane["gpt_cond_latent"], dane["speaker_embedding"]
    log.warning("Cache nie znaleziony - buduje...")
    zbuduj_cache_glosu(tts, probki)
    dane = torch.load(
        CACHE_PATH, map_location=tts.synthesizer.tts_model.device, weights_only=True
    )
    return dane["gpt_cond_latent"], dane["speaker_embedding"]


def warmup(tts: TTS, gpt_cond_latent, speaker_embedding) -> None:
    # Rozgrzewka CUDA - pierwsze wywoanie zawsze wolniejsze przez JIT kerneli
    log.info("Rozgrzewanie CUDA (warmup)...")
    model = tts.synthesizer.tts_model
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in model.inference_stream(
            text="Test.",
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **SYNTH_PARAMS,
        ):
            pass
    log.info("Warmup zakonczony - serwer gotowy.")


def generuj(
    tts: TTS, gpt_cond_latent, speaker_embedding, text: str, output_path: Path
) -> dict:
    model = tts.synthesizer.tts_model
    sample_rate = 24000
    t0 = time.perf_counter()
    t_first = None
    chunks = []

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for chunk in model.inference_stream(
            text=text,
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **SYNTH_PARAMS,
        ):
            if t_first is None:
                t_first = time.perf_counter() - t0
            chunks.append(chunk)

    wav = torch.cat(chunks, dim=-1).unsqueeze(0).float().cpu()
    torchaudio.save(str(output_path), wav, sample_rate)
    elapsed = time.perf_counter() - t0
    dur = wav.shape[-1] / sample_rate

    return {
        "latency_first_chunk": round(t_first or 0.0, 3),
        "total_time": round(elapsed, 3),
        "audio_duration": round(dur, 2),
        "output": str(output_path),
    }


def obsluz_klienta(conn, tts, gpt_cond_latent, speaker_embedding):
    try:
        # Format: 4 bajty dugosci JSON | JSON
        raw_len = conn.recv(4)
        if not raw_len:
            return
        msg_len = struct.unpack(">I", raw_len)[0]
        data = b""
        while len(data) < msg_len:
            chunk = conn.recv(msg_len - len(data))
            if not chunk:
                break
            data += chunk
        req = json.loads(data.decode("utf-8"))
        text = req.get("text", "")
        out_name = req.get("output", "daemon_out.wav")
        output_path = OUTPUT_DIR / out_name

        log.info("[REQ] %s...", text[:60])
        result = generuj(tts, gpt_cond_latent, speaker_embedding, text, output_path)
        log.info(
            "[OK] 1.chunk=%ss total=%ss",
            result["latency_first_chunk"],
            result["total_time"],
        )

        resp = json.dumps(result).encode("utf-8")
        conn.sendall(struct.pack(">I", len(resp)) + resp)
    except Exception as e:
        log.exception("[ERR] %s", e)
    finally:
        conn.close()


def tryb_serwer(tts, gpt_cond_latent, speaker_embedding):
    log.info("Serwer nasluchuje na %s:%s", SOCKET_HOST, SOCKET_PORT)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((SOCKET_HOST, SOCKET_PORT))
    srv.listen(5)
    while True:
        conn, addr = srv.accept()
        t = threading.Thread(
            target=obsluz_klienta,
            args=(conn, tts, gpt_cond_latent, speaker_embedding),
            daemon=True,
        )
        t.start()


def tryb_jednorazowy(tts, gpt_cond_latent, speaker_embedding, text: str):
    output_path = OUTPUT_DIR / "daemon_final.wav"
    log.info("Daemon generuje glos...")
    result = generuj(tts, gpt_cond_latent, speaker_embedding, text, output_path)
    log.info("Latencja 1. chunk: %ss", result["latency_first_chunk"])
    log.info(
        "Czas calkowity: %ss | Czas audio: %ss",
        result["total_time"],
        result["audio_duration"],
    )
    log.info("Plik zapisany: %s", result["output"])


def main():
    args = sys.argv[1:]
    probki = zbierz_probki()
    tts = zaladuj_model()

    if "--build-voice" in args:
        zbuduj_cache_glosu(tts, probki)
        log.info("Cache przebudowany.")
        return

    gpt_cond_latent, speaker_embedding = zaladuj_lub_zbuduj_cache(tts, probki)

    if "--server" in args:
        # Tryb serwera: warmup + nasuchiwanie - model zyje przez cay czas = instant po starcie
        warmup(tts, gpt_cond_latent, speaker_embedding)
        tryb_serwer(tts, gpt_cond_latent, speaker_embedding)
    else:
        # Tryb jednorazowy: szybki, bez warmupa
        text = (
            " ".join(args)
            if args
            else (
                "Wodzu, system melduje pena gotowosc do startu InPost AirLines. "
                "Barszcz sosnowskiego smakuje wybornie i wasnie sie konczy gotowac."
            )
        )
        tryb_jednorazowy(tts, gpt_cond_latent, speaker_embedding, text)


if __name__ == "__main__":
    main()
