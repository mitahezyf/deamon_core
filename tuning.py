import logging
from pathlib import Path

import torch
from TTS.api import TTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] tuning - %(message)s",
)
log = logging.getLogger("tuning")

# Skrypt do ręcznego przebudowania cache embeddingów głosu
# Użyj gdy dodasz nowe próbki do voice_samples/

PROJECT_DIR = Path(r"K:\DAEMON_PROJECT")
SAMPLES_DIR = PROJECT_DIR / "voice_samples"
CACHE_PATH = PROJECT_DIR / "daemon_voice_cache.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Ladowanie modelu na %s...", device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Automatyczne zbieranie wszystkich próbek
probki = sorted(SAMPLES_DIR.glob("*.wav"))
if not probki:
    raise FileNotFoundError(f"Brak plików .wav w {SAMPLES_DIR}")

log.info("Znaleziono probki: %s", [p.name for p in probki])
log.info("Obliczam embeddingi glosu ze wszystkich probek...")

model = tts.synthesizer.tts_model
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[str(p) for p in probki]
)

torch.save(
    {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding},
    CACHE_PATH,
)

log.info("Cache zapisany w: %s", CACHE_PATH)
log.info("Teraz daemon_vox.py bedzie ladowal go automatycznie zamiast liczyc od nowa.")
