from __future__ import annotations

import os
from pathlib import Path


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def model_store_dir() -> Path:
    raw = os.getenv("DAEMON_MODEL_STORE_DIR", "").strip()
    if raw:
        return Path(raw)
    return _project_dir() / ".model_store"


def configure_model_cache_paths() -> dict[str, Path]:
    base = model_store_dir()
    hf_home = base / "huggingface"
    hf_hub = hf_home / "hub"
    tts_home = base / "tts"

    for path in (base, hf_home, hf_hub, tts_home):
        path.mkdir(parents=True, exist_ok=True)

    # Preserve explicit user overrides if already set in environment.
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub))
    os.environ.setdefault("TTS_HOME", str(tts_home))

    return {
        "model_store_dir": base,
        "hf_home": hf_home,
        "hf_hub": hf_hub,
        "tts_home": tts_home,
    }


MODEL_PATHS = configure_model_cache_paths()
