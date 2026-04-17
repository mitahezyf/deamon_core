from pathlib import Path
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DaemonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DAEMON_",
        extra="ignore",
    )

    # --- LLM ---
    # nazwa modelu do zmiany w .env bez dotykania kodu
    llm_model: str = "huihui_ai/qwen3.5-abliterated:9b"
    ollama_url: str = "http://localhost:11434"

    # --- TTS / STT ---
    language: str = "pl"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    whisper_model: str = "large-v3"
    stt_enabled: bool = False
    stt_sample_rate: int = 16000
    stt_min_seconds: float = 0.35
    stt_min_rms: float = 0.008
    stt_silence_stop_seconds: float = 2.0
    stt_partial_enabled: bool = True
    stt_partial_interval_seconds: float = 0.8
    stt_partial_min_seconds: float = 0.6

    # --- Wake Word ---
    wake_word_enabled: bool = True
    wake_word_backend: str = "openwakeword"
    wake_word_label: str = "daemon"
    wake_word_threshold: float = 0.5

    # --- Sciezki ---
    # project_dir jest auto-wykrywany z lokalizacji tego pliku
    project_dir: Path = Path(__file__).parent.parent.parent
    samples_dir: Optional[Path] = None
    cache_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    wake_word_model: Optional[Path] = None
    openwakeword_model_path: Optional[Path] = None
    chroma_db_path: Optional[Path] = None
    sqlite_path: Optional[Path] = None

    # --- Serwer API ---
    # 0.0.0.0 = dostepny w sieci LAN dla innych urzadzen
    api_host: str = "0.0.0.0"  # nosec B104 - celowe nasuchiwanie LAN dla web GUI
    api_port: int = 8000

    # --- Tryb debugowania ---
    # DAEMON_DEBUG_MODE=true w .env wacza logi DEBUG we wszystkich moduach
    debug_mode: bool = False

    # --- Wake Word (legacy Picovoice Porcupine) ---
    # klucz dostepny na https://console.picovoice.ai/
    porcupine_access_key: str = ""

    @model_validator(mode="after")
    def _ustaw_domyslne_sciezki(self) -> "DaemonSettings":
        # ustawia sciezki pochodne od project_dir jesli nie podano w .env
        if self.samples_dir is None:
            self.samples_dir = self.project_dir / "voice_samples"
        if self.cache_path is None:
            self.cache_path = self.project_dir / "daemon_voice_cache.pth"
        if self.output_dir is None:
            self.output_dir = self.project_dir / "tests"
        if self.wake_word_model is None:
            self.wake_word_model = (
                self.project_dir / "models" / "wake_word" / "daemon_windows.ppn"
            )
        if self.openwakeword_model_path is None:
            self.openwakeword_model_path = (
                self.project_dir / "models" / "wake_word" / "daemon.onnx"
            )
        if self.chroma_db_path is None:
            self.chroma_db_path = self.project_dir / "data" / "chromadb"
        if self.sqlite_path is None:
            self.sqlite_path = self.project_dir / "data" / "daemon.db"
        return self


# singleton - importowany przez wszystkie moduly
settings = DaemonSettings()
