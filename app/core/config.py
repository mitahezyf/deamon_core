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
    # URL Ollamy na stacji Windows (RTX 3090)
    llm_model: str = "qwen2.5:14b-instruct"
    ollama_url: str = "http://192.168.0.100:11434"

    # --- TTS (Piper ONNX CPU) ---
    language: str = "pl"

    # --- Sciezki ---
    # project_dir jest auto-wykrywany z lokalizacji tego pliku
    project_dir: Path = Path(__file__).parent.parent.parent
    piper_model_path: Optional[Path] = None
    piper_config_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # --- Serwer API ---
    # 0.0.0.0 = dostepny w sieci LAN dla innych urzadzen
    api_host: str = "0.0.0.0"  # nosec B104 - celowe nasluchiwanie LAN dla web GUI
    api_port: int = 8000

    # --- Tryb debugowania ---
    # DAEMON_DEBUG_MODE=true w .env wlacza logi DEBUG we wszystkich modulach
    debug_mode: bool = False

    @model_validator(mode="after")
    def _ustaw_domyslne_sciezki(self) -> "DaemonSettings":
        # ustawia sciezki pochodne od project_dir jesli nie podano w .env
        if self.piper_model_path is None:
            self.piper_model_path = (
                self.project_dir / "models" / "piper" / "pl_PL-darkman-medium.onnx"
            )
        if self.piper_config_path is None:
            self.piper_config_path = (
                self.project_dir / "models" / "piper" / "pl_PL-darkman-medium.onnx.json"
            )
        if self.output_dir is None:
            self.output_dir = self.project_dir / "tests"
        return self


# singleton - importowany przez wszystkie moduly
settings = DaemonSettings()

