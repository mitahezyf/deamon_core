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

    # --- Modele LLM / VLM (Ollama na Windows + RTX 3090) ---
    llm_model: str = "huihui_ai/qwen3.5-abliterated:9b"
    router_model: str = "huihui_ai/qwen3.5-abliterated:0.8b"
    ollama_url: str = "http://192.168.0.215:11434"

    # --- TTS (Piper ONNX CPU) ---
    language: str = "pl"

    # --- Sciezki bazowe ---
    project_dir: Path = Path(__file__).parent.parent.parent
    piper_model_path: Optional[Path] = None
    piper_config_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # --- Serwer API ---
    api_host: str = "0.0.0.0"  # nosec B104 - nasluchiwanie w sieci LAN
    api_port: int = 8000

    # --- Debugowanie ---
    debug_mode: bool = False

    @model_validator(mode="after")
    def _ustaw_domyslne_sciezki(self) -> "DaemonSettings":
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


# Singleton importowany przez reszte modulow serwera
settings = DaemonSettings()