from pathlib import Path
from typing import Optional
import os

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DAEMON_",
        extra="ignore",
    )

    ollama_host: str = "http://192.168.0.215:11434"
    memgraph_uri: str = "bolt://192.168.0.106:7687"
    model_brain: str = "huihui_ai/qwen3.5-abliterated:9b"
    model_router: str = "huihui_ai/qwen3.5-abliterated:0.8b"

    router_timeout: float = 3.0
    brain_ttft_timeout: float = 15.0

    language: str = "pl"
    
    piper_model_path: Optional[Path] = None
    piper_config_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    host: str = "0.0.0.0"
    port: int = 8000
    debug_mode: bool = False

    @model_validator(mode="after")
    def _ustaw_domyslne_sciezki(self) -> "ServerSettings":
        if self.piper_model_path is None:
            self.piper_model_path = PROJECT_ROOT / "models" / "piper" / "pl_PL-darkman-medium.onnx"
        if self.piper_config_path is None:
            self.piper_config_path = PROJECT_ROOT / "models" / "piper" / "pl_PL-darkman-medium.onnx.json"
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "tests"
        return self

    def get_brain_prompt(self) -> str:
        prompt_path = PROJECT_ROOT / "config" / "prompts" / "brain_system.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Jesteś DAEMON. Zwracaj się per wodzu."

    def get_router_prompt(self) -> str:
        prompt_path = PROJECT_ROOT / "config" / "prompts" / "router_system.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Rozpoznaj intencję JSONem."
