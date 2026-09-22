import os
from pathlib import Path
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    # Sprawdzamy oba pliki .env (w głownym katalogu albo w klienckim)
    env_paths = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env"
    ]
    for p in env_paths:
        if p.exists():
            load_dotenv(dotenv_path=p)
except ImportError:
    pass

@dataclass
class ClientSettings:
    daemon_server_url: str = os.getenv("DAEMON_SERVER_URL", "ws://192.168.0.103:8000/api/v1/ws/agent")
    wakeword_model: str = os.getenv("WAKEWORD_MODEL", "alexa")
    wakeword_threshold: float = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
    vad_silence_timeout_ms: int = int(os.getenv("VAD_SILENCE_TIMEOUT_MS", "700"))
    whisper_model_name: str = os.getenv("WHISPER_MODEL_NAME", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cuda")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    daemon_ollama_host: str = os.getenv("DAEMON_OLLAMA_HOST", "http://192.168.0.215:11434")
    model_router: str = os.getenv("DAEMON_MODEL_ROUTER", "huihui_ai/qwen3.5-abliterated:0.8b")
