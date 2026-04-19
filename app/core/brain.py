from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("brain")


class DaemonBrain:
    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        # Ollama is expected to run externally; here we only mark client readiness.
        self._loaded = True
        log.info("DaemonBrain loaded (model=%s)", settings.llm_model)

    def reply(self, prompt: str) -> str:
        if not self._loaded:
            raise RuntimeError("LLM is not loaded")

        text = prompt.strip()
        if not text:
            return ""

        payload: dict[str, Any] = {
            "model": settings.llm_model,
            "prompt": text,
            "stream": False,
        }

        url = settings.ollama_url.rstrip("/") + "/api/generate"
        timeout = httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=3.0)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            log.error("LLM request failed: %s", exc, exc_info=True)
            raise RuntimeError("llm_request_failed") from exc

        result = str(data.get("response", "")).strip()
        return result
