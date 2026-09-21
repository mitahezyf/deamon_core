"""
app/core/vox.py — Silnik mowy DAEMON

ETAP AKTUALNY: STUB (po Etapie 1 — detoks Coqui)
ETAP DOCELOWY: Piper TTS ONNX CPU (do zaimplementowania w Etapie 2)

Wymagania docelowe (wg ai/02_DEAMON_CORE_CODEBASE.md):
  - Piper TTS z modelem pl_PL-darkman-medium.onnx (models/piper/)
  - Asynchroniczny generator PCM 22050 Hz, 16-bit mono
  - Zero torch, zero CUDA, zero Coqui TTS
"""

import time
from pathlib import Path
from typing import Iterator

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("vox")


class DaemonVox:
    """
    Stub silnika TTS. Pelna implementacja Piper TTS ONNX zostanie wdrozona w Etapie 2.

    Interfejs publiczny (stabilny — nie zmienia sie miedzy etapami):
      - load() -> None
      - is_loaded -> bool
      - stream_chunks(text) -> Iterator[np.ndarray]
      - synthesize_to_file(text, output_path) -> dict
    """

    def __init__(self) -> None:
        self._loaded: bool = False
        log.debug("DaemonVox zainicjalizowany (STUB — Piper TTS niezaimplementowany)")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # --- publiczne API ---

    def load(self) -> None:
        """
        Laduje model Piper TTS ONNX.
        STUB: Loguje ostrzezenie i wraca. Pelna implementacja w Etapie 2.
        """
        piper_model = settings.project_dir / "models" / "piper" / "pl_PL-darkman-medium.onnx"
        if piper_model.exists():
            log.warning(
                "DaemonVox.load(): Model Piper znaleziony (%s), "
                "ale implementacja ONNX nie jest jeszcze wdrozona (Etap 2). "
                "Serwer startuje bez TTS.",
                piper_model,
            )
        else:
            log.warning(
                "DaemonVox.load(): Model Piper nie znaleziony w %s. "
                "Serwer startuje bez TTS.",
                piper_model.parent,
            )
        self._loaded = False

    def stream_chunks(self, text: str) -> Iterator[np.ndarray]:
        """
        Generator zwracajacy kolejne chunki PCM (numpy float32).
        STUB: Zwraca pusta sekwencje — pelna implementacja Piper w Etapie 2.
        """
        log.warning(
            "stream_chunks() wywolany na STUB DaemonVox — brak syntezy (Etap 2 TODO). "
            "Tekst: %r",
            text[:60],
        )
        return iter([])

    def synthesize_to_file(self, text: str, output_path: Path) -> dict:
        """
        Syntetyzuje tekst i zapisuje WAV.
        STUB: Zwraca zerowe metryki — pelna implementacja Piper w Etapie 2.
        """
        log.warning(
            "synthesize_to_file() wywolany na STUB DaemonVox — brak syntezy (Etap 2 TODO). "
            "Tekst: %r",
            text[:60],
        )
        return {
            "latency_first_chunk": 0.0,
            "total_time": 0.0,
            "audio_duration": 0.0,
            "output": str(output_path),
        }
