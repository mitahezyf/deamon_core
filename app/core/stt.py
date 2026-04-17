from __future__ import annotations

from typing import Any, Iterable, Protocol, cast

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("stt")
_TARGET_SAMPLE_RATE = 16000


class _SttSegment(Protocol):
    text: str


class _SttModel(Protocol):
    def transcribe(
        self, audio: np.ndarray, **kwargs: Any
    ) -> tuple[Iterable[_SttSegment], Any]: ...


class DaemonStt:
    def __init__(self) -> None:
        self._model: _SttModel | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if not settings.stt_enabled:
            log.info("STT disabled in config")
            self._loaded = False
            return

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover - zalezne od srodowiska
            raise RuntimeError("faster-whisper package is not available") from exc

        compute_type = "float16" if _cuda_available() else "int8"
        self._model = WhisperModel(
            settings.whisper_model,
            device="cuda" if _cuda_available() else "cpu",
            compute_type=compute_type,
        )
        self._loaded = True
        log.info(
            "DaemonStt loaded (model=%s, compute_type=%s)",
            settings.whisper_model,
            compute_type,
        )

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate: int) -> str:
        if not self._loaded or self._model is None:
            raise RuntimeError("STT is not loaded")

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if not audio_bytes:
            return ""

        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return ""

        audio_seconds = pcm.size / float(sample_rate)
        if audio_seconds < settings.stt_min_seconds:
            log.debug("Skipping STT: audio too short (%.3fs)", audio_seconds)
            return ""

        audio = pcm.astype(np.float32) / 32768.0
        if sample_rate != _TARGET_SAMPLE_RATE:
            audio = _resample_audio(audio, sample_rate, _TARGET_SAMPLE_RATE)

        text = self._transcribe_with_vad(audio, use_vad=True)
        if text:
            return text

        # Fallback dla live-mic: czasem VAD odcina cichy start i wynik jest pusty.
        text = self._transcribe_with_vad(audio, use_vad=False)
        if text:
            log.debug("STT fallback bez VAD zwrocil tekst (%.3fs)", audio_seconds)
        else:
            log.debug(
                "STT zwrocil pusty tekst (%.3fs, sr=%d)", audio_seconds, sample_rate
            )
        return text

    def _transcribe_with_vad(self, audio: np.ndarray, use_vad: bool) -> str:
        model = cast(_SttModel, self._model)
        segments, _info = model.transcribe(
            audio,
            language=settings.language,
            vad_filter=use_vad,
            beam_size=1,
            best_of=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resample_audio(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    if audio.size == 0:
        return audio

    # Liniowy resampling jest wystarczający dla krótkich ramek live-mic.
    target_size = max(
        1, int(round(audio.size * float(target_rate) / float(source_rate)))
    )
    source_idx = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_idx = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_idx, source_idx, audio).astype(np.float32)
