from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("ears")


class WakeWordDetector(Protocol):
    def predict(self, samples: np.ndarray) -> dict[str, float]: ...


@dataclass
class WakeWordEvent:
    event: str
    detected: bool
    label: str
    score: float
    threshold: float
    stt_enabled: bool

    def as_dict(self) -> dict:
        return {
            "event": self.event,
            "detected": self.detected,
            "label": self.label,
            "score": self.score,
            "threshold": self.threshold,
            "stt_enabled": self.stt_enabled,
        }


class _OpenWakeWordAdapter:
    def __init__(self, model_path: Path | None) -> None:
        try:
            from openwakeword.model import Model
        except (
            Exception
        ) as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("openwakeword package is not available") from exc

        kwargs: dict = {}
        # ONNX jest stabilniejszym wyborem na Windows, bo tflite-runtime
        # czesto nie jest dostepny w tym srodowisku.
        kwargs["inference_framework"] = "onnx"

        if not (model_path and model_path.exists()):
            try:
                import openwakeword
                from openwakeword.utils import download_models

                pretrained = openwakeword.get_pretrained_model_paths("onnx")
                if any(not Path(path).exists() for path in pretrained):
                    log.info("Brak zasobow openwakeword - pobieram modele domyslne...")
                    download_models()
            except Exception as exc:  # pragma: no cover - zalezne od runtime/network
                log.warning("Nie udalo sie przygotowac modeli openwakeword: %s", exc)

        if model_path and model_path.exists():
            kwargs["wakeword_models"] = [str(model_path)]
            log.info("openwakeword using custom model: %s", model_path)
        else:
            log.info("openwakeword using default built-in models (onnx)")

        self._model = Model(**kwargs)

    def predict(self, samples: np.ndarray) -> dict[str, float]:
        raw = self._model.predict(samples)
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        if isinstance(raw, list) and raw and isinstance(raw[-1], dict):
            return {str(k): float(v) for k, v in raw[-1].items()}
        raise RuntimeError("Unsupported openwakeword output format")


class DaemonEars:
    def __init__(self) -> None:
        self._detector: WakeWordDetector | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if not settings.wake_word_enabled:
            log.info("Wake word disabled in config")
            self._loaded = False
            return

        if settings.wake_word_backend != "openwakeword":
            raise RuntimeError(
                f"Unsupported wake word backend: {settings.wake_word_backend}"
            )

        self._detector = _OpenWakeWordAdapter(settings.openwakeword_model_path)
        self._loaded = True
        log.info("DaemonEars loaded (backend=%s)", settings.wake_word_backend)

    def process_audio_chunk(self, audio_bytes: bytes) -> dict:
        if not self._loaded or self._detector is None:
            return {
                "event": "ears_not_ready",
                "detected": False,
                "label": settings.wake_word_label,
                "score": 0.0,
                "threshold": settings.wake_word_threshold,
                "stt_enabled": settings.stt_enabled,
            }

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return {
                "event": "empty_chunk",
                "detected": False,
                "label": settings.wake_word_label,
                "score": 0.0,
                "threshold": settings.wake_word_threshold,
                "stt_enabled": settings.stt_enabled,
            }
        samples /= 32768.0

        scores = self._detector.predict(samples)
        if not scores:
            return {
                "event": "listening",
                "detected": False,
                "label": settings.wake_word_label,
                "score": 0.0,
                "threshold": settings.wake_word_threshold,
                "stt_enabled": settings.stt_enabled,
            }

        label, score = max(scores.items(), key=lambda item: item[1])
        wanted = settings.wake_word_label.lower()
        is_target = wanted in label.lower()
        detected = bool(is_target and score >= settings.wake_word_threshold)

        event = WakeWordEvent(
            event="wake_word_detected" if detected else "listening",
            detected=detected,
            label=label,
            score=round(float(score), 4),
            threshold=settings.wake_word_threshold,
            stt_enabled=settings.stt_enabled,
        )
        return event.as_dict()
