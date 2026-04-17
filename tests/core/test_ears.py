import numpy as np
import pytest

from app.core.ears import DaemonEars


class _DummyDetector:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, _samples: np.ndarray) -> dict[str, float]:
        return self._scores


def test_process_audio_chunk_returns_not_ready_when_unloaded():
    ears = DaemonEars()
    result = ears.process_audio_chunk((b"\x00\x00") * 80)
    assert result["event"] == "ears_not_ready"
    assert result["detected"] is False


def test_process_audio_chunk_returns_empty_chunk():
    ears = DaemonEars()
    ears._loaded = True
    ears._detector = _DummyDetector({"daemon": 0.1})

    result = ears.process_audio_chunk(b"")
    assert result["event"] == "empty_chunk"


def test_process_audio_chunk_detects_wake_word(monkeypatch):
    monkeypatch.setattr("app.core.ears.settings.wake_word_threshold", 0.5)
    monkeypatch.setattr("app.core.ears.settings.wake_word_label", "daemon")

    ears = DaemonEars()
    ears._loaded = True
    ears._detector = _DummyDetector({"daemon": 0.91})

    result = ears.process_audio_chunk((b"\x01\x00") * 160)
    assert result["event"] == "wake_word_detected"
    assert result["detected"] is True
    assert result["score"] >= 0.9


def test_process_audio_chunk_stays_listening_if_other_label(monkeypatch):
    monkeypatch.setattr("app.core.ears.settings.wake_word_threshold", 0.5)
    monkeypatch.setattr("app.core.ears.settings.wake_word_label", "daemon")

    ears = DaemonEars()
    ears._loaded = True
    ears._detector = _DummyDetector({"alexa": 0.99})

    result = ears.process_audio_chunk((b"\x01\x00") * 160)
    assert result["event"] == "listening"
    assert result["detected"] is False


def test_load_raises_for_unsupported_backend(monkeypatch):
    monkeypatch.setattr("app.core.ears.settings.wake_word_backend", "porcupine")

    ears = DaemonEars()
    with pytest.raises(RuntimeError, match="Unsupported wake word backend"):
        ears.load()
