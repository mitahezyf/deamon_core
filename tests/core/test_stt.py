import builtins
import sys
import types

import numpy as np
import pytest

from app.core.stt import DaemonStt


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


def test_load_disabled_sets_not_loaded(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_enabled", False)
    stt = DaemonStt()
    stt.load()
    assert stt.is_loaded is False


def test_transcribe_requires_loaded_model():
    stt = DaemonStt()
    with pytest.raises(RuntimeError, match="STT is not loaded"):
        stt.transcribe_pcm16((b"\x01\x00") * 1000, sample_rate=16000)


def test_load_enabled_initializes_model(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_enabled", True)
    monkeypatch.setattr("app.core.stt.settings.whisper_model", "tiny")
    monkeypatch.setattr("app.core.stt._cuda_available", lambda: False)

    created = {}

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            created["model_name"] = model_name
            created["device"] = device
            created["compute_type"] = compute_type

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    stt = DaemonStt()
    stt.load()

    assert stt.is_loaded is True
    assert created["model_name"] == "tiny"
    assert created["device"] == "cpu"
    assert created["compute_type"] == "int8"


def test_load_enabled_raises_when_package_missing(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_enabled", True)
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("missing faster_whisper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    stt = DaemonStt()
    with pytest.raises(RuntimeError, match="faster-whisper package is not available"):
        stt.load()


def test_transcribe_skips_too_short_audio(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_min_seconds", 1.0)

    stt = DaemonStt()
    stt._loaded = True
    stt._model = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([], {}))

    text = stt.transcribe_pcm16((b"\x01\x00") * 1000, sample_rate=16000)
    assert text == ""


def test_transcribe_rejects_non_positive_sample_rate(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_min_seconds", 0.01)

    stt = DaemonStt()
    stt._loaded = True
    stt._model = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([], {}))

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        stt.transcribe_pcm16((b"\x01\x00") * 1000, sample_rate=0)


def test_transcribe_empty_audio_returns_empty_text(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_min_seconds", 0.01)

    stt = DaemonStt()
    stt._loaded = True
    stt._model = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([], {}))

    assert stt.transcribe_pcm16(b"", sample_rate=16000) == ""


def test_transcribe_pcm16_joins_segments(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_min_seconds", 0.01)
    monkeypatch.setattr("app.core.stt.settings.language", "pl")

    called = {}

    def fake_transcribe(audio: np.ndarray, **kwargs):
        called["shape"] = audio.shape
        called["dtype"] = str(audio.dtype)
        called["kwargs"] = kwargs
        return iter([_Segment("czesc"), _Segment("swiecie")]), {"language": "pl"}

    stt = DaemonStt()
    stt._loaded = True
    stt._model = types.SimpleNamespace(transcribe=fake_transcribe)

    pcm = (b"\x01\x00") * 8000
    text = stt.transcribe_pcm16(pcm, sample_rate=16000)

    assert text == "czesc swiecie"
    assert called["dtype"] == "float32"
    assert called["shape"] == (8000,)
    assert called["kwargs"]["language"] == "pl"
    assert called["kwargs"]["vad_filter"] is True


def test_transcribe_fallback_without_vad_when_first_pass_empty(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_min_seconds", 0.01)
    monkeypatch.setattr("app.core.stt.settings.language", "pl")

    calls = []

    def fake_transcribe(_audio: np.ndarray, **kwargs):
        calls.append(kwargs["vad_filter"])
        if kwargs["vad_filter"] is True:
            return iter([]), {"language": "pl"}
        return iter([_Segment("dziala")]), {"language": "pl"}

    stt = DaemonStt()
    stt._loaded = True
    stt._model = types.SimpleNamespace(transcribe=fake_transcribe)

    pcm = (b"\x01\x00") * 8000
    text = stt.transcribe_pcm16(pcm, sample_rate=16000)

    assert text == "dziala"
    assert calls == [True, False]


def test_transcribe_resamples_audio_before_model_call(monkeypatch):
    monkeypatch.setattr("app.core.stt.settings.stt_min_seconds", 0.01)
    monkeypatch.setattr("app.core.stt.settings.language", "pl")

    captured = {}

    def fake_transcribe(audio: np.ndarray, **kwargs):
        captured["shape"] = audio.shape
        captured["dtype"] = str(audio.dtype)
        captured["vad_filter"] = kwargs["vad_filter"]
        return iter([_Segment("ok")]), {"language": "pl"}

    stt = DaemonStt()
    stt._loaded = True
    stt._model = types.SimpleNamespace(transcribe=fake_transcribe)

    pcm = (b"\x01\x00") * 8000
    text = stt.transcribe_pcm16(pcm, sample_rate=24000)

    assert text == "ok"
    assert captured["dtype"] == "float32"
    assert captured["shape"] == (5333,)
    assert captured["vad_filter"] is True
