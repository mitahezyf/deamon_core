from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.core.config import settings
from app.core.vox import DaemonVox


class _FakeModel:
    def __init__(self, *, half_raises: bool = False) -> None:
        self.device = "cpu"
        self._half_raises = half_raises
        self.latent_calls = 0

    def eval(self) -> None:
        return None

    def half(self) -> None:
        if self._half_raises:
            raise RuntimeError("fp16 unsupported")

    def float(self) -> None:
        return None

    def get_conditioning_latents(self, audio_path):
        self.latent_calls += 1
        value = float(self.latent_calls)
        return torch.tensor([value]), torch.tensor([value + 0.5])

    def inference_stream(self, **_kwargs):
        for _ in range(3):
            yield torch.zeros(1024)


class _FakeTTS:
    def __init__(self, model: _FakeModel) -> None:
        self.synthesizer = SimpleNamespace(tts_model=model)

    def to(self, _device: str) -> "_FakeTTS":
        return self


@pytest.fixture
def vox_fs(tmp_path, monkeypatch):
    samples_dir = tmp_path / "voice_samples"
    samples_dir.mkdir()
    (samples_dir / "sample.wav").write_bytes(b"RIFF")

    cache_path = tmp_path / "voice_cache.pth"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(settings, "samples_dir", samples_dir)
    monkeypatch.setattr(settings, "cache_path", cache_path)
    monkeypatch.setattr(settings, "output_dir", output_dir)

    return SimpleNamespace(
        samples_dir=samples_dir, cache_path=cache_path, output_dir=output_dir
    )


def _patch_tts(mocker, *, half_raises: bool = False):
    created_models: list[_FakeModel] = []

    def factory(_model_name: str):
        model = _FakeModel(half_raises=half_raises)
        created_models.append(model)
        return _FakeTTS(model)

    mocker.patch("app.core.vox.TTS", side_effect=factory)
    return created_models


def test_vox_initialization_device():
    vox = DaemonVox()
    assert vox._device in ["cuda", "cpu"]


def test_warmup_fails_without_load():
    vox = DaemonVox()
    with pytest.raises(RuntimeError, match=r"Najpierw wywołaj DaemonVox.load\(\)"):
        vox.warmup()


def test_load_builds_cache_when_missing(mocker, vox_fs):
    models = _patch_tts(mocker)
    vox = DaemonVox()

    vox.load()

    assert vox_fs.cache_path.exists()
    assert vox._tts is not None
    assert models[0].latent_calls == 1
    assert torch.equal(vox._gpt_cond_latent, torch.tensor([1.0]))
    assert torch.equal(vox._speaker_embedding, torch.tensor([1.5]))


def test_load_uses_existing_cache_without_rebuild(mocker, vox_fs):
    models = _patch_tts(mocker)
    torch.save(
        {
            "gpt_cond_latent": torch.tensor([7.0]),
            "speaker_embedding": torch.tensor([8.0]),
        },
        vox_fs.cache_path,
    )

    vox = DaemonVox()
    vox.load()

    assert models[0].latent_calls == 0
    assert torch.equal(vox._gpt_cond_latent, torch.tensor([7.0]))
    assert torch.equal(vox._speaker_embedding, torch.tensor([8.0]))


def test_rebuild_cache_recomputes_latents(mocker, vox_fs):
    models = _patch_tts(mocker)
    torch.save(
        {
            "gpt_cond_latent": torch.tensor([0.0]),
            "speaker_embedding": torch.tensor([0.0]),
        },
        vox_fs.cache_path,
    )

    vox = DaemonVox()
    vox.load()
    vox.rebuild_cache()

    assert models[0].latent_calls == 1
    assert torch.equal(vox._gpt_cond_latent, torch.tensor([1.0]))
    assert torch.equal(vox._speaker_embedding, torch.tensor([1.5]))


def test_rebuild_cache_works_without_prior_load(mocker, vox_fs):
    models = _patch_tts(mocker)
    vox = DaemonVox()

    vox.rebuild_cache()

    assert vox_fs.cache_path.exists()
    assert models[0].latent_calls == 1
    assert torch.equal(vox._gpt_cond_latent, torch.tensor([1.0]))


def test_stream_chunks_returns_numpy_arrays(mocker, vox_fs):
    _patch_tts(mocker)
    vox = DaemonVox()
    vox.load()

    chunks = list(vox.stream_chunks("Witaj świecie"))

    assert len(chunks) == 3
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)


def test_synthesize_to_file_writes_audio_and_metrics(mocker, vox_fs):
    _patch_tts(mocker)
    vox = DaemonVox()
    vox.load()

    output_path = vox_fs.output_dir / "test_out.wav"
    result = vox.synthesize_to_file("Tekst testowy", output_path)

    assert output_path.exists()
    assert result["output"] == str(output_path)
    assert result["latency_first_chunk"] >= 0.0
    assert result["total_time"] >= 0.0
    assert result["audio_duration"] > 0.0


def test_load_fails_when_no_voice_samples(mocker, tmp_path, monkeypatch):
    _patch_tts(mocker)
    empty_samples_dir = tmp_path / "empty_samples"
    empty_samples_dir.mkdir()

    monkeypatch.setattr(settings, "samples_dir", empty_samples_dir)
    monkeypatch.setattr(settings, "cache_path", tmp_path / "voice_cache.pth")

    vox = DaemonVox()
    with pytest.raises(FileNotFoundError, match="Brak plików .wav"):
        vox.load()


def test_load_handles_fp16_fallback(mocker, vox_fs):
    _patch_tts(mocker, half_raises=True)
    vox = DaemonVox()

    vox.load()

    assert vox._tts is not None
