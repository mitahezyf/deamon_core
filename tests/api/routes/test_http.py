import base64


def test_health_endpoint(test_client, vox_mock):
    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["vox_loaded"] is True
    assert data["ears_loaded"] is True
    assert "device" in data
    assert "llm_model" in data
    assert data["api_port"] == 8000
    vox_mock.load.assert_called_once()
    vox_mock.warmup.assert_called_once()


def test_status_endpoint(test_client):
    response = test_client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["vox_loaded"] is True
    assert data["ears_loaded"] is True
    assert data["device"] in ["cpu", "cuda"]
    assert data["language"] == "pl"
    assert data["stt_enabled"] is False
    assert data["stt_loaded"] is True
    assert data["stt_sample_rate"] == 16000
    assert data["wake_word_enabled"] is True
    assert data["wake_word_backend"] == "openwakeword"
    assert data["wake_word_label"] == "daemon"
    assert data["api_port"] == 8000


def test_public_config_endpoint(test_client):
    response = test_client.get("/config/public")
    assert response.status_code == 200

    data = response.json()
    assert data["llm_model"]
    assert data["ollama_url"].startswith("http")
    assert data["language"] == "pl"
    assert data["tts_model"]
    assert data["stt_enabled"] is False
    assert data["stt_sample_rate"] == 16000
    assert data["wake_word_enabled"] is True
    assert data["wake_word_backend"] == "openwakeword"
    assert data["wake_word_label"] == "daemon"
    assert data["wake_word_model"].endswith("daemon_windows.ppn")
    assert data["openwakeword_model_path"].endswith("daemon.onnx")


def test_synthesize_endpoint(test_client, vox_mock, mocker, tmp_path):
    # Podmieniamy domyslny folder testow na tymczasowy
    mocker.patch("app.api.routes.http.settings.output_dir", tmp_path)

    payload = {
        "text": "Witaj swiecie!",
        "output": "test_output.wav",
    }

    response = test_client.post("/synthesize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "latency_first_chunk" in data
    assert "total_time" in data
    assert data["output"].endswith("test_output.wav")

    # Upewniamy sie, ze plik zosta stworzony
    assert (tmp_path / "test_output.wav").exists()
    vox_mock.synthesize_to_file.assert_called_once()
    _, called_output = vox_mock.synthesize_to_file.call_args.args
    assert called_output == tmp_path / "test_output.wav"


def test_transcribe_endpoint(test_client, stt_mock):
    pcm = (b"\x01\x00") * 8000
    payload = {
        "audio_b64": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": 16000,
    }

    response = test_client.post("/transcribe", json=payload)
    assert response.status_code == 200
    assert response.json()["text"] == "testowa transkrypcja"
    stt_mock.transcribe_pcm16.assert_called_once_with(pcm, sample_rate=16000)


def test_transcribe_endpoint_rejects_invalid_base64(test_client):
    payload = {
        "audio_b64": "@@@",
        "sample_rate": 16000,
    }

    response = test_client.post("/transcribe", json=payload)
    assert response.status_code == 400


def test_transcribe_endpoint_returns_empty_when_stt_not_loaded(test_client, stt_mock):
    stt_mock.is_loaded = False
    payload = {
        "audio_b64": base64.b64encode((b"\x00\x00") * 100).decode("ascii"),
        "sample_rate": 16000,
    }

    response = test_client.post("/transcribe", json=payload)
    assert response.status_code == 200
    assert response.json()["text"] == ""
