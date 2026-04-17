def test_health_endpoint(test_client, vox_mock):
    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["vox_loaded"] is True
    assert "device" in data
    assert "llm_model" in data
    assert data["api_port"] == 8000
    vox_mock.load.assert_called_once()
    vox_mock.warmup.assert_called_once()


def test_synthesize_endpoint(test_client, vox_mock, mocker, tmp_path):
    # Podmieniamy domyślny folder testów na tymczasowy
    mocker.patch("app.api.routes.http.settings.output_dir", tmp_path)

    payload = {
        "text": "Witaj świecie!",
        "output": "test_output.wav",
    }

    response = test_client.post("/synthesize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "latency_first_chunk" in data
    assert "total_time" in data
    assert data["output"].endswith("test_output.wav")

    # Upewniamy się, że plik został stworzony
    assert (tmp_path / "test_output.wav").exists()
    vox_mock.synthesize_to_file.assert_called_once()
    _, called_output = vox_mock.synthesize_to_file.call_args.args
    assert called_output == tmp_path / "test_output.wav"
