import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

os.environ["DAEMON_DEBUG_MODE"] = "true"


@pytest.fixture
def vox_mock(mocker):
    """Mock DaemonVox używany przez lifespan FastAPI."""
    mock_vox_class = mocker.patch("app.api.main.DaemonVox", autospec=True)
    mock_instance = mock_vox_class.return_value

    mock_instance._tts = object()
    mock_instance._device = "cpu"

    def dummy_stream(_text):
        for _ in range(3):
            yield np.zeros(1024, dtype=np.float32)

    mock_instance.stream_chunks.side_effect = dummy_stream

    def dummy_synthesize(_text, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF")
        return {
            "latency_first_chunk": 0.1,
            "total_time": 0.3,
            "audio_duration": 1.5,
            "output": str(output_path),
        }

    mock_instance.synthesize_to_file.side_effect = dummy_synthesize
    return mock_instance


@pytest.fixture
def test_client(vox_mock):
    """Klient HTTP/WS oparty o realny lifecycle aplikacji."""
    with TestClient(app) as client:
        yield client
