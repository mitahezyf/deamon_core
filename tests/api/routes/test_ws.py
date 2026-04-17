import struct

import pytest
from starlette.websockets import WebSocketDisconnect


def test_ws_synthesize(test_client):
    with test_client.websocket_connect("/ws/synthesize") as websocket:
        # Wysyamy wiadomosc
        websocket.send_text("Witaj swiecie!")

        # Odbieramy 3 chunki z audio (zgodnie z naszym mockiem)
        for _ in range(3):
            data = websocket.receive_bytes()
            # Sprawdzamy nagowek i to, ze przysano jakies dane (rozmiar > 0)
            length = struct.unpack(">I", data[:4])[0]
            assert length > 0
            assert len(data) == 4 + length

        # Na koniec powinien przyjsc pusty pakiet o rozmiarze 0
        end_data = websocket.receive_bytes()
        end_length = struct.unpack(">I", end_data[:4])[0]
        assert end_length == 0
        assert len(end_data) == 4


def test_ws_ignores_empty_messages(test_client):
    with test_client.websocket_connect("/ws/synthesize") as websocket:
        # Wysyamy pusta wiadomosc, serwer powinien ja zignorowac
        # bez zamykania poaczenia i bez wysyania pustego pakietu stopu
        websocket.send_text("   ")

        # Testujemy czy poaczenie jest dalej otwarte, wysyajac prawdziwa wiadomosc
        websocket.send_text("Teraz tak")
        data = websocket.receive_bytes()
        length = struct.unpack(">I", data[:4])[0]
        assert length > 0


def test_ws_closes_with_internal_error_code(test_client, vox_mock):
    vox_mock.stream_chunks.side_effect = RuntimeError("boom")

    with test_client.websocket_connect("/ws/synthesize") as websocket:
        websocket.send_text("Wymus bad")
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_bytes()

    assert exc.value.code == 1011


def test_ws_ears_listen_returns_ready_and_listening_event(test_client):
    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        ready = websocket.receive_json()
        assert ready["event"] == "ready"

        websocket.send_bytes((b"\x00\x00") * 160)
        event = websocket.receive_json()
        assert event["event"] == "listening"
        assert event["detected"] is False


def test_ws_ears_start_stop_capture_returns_transcript(test_client, stt_mock):
    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("start_capture")
        started = websocket.receive_json()
        assert started["event"] == "capture_started"

        chunk = (b"\x01\x00") * 1600
        websocket.send_bytes(chunk)
        listening = websocket.receive_json()
        assert listening["event"] == "listening"

        websocket.send_text("stop_capture")
        transcript = websocket.receive_json()
        assert transcript["event"] == "transcript"
        assert transcript["text"] == "testowa transkrypcja"
        stt_mock.transcribe_pcm16.assert_called_once()


def test_ws_ears_invalid_sample_rate_returns_error(test_client):
    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("set_sample_rate:abc")
        event = websocket.receive_json()
        assert event["event"] == "stt_error"
        assert event["message"] == "invalid_sample_rate"


def test_ws_ears_valid_sample_rate_returns_ack(test_client):
    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("set_sample_rate:22050")
        event = websocket.receive_json()
        assert event["event"] == "sample_rate_set"
        assert event["sample_rate"] == 22050


def test_ws_ears_flush_without_audio_returns_empty_transcript(test_client):
    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("flush_stt")
        event = websocket.receive_json()
        assert event["event"] == "transcript"
        assert event["text"] == ""


def test_ws_ears_unknown_command(test_client):
    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("nieznana_komenda")
        event = websocket.receive_json()
        assert event["event"] == "unknown_command"
        assert event["command"] == "nieznana_komenda"


def test_ws_ears_stop_capture_when_stt_not_loaded_returns_error(test_client, stt_mock):
    stt_mock.is_loaded = False

    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("start_capture")
        websocket.receive_json()
        websocket.send_bytes((b"\x01\x00") * 1600)
        websocket.receive_json()
        websocket.send_text("stop_capture")
        event = websocket.receive_json()

    assert event["event"] == "stt_error"
    assert event["message"] == "stt_not_ready"
    assert event["sample_rate"] == 16000


def test_ws_ears_stop_capture_returns_stt_error_on_exception(test_client, stt_mock):
    stt_mock.transcribe_pcm16.side_effect = RuntimeError("stt boom")

    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_text("start_capture")
        websocket.receive_json()
        websocket.send_bytes((b"\x01\x00") * 1600)
        websocket.receive_json()
        websocket.send_text("stop_capture")
        event = websocket.receive_json()

    assert event["event"] == "stt_error"
    assert "stt boom" in event["message"]


def test_ws_ears_auto_capture_starts_after_wake_word(
    test_client, ears_mock, monkeypatch
):
    monkeypatch.setattr("app.api.routes.ws.settings.stt_enabled", True)

    ears_mock.process_audio_chunk.side_effect = [
        {
            "event": "wake_word_detected",
            "detected": True,
            "label": "daemon",
            "score": 0.91,
            "threshold": 0.5,
            "stt_enabled": True,
        },
        {
            "event": "listening",
            "detected": False,
            "label": "daemon",
            "score": 0.11,
            "threshold": 0.5,
            "stt_enabled": True,
        },
    ]

    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_bytes((b"\x01\x00") * 160)
        first = websocket.receive_json()
        started = websocket.receive_json()
        websocket.send_bytes((b"\x01\x00") * 160)
        second = websocket.receive_json()
        websocket.send_text("stop_capture")
        transcript = websocket.receive_json()

    assert first["event"] == "wake_word_detected"
    assert started["event"] == "capture_started"
    assert started["source"] == "wake_word"
    assert second["event"] == "listening"
    assert transcript["event"] == "transcript"


def test_ws_ears_closes_with_internal_error_when_ears_processing_crashes(
    test_client, ears_mock
):
    ears_mock.process_audio_chunk.side_effect = RuntimeError("ears boom")

    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_bytes((b"\x01\x00") * 160)
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_json()

    assert exc.value.code == 1011


def test_ws_ears_listen_detects_wake_word(test_client, ears_mock):
    ears_mock.process_audio_chunk.return_value = {
        "event": "wake_word_detected",
        "detected": True,
        "label": "daemon",
        "score": 0.92,
        "threshold": 0.5,
        "stt_enabled": False,
    }

    with test_client.websocket_connect("/ws/ears/listen") as websocket:
        websocket.receive_json()
        websocket.send_bytes((b"\x01\x00") * 160)
        event = websocket.receive_json()

    assert event["event"] == "wake_word_detected"
    assert event["detected"] is True
    assert event["label"] == "daemon"
