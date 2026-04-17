import struct

import pytest
from starlette.websockets import WebSocketDisconnect


def test_ws_synthesize(test_client):
    with test_client.websocket_connect("/ws/synthesize") as websocket:
        # Wysyłamy wiadomość
        websocket.send_text("Witaj świecie!")

        # Odbieramy 3 chunki z audio (zgodnie z naszym mockiem)
        for _ in range(3):
            data = websocket.receive_bytes()
            # Sprawdzamy nagłówek i to, że przysłano jakieś dane (rozmiar > 0)
            length = struct.unpack(">I", data[:4])[0]
            assert length > 0
            assert len(data) == 4 + length

        # Na koniec powinien przyjść pusty pakiet o rozmiarze 0
        end_data = websocket.receive_bytes()
        end_length = struct.unpack(">I", end_data[:4])[0]
        assert end_length == 0
        assert len(end_data) == 4


def test_ws_ignores_empty_messages(test_client):
    with test_client.websocket_connect("/ws/synthesize") as websocket:
        # Wysyłamy pustą wiadomość, serwer powinien ją zignorować
        # bez zamykania połączenia i bez wysyłania pustego pakietu stopu
        websocket.send_text("   ")

        # Testujemy czy połączenie jest dalej otwarte, wysyłając prawdziwą wiadomość
        websocket.send_text("Teraz tak")
        data = websocket.receive_bytes()
        length = struct.unpack(">I", data[:4])[0]
        assert length > 0


def test_ws_closes_with_internal_error_code(test_client, vox_mock):
    vox_mock.stream_chunks.side_effect = RuntimeError("boom")

    with test_client.websocket_connect("/ws/synthesize") as websocket:
        websocket.send_text("Wymuś błąd")
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_bytes()

    assert exc.value.code == 1011
