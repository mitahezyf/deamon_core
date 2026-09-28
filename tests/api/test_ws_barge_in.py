"""
Testy: barge-in / abort podczas tool_call, przerwanie mid-stream.
Pokrywają:
- Anulowanie zadania w trakcie generowania (CancelledError)
- Bezpieczne zamykanie WebSocket po abort
- ClientSession cleanup z tool_result_futures
"""

import asyncio
import json
import pytest

from app.api.routes.ws import ClientSession, safe_send_text, safe_send_bytes
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.websockets import WebSocketState


class TestClientSessionBargeIn:

    def test_session_has_tool_result_futures(self):
        """ClientSession posiada słownik tool_result_futures."""
        ws_mock = MagicMock()
        session = ClientSession(ws_mock, "192.168.0.1")
        assert hasattr(session, "tool_result_futures")
        assert isinstance(session.tool_result_futures, dict)

    @pytest.mark.asyncio
    async def test_disconnect_and_cancel_clears_task(self):
        """disconnect_and_cancel anuluje zadanie i zamyka WebSocket."""
        ws_mock = MagicMock()
        ws_mock.client_state = WebSocketState.CONNECTED
        ws_mock.close = AsyncMock()

        session = ClientSession(ws_mock, "test_client")

        # Symulujemy zadanie, które da się anulować
        async def long_running():
            await asyncio.sleep(100)

        session.current_task = asyncio.create_task(long_running())
        await session.disconnect_and_cancel()

        assert session.current_task.cancelled() or session.current_task.done()
        ws_mock.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_result_future_resolved(self):
        """Future w tool_result_futures jest poprawnie rozwiązywany."""
        ws_mock = MagicMock()
        session = ClientSession(ws_mock, "test_client")

        future = asyncio.get_event_loop().create_future()
        session.tool_result_futures["req_001"] = future

        # Symulacja odpowiedzi z klienta
        future.set_result("Zwiększyłem głośność.")
        result = await future
        assert result == "Zwiększyłem głośność."

    @pytest.mark.asyncio
    async def test_tool_result_future_timeout(self):
        """Timeout Future rzuca TimeoutError."""
        ws_mock = MagicMock()
        session = ClientSession(ws_mock, "test_client")

        future = asyncio.get_event_loop().create_future()
        session.tool_result_futures["req_002"] = future

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, timeout=0.1)


class TestSafeSendGuards:

    @pytest.mark.asyncio
    async def test_safe_send_text_disconnected(self):
        """safe_send_text zwraca False dla zamkniętego socketa."""
        ws_mock = MagicMock()
        ws_mock.client_state = WebSocketState.DISCONNECTED
        result = await safe_send_text(ws_mock, "test")
        assert result is False

    @pytest.mark.asyncio
    async def test_safe_send_text_connected(self):
        """safe_send_text zwraca True dla aktywnego socketa."""
        ws_mock = MagicMock()
        ws_mock.client_state = WebSocketState.CONNECTED
        ws_mock.send_text = AsyncMock()
        result = await safe_send_text(ws_mock, "test")
        assert result is True
        ws_mock.send_text.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_safe_send_bytes_disconnected(self):
        """safe_send_bytes zwraca False dla zamkniętego socketa."""
        ws_mock = MagicMock()
        ws_mock.client_state = WebSocketState.DISCONNECTED
        result = await safe_send_bytes(ws_mock, b"test")
        assert result is False

    @pytest.mark.asyncio
    async def test_safe_send_text_runtime_error(self):
        """safe_send_text obsługuje RuntimeError gracefully."""
        ws_mock = MagicMock()
        ws_mock.client_state = WebSocketState.CONNECTED
        ws_mock.send_text = AsyncMock(side_effect=RuntimeError("closed"))
        result = await safe_send_text(ws_mock, "test")
        assert result is False
