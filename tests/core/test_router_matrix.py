"""
Testy: Router Safe API — schema validation, reject unknown tools.
Pokrywają:
- Walidacja ToolCallPayload (poprawne i błędne dane)
- ExecActionEvent z request_id
- ExecActionResultEvent
- IntentDecision discriminated union
"""

import json
import pytest

from app.api.schemas import (
    ToolCallPayload,
    ExecActionEvent,
    ExecActionResultEvent,
    IntentDecision,
    VolumeControlIntent,
    AppControlIntent,
    LLMQueryIntent,
    VisionQueryIntent,
    SystemStatusIntent,
)
from pydantic import TypeAdapter, ValidationError


class TestToolCallPayload:

    def test_valid_desktop_action(self):
        payload = ToolCallPayload(name="desktop_action", arguments={"action": "vol_up"})
        assert payload.name == "desktop_action"
        assert payload.arguments["action"] == "vol_up"

    def test_valid_request_frame(self):
        payload = ToolCallPayload(name="request_frame", arguments={})
        assert payload.name == "request_frame"

    def test_invalid_tool_name_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallPayload(name="run_shell", arguments={})

    def test_invalid_tool_name_exec(self):
        with pytest.raises(ValidationError):
            ToolCallPayload(name="exec", arguments={"cmd": "rm -rf /"})

    def test_empty_arguments_allowed(self):
        payload = ToolCallPayload(name="desktop_action")
        assert payload.arguments == {}

    def test_from_json_valid(self):
        raw = '{"name": "desktop_action", "arguments": {"action": "open_browser"}}'
        payload = ToolCallPayload.model_validate_json(raw)
        assert payload.name == "desktop_action"

    def test_from_json_invalid_tool(self):
        raw = '{"name": "format_disk", "arguments": {}}'
        with pytest.raises(ValidationError):
            ToolCallPayload.model_validate_json(raw)


class TestExecActionEvent:

    def test_with_request_id(self):
        ev = ExecActionEvent(
            action="vol_up",
            session_id="win_client",
            request_id="abc123",
        )
        data = json.loads(ev.model_dump_json())
        assert data["event_type"] == "exec_action"
        assert data["request_id"] == "abc123"

    def test_without_request_id(self):
        """Kompatybilność wsteczna — request_id jest opcjonalne."""
        ev = ExecActionEvent(action="vol_up", session_id="win_client")
        data = json.loads(ev.model_dump_json())
        assert data["request_id"] is None

    def test_with_payload(self):
        ev = ExecActionEvent(
            action="open_app",
            payload={"app_name": "chrome"},
            session_id="win_client",
        )
        assert ev.payload["app_name"] == "chrome"


class TestExecActionResultEvent:

    def test_ok_result(self):
        ev = ExecActionResultEvent(
            request_id="abc123",
            status="ok",
            result="Zwiększyłem głośność.",
            session_id="win_client",
        )
        data = json.loads(ev.model_dump_json())
        assert data["event_type"] == "exec_action_result"
        assert data["status"] == "ok"
        assert data["result"] == "Zwiększyłem głośność."

    def test_error_result(self):
        ev = ExecActionResultEvent(
            request_id="xyz789",
            status="error",
            result="Unknown action: format_disk",
            session_id="win_client",
        )
        assert ev.status == "error"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            ExecActionResultEvent(
                request_id="test",
                status="unknown_status",
                result="test",
                session_id="test",
            )


class TestIntentDecisionDiscriminator:

    def test_volume_control_from_json(self):
        adapter = TypeAdapter(IntentDecision)
        raw = '{"intent_type": "VOLUME_CONTROL", "action": "volume_up"}'
        intent = adapter.validate_json(raw)
        assert isinstance(intent, VolumeControlIntent)

    def test_llm_query_from_json(self):
        adapter = TypeAdapter(IntentDecision)
        raw = '{"intent_type": "LLM_QUERY", "query": "Opowiedz mi o Pythonie"}'
        intent = adapter.validate_json(raw)
        assert isinstance(intent, LLMQueryIntent)

    def test_vision_query_from_json(self):
        adapter = TypeAdapter(IntentDecision)
        raw = '{"intent_type": "VISION_QUERY", "query": "Co jest na ekranie?"}'
        intent = adapter.validate_json(raw)
        assert isinstance(intent, VisionQueryIntent)

    def test_invalid_intent_type_rejected(self):
        adapter = TypeAdapter(IntentDecision)
        raw = '{"intent_type": "HACK_SYSTEM", "query": "test"}'
        with pytest.raises(ValidationError):
            adapter.validate_json(raw)
