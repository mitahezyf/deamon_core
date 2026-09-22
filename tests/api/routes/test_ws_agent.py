import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json
import struct

from app.api.main import app
from app.api.schemas import SystemCommandIntent, LLMQueryIntent

client = TestClient(app)

@pytest.fixture
def mock_router():
    with patch("app.core.router.DaemonRouter.route", new_callable=AsyncMock) as mock:
        yield mock

def test_ws_agent_system_command(mock_router):
    # Mock zwrotki z routera (komenda systemowa)
    mock_router.return_value = SystemCommandIntent(action="pause")
    
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/agent") as websocket:
            # Stan poczatkowy
            init_state = websocket.receive_json()
            assert init_state["event_type"] == "state_change"
            assert init_state["new_state"] == "STANDBY"
            
            # Wysylamy zadanie uzytkownika
            prompt = {
                "event_type": "user_prompt",
                "text": "stop",
                "session_id": "test_123"
            }
            websocket.send_json(prompt)
            
            # Powinien przejsc przez ROUTING
            state_routing = websocket.receive_json()
            assert state_routing["new_state"] == "ROUTING"
            
            # Powinien odeslac exec_action
            exec_action = websocket.receive_json()
            assert exec_action["event_type"] == "exec_action"
            assert exec_action["action"] == "pause"
            
            # Powrot do STANDBY
            state_standby = websocket.receive_json()
            assert state_standby["new_state"] == "STANDBY"

@patch("app.core.brain.DaemonBrain.stream_chat")
@patch("app.core.vox.DaemonVox.stream_sentences")
def test_ws_agent_llm_query(mock_vox_stream, mock_brain_stream, mock_router):
    # Mock zwrotki z routera (pytanie do LLM)
    mock_router.return_value = LLMQueryIntent(query="powiedz cos")
    
    # Mock asynchronicznych generatorow
    async def mock_text_stream(*args, **kwargs):
        yield "To jest odpowiedz LLM."
        
    async def mock_audio_stream(*args, **kwargs):
        yield b"fake_pcm_data"
        
    mock_brain_stream.return_value = mock_text_stream()
    mock_vox_stream.return_value = mock_audio_stream()
    
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/agent") as websocket:
            websocket.receive_json() # init STANDBY
            
            prompt = {
                "event_type": "user_prompt",
                "text": "powiedz cos",
                "session_id": "test_456"
            }
            websocket.send_json(prompt)
            
            # Zmiany stanow w czasie pracy:
            assert websocket.receive_json()["new_state"] == "ROUTING"
            assert websocket.receive_json()["new_state"] == "INFERENCE_LLM"
            assert websocket.receive_json()["new_state"] == "STREAMING_TTS"
            assert websocket.receive_json()["new_state"] == "SPEAKING"
            
            # Oczekujemy paczki audio (bajtowej)
            pcm_packet = websocket.receive_bytes()
            assert len(pcm_packet) > 4 # naglowek + dane
            
            # Oczekujemy zakonczenia (paczek 0)
            empty_packet = websocket.receive_bytes()
            assert len(empty_packet) == 4
            import struct
            assert struct.unpack(">I", empty_packet)[0] == 0
            
            # Powrot doSTANDBY
            assert websocket.receive_json()["new_state"] == "STANDBY"
