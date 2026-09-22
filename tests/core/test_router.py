import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.core.router import DaemonRouter
from app.api.schemas import SystemCommandIntent, VisionQueryIntent, LLMQueryIntent

@pytest.mark.asyncio
async def test_router_heuristic():
    router = DaemonRouter()
    
    # "stop" -> pause
    intent = await router.route("stop")
    assert isinstance(intent, SystemCommandIntent)
    assert intent.action == "pause"
    
    # "głośniej" -> volume_up
    intent = await router.route("głośniej")
    assert isinstance(intent, SystemCommandIntent)
    assert intent.action == "volume_up"

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_router_llm_parsing(mock_post):
    # Mock udanej odpowiedzi Ollamy z JSON zgodnym ze schematem VisionQuery
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"intent_type": "VisionQuery", "query": "co widzisz na ekranie?"}'
            }
        }]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    router = DaemonRouter()
    intent = await router.route("co widzisz na ekranie?")
    
    assert isinstance(intent, VisionQueryIntent)
    assert intent.query == "co widzisz na ekranie?"

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_router_fallback_timeout(mock_post):
    # Mock Timeout Exception
    mock_post.side_effect = httpx.TimeoutException("Symulacja timeoutu")
    
    router = DaemonRouter()
    intent = await router.route("opowiedz mi bajke")
    
    assert isinstance(intent, LLMQueryIntent)
    assert intent.query == "opowiedz mi bajke"

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_router_fallback_bad_json(mock_post):
    # Mock zlego JSONa (brak pola action lub zly intent_type)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"intent_type": "NieznanyTyp"}'
            }
        }]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    router = DaemonRouter()
    intent = await router.route("bla bla bla")
    
    assert isinstance(intent, LLMQueryIntent)
    assert intent.query == "bla bla bla"
