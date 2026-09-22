import pytest
from app.core.brain import SentenceBuffer, DaemonBrain

def test_sentence_buffer():
    sb = SentenceBuffer(min_length=15)
    
    # Krotki tekst, nie powinien wypchnac
    sentences = sb.add("Krotkie. ")
    assert len(sentences) == 0
    
    # Dodajemy dluzszy tekst z kropka
    sentences = sb.add("To jest znacznie dluzsze zdanie.")
    assert len(sentences) == 1
    assert sentences[0] == "Krotkie. To jest znacznie dluzsze zdanie."
    
    # Test nowej linii
    sentences = sb.add("Krotki\n")
    assert len(sentences) == 1
    assert sentences[0] == "Krotki"
    
    # Flush
    sentences = sb.add("Reszta")
    assert len(sentences) == 0
    assert sb.flush() == "Reszta"
    assert sb.flush() is None

from unittest.mock import patch

@pytest.mark.asyncio
@patch("app.core.brain.AsyncOpenAI")
async def test_brain_stream_chat_mocked(mock_openai_cls):
    class MockDelta:
        def __init__(self, content):
            self.content = content
            
    class MockChoice:
        def __init__(self, content):
            self.delta = MockDelta(content)
            
    class MockChunk:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    async def mock_stream():
        yield MockChunk("Cześć! ")
        yield MockChunk("Oto test ")
        yield MockChunk("streamingu.")

    async def mock_create(*args, **kwargs):
        return mock_stream()

    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create = mock_create

    brain = DaemonBrain()
    brain.load()
    
    tokens = []
    async for token in brain.stream_chat("Test", image_b64=None):
        tokens.append(token)
        
    assert "".join(tokens) == "Cześć! Oto test streamingu."
