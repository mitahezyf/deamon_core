import pytest
from app.core.vox import DaemonVox

@pytest.mark.asyncio
async def test_vox_synthesize_raw():
    vox = DaemonVox()
    vox.load()
    
    # Pomijamy jesli model nie zaladowany (np. brak plikow na CI)
    if not vox.is_loaded:
        pytest.skip("Model Piper nie jest zaladowany")
        
    pcm_bytes = vox.synthesize_raw("To jest test.")
    
    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) > 0, "Zwrócone dane audio nie powinny być puste"

@pytest.mark.asyncio
async def test_vox_stream_sentences():
    vox = DaemonVox()
    vox.load()
    
    if not vox.is_loaded:
        pytest.skip("Model Piper nie jest zaladowany")
        
    async def text_stream():
        yield "To jest pierwsze zdanie. "
        yield "A to drugie."

    chunks = []
    async for chunk in vox.stream_sentences(text_stream()):
        chunks.append(chunk)
        
    assert len(chunks) > 0, "Powinno wygenerowac przynajmniej jeden chunk audio"
    assert all(isinstance(c, bytes) for c in chunks), "Chunki musza byc bajtami PCM"
