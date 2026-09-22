from typing import AsyncIterator, Optional
import re
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("brain")


class SentenceBuffer:
    """Akumuluje strumien tokenow tekstu i wypycha cale zdania."""
    def __init__(self, min_length: int = 15):
        self.buffer = ""
        self.min_length = min_length
        self.boundary_pattern = re.compile(r'([.!?\n]+)')

    def add(self, text: str) -> list[str]:
        """Zwraca liste pelnych zdan wyodrebnionych z bufora."""
        self.buffer += text
        sentences = []
        
        while True:
            matches = list(self.boundary_pattern.finditer(self.buffer))
            if not matches:
                break
                
            valid_match = None
            for match in matches:
                end_pos = match.end()
                sentence_candidate = self.buffer[:end_pos]
                if len(sentence_candidate.strip()) >= self.min_length or '\n' in sentence_candidate:
                    valid_match = match
                    break
                    
            if valid_match:
                end_pos = valid_match.end()
                sentences.append(self.buffer[:end_pos].strip())
                self.buffer = self.buffer[end_pos:]
            else:
                break
                
        return sentences

    def flush(self) -> Optional[str]:
        """Zwraca reszte z bufora (np. przy koncu generowania)."""
        sentence = self.buffer.strip()
        self.buffer = ""
        if sentence:
            return sentence
        return None


class DaemonBrain:
    def __init__(self) -> None:
        self._loaded = False
        self.client: Optional[AsyncOpenAI] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self.client = AsyncOpenAI(
            base_url=f"{settings.ollama_url}/v1",
            api_key="ollama"  # Wymagane przez biblioteke openai, mimo ze Ollama tego nie uzywa
        )
        self._loaded = True
        log.info("DaemonBrain loaded (model=%s, url=%s)", settings.llm_model, settings.ollama_url)

    async def stream_chat(self, prompt: str, image_b64: Optional[str] = None) -> AsyncIterator[str]:
        if not self._loaded or not self.client:
            raise RuntimeError("LLM is not loaded")

        messages = []
        if image_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        log.debug("Brain: Rozpoczynam streaming odpowiedzi (model: %s)", settings.llm_model)
        
        try:
            stream = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            log.error("LLM streaming failed: %s", exc)
            yield "Przepraszam, wystąpił błąd generatora LLM."
