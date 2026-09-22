from typing import AsyncIterator, Optional
import re
import httpx
import json

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
        # Sanityzacja Markdown - usuwa znaki formatowania nie psujac slow
        text = re.sub(r'[*_#`~>]', '', text)
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

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        log.info("DaemonBrain loaded (model=%s, url=%s)", settings.llm_model, settings.ollama_url)

    async def stream_chat(self, prompt: str, image_b64: Optional[str] = None) -> AsyncIterator[str]:
        if not self._loaded:
            raise RuntimeError("LLM is not loaded")

        system_prompt = (
            "Jesteś DAEMON, lokalnym asystentem technicznym. Zwracaj się per 'wodzu'. "
            "ZAWSZE odpowiadaj wyłącznie w języku polskim. Twoje wypowiedzi trafiają bezpośrednio "
            "do syntezatora mowy TTS, dlatego kategorycznie ZAKAZANE jest stosowanie jakiegokolwiek formatowania "
            "Markdown (żadnych gwiazdek, backticków, hashy, myślników wyliczeniowych, bloków kodu ani emotikonów). "
            "Odpowiadaj zwięźle, konkretnie i naturalnym językiem mówionym (maksymalnie 1-2 zdania)."
        )

        messages = [{"role": "system", "content": system_prompt}]
        msg = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]
        messages.append(msg)

        log.debug("Brain: Rozpoczynam streaming odpowiedzi (model: %s)", settings.llm_model)
        
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        url = f"{settings.ollama_url}/api/chat"
        
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=payload, timeout=None) as response:
                    response.raise_for_status()
                    full_text = ""
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                full_text += content
                                yield content
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            log.error("JSON decode error w brain.py: %s (line: %s)", e, line)
                        except Exception as e:
                            log.error("Nieoczekiwany blad przetwarzania chunka w brain.py: %s", e, exc_info=True)
                                
                    log.info(f"[DAEMON BRAIN ODPOWIEDŹ]: {full_text}")
        except Exception as exc:
            log.error("LLM streaming failed: %s", exc, exc_info=True)
            yield "Przepraszam, wystąpił błąd generatora LLM."
