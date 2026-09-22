from typing import AsyncIterator, Optional
import re
import httpx
import json
import asyncio

from config import server_settings
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
        log.info("DaemonBrain loaded (model=%s, url=%s)", server_settings.model_brain, server_settings.ollama_host)

    async def stream_chat(self, prompt: str, image_b64: Optional[str] = None) -> AsyncIterator[str]:
        if not self._loaded:
            raise RuntimeError("LLM is not loaded")

        system_prompt = server_settings.get_brain_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        msg = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]
        messages.append(msg)

        log.debug("Brain: Rozpoczynam streaming odpowiedzi (model: %s)", server_settings.model_brain)
        
        payload = {
            "model": server_settings.model_brain,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            },
            "keep_alive": "15m"
        }
        
        url = f"{server_settings.ollama_host}/api/chat"
        probe_url = f"{server_settings.ollama_host}/api/tags"

        # Asynchroniczny probe diagnostyczny GET /api/tags z timeoutem 2s
        try:
            async with httpx.AsyncClient(timeout=2.0) as probe_client:
                probe_res = await probe_client.get(probe_url)
                if probe_res.status_code == 200:
                    tags_data = probe_res.json()
                    models_list = [m.get("name") for m in tags_data.get("models", [])]
                    log.info("Brain probe [OK]: Ollama aktywna na %s. Dostępne modele: %s", probe_url, models_list)
                else:
                    log.warning("Brain probe: Ollama zwróciła status %s na %s", probe_res.status_code, probe_url)
        except Exception as probe_err:
            log.warning("Brain probe: Brak bezpośredniej odpowiedzi z Ollamy na %s (%s)", probe_url, probe_err)

        ttft_timeout = getattr(server_settings, "brain_ttft_timeout", 60.0)
        timeout_cfg = httpx.Timeout(
            timeout=180.0,
            connect=5.0,
            read=ttft_timeout,
            write=10.0,
            pool=5.0
        )
        
        try:
            async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    full_text = ""
                    buffer = ""
                    in_thinking = False
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                                content = chunk.message.content or ""
                            elif isinstance(chunk, dict):
                                content = chunk.get("message", {}).get("content", "")
                            else:
                                content = ""
                            
                            if content:
                                if "<think>" in content:
                                    in_thinking = True
                                    content = content.replace("<think>", "")
                                if "</think>" in content:
                                    in_thinking = False
                                    content = content.replace("</think>", "")
                                    
                                if in_thinking:
                                    continue
                                
                                buffer += content
                                
                                # Send complete content out of buffer
                                if buffer:
                                    full_text += buffer
                                    yield buffer
                                    buffer = ""

                            if isinstance(chunk, dict) and chunk.get("done", False):
                                if buffer:
                                    full_text += buffer
                                    yield buffer
                                break
                        except json.JSONDecodeError as e:
                            log.error("JSON decode error w brain.py: %s (line: %s)", e, line)
                        except Exception as e:
                            log.error("Nieoczekiwany blad przetwarzania chunka w brain.py: %s", e, exc_info=True)
                                
                    log.info(f"[DAEMON BRAIN ODPOWIEDŹ]: {full_text}")
        except asyncio.CancelledError:
            log.info("Brain: Generowanie przerwane (barge-in / Cancelled).")
            raise
        except httpx.TimeoutException:
            log.warning("Brain: Timeout TTFT (%ss) przy zapytaniu do Ollamy.", ttft_timeout)
            yield "Przepraszam, model nie odpowiedział w wyznaczonym czasie."
        except httpx.ConnectError as e:
            log.error("Brain: Błąd połączenia z Ollamą: %s", e)
            yield "Przepraszam, brak połączenia z serwerem Ollama."
        except Exception as exc:
            log.error("LLM streaming failed: %s", exc, exc_info=True)
            yield "Przepraszam, wystąpił błąd generatora LLM."
