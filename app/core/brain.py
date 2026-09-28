from typing import AsyncIterator, Optional, Callable, Awaitable
import re
import httpx
import json
import asyncio

from config import server_settings
from app.core.logger import get_logger

log = get_logger("brain")

# ============================================================================
# Safe API Whitelist — ściśle zmapowana 1:1 z client_actions.py
# ============================================================================
ALLOWED_TOOLS = {"desktop_action", "request_frame"}
ALLOWED_ACTIONS = frozenset([
    "vol_up", "vol_down", "vol_mute", "vol_unmute",
    "open_browser", "open_calc", "open_notepad",
    "open_taskmgr", "open_powershell", "lock_system",
    "get_time", "get_date", "get_sys_status",
])


# ============================================================================
# TagAwareBuffer — maszyna stanów filtrująca <tool_call> i <think> ze strumienia
# ============================================================================
class TagAwareBuffer:
    """
    Bufor z maszyną stanów rozstrzygającą, czy napływające tokeny to:
    - zwykły tekst (przepuszczany do TTS),
    - znacznik <tool_call>…</tool_call> (wyodrębniony jako JSON),
    - znacznik <think>…</think> (odrzucany w ciszy).

    Obsługuje poszatkowane tokeny SSE (np. "<tool_" + "call>").
    Żaden fragment XML/JSON nie trafia do silnika mowy.
    """

    _OPEN_TAGS = ("<tool_call>", "<think>")

    def __init__(self):
        self.state: str = "NORMAL"       # NORMAL | IN_TOOL_CALL | IN_THINK
        self._tag_candidate: str = ""     # bufor kandydujących znaków otwierającego tagu
        self._tool_call_buf: str = ""     # treść wewnątrz <tool_call>…</tool_call>
        self._think_buf: str = ""         # bufor wewnątrz <think> (do detekcji zamknięcia)

    # ---- Prywatna metoda: czy `candidate` jest prefixem dowolnego OPEN tagu ----
    def _is_prefix_of_any(self, candidate: str) -> bool:
        return any(tag.startswith(candidate) for tag in self._OPEN_TAGS)

    def feed(self, text: str) -> tuple[str, list[str]]:
        """
        Przetwarza fragment tokenu.
        Zwraca (clean_text_for_tts, list_of_tool_call_json_strings).
        """
        clean_parts: list[str] = []
        tool_calls: list[str] = []

        for ch in text:
            if self.state == "NORMAL":
                if ch == "<" and not self._tag_candidate:
                    # Potencjalny początek tagu — zawieszamy wypychanie
                    self._tag_candidate = "<"
                elif self._tag_candidate:
                    self._tag_candidate += ch
                    if self._is_prefix_of_any(self._tag_candidate):
                        # Nadal pasuje do prefiksu — kontynuujemy buforowanie
                        if self._tag_candidate == "<tool_call>":
                            self.state = "IN_TOOL_CALL"
                            self._tag_candidate = ""
                            self._tool_call_buf = ""
                        elif self._tag_candidate == "<think>":
                            self.state = "IN_THINK"
                            self._tag_candidate = ""
                            self._think_buf = ""
                    else:
                        # Nie pasuje — redukujemy od lewej, aby nie zgubić np. drugiego '<'
                        while self._tag_candidate and not self._is_prefix_of_any(self._tag_candidate):
                            clean_parts.append(self._tag_candidate[0])
                            self._tag_candidate = self._tag_candidate[1:]
                        if self._tag_candidate:
                            if self._tag_candidate == "<tool_call>":
                                self.state = "IN_TOOL_CALL"
                                self._tag_candidate = ""
                                self._tool_call_buf = ""
                            elif self._tag_candidate == "<think>":
                                self.state = "IN_THINK"
                                self._tag_candidate = ""
                                self._think_buf = ""
                else:
                    clean_parts.append(ch)

            elif self.state == "IN_TOOL_CALL":
                self._tool_call_buf += ch
                if self._tool_call_buf.endswith("</tool_call>"):
                    json_str = self._tool_call_buf[: -len("</tool_call>")].strip()
                    if json_str:
                        tool_calls.append(json_str)
                    self._tool_call_buf = ""
                    self.state = "NORMAL"

            elif self.state == "IN_THINK":
                self._think_buf += ch
                if self._think_buf.endswith("</think>"):
                    self._think_buf = ""
                    self.state = "NORMAL"

        return "".join(clean_parts), tool_calls

    def flush(self) -> tuple[str, list[str]]:
        """
        Wypycha resztki z buforów na koniec strumienia.
        Zwraca (remaining_clean_text, remaining_tool_calls).
        """
        remaining = ""
        tool_calls: list[str] = []

        # Tag candidate, który nigdy nie trafił do żadnego stanu
        if self._tag_candidate:
            remaining += self._tag_candidate
            self._tag_candidate = ""

        # Niedokończony tool_call (model urwał w połowie) — ignorujemy JSON
        if self.state == "IN_TOOL_CALL" and self._tool_call_buf:
            log.warning("TagAwareBuffer.flush: Niedokończony <tool_call>, ignoruję: %s",
                        self._tool_call_buf[:80])
            self._tool_call_buf = ""

        # Niedokończony think — po prostu odrzucamy
        if self.state == "IN_THINK":
            self._think_buf = ""

        self.state = "NORMAL"
        return remaining, tool_calls


# ============================================================================
# SentenceBuffer — bufor zdań do Piper TTS
# ============================================================================
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


# ============================================================================
# DaemonBrain — Moduł inferencji LLM
# ============================================================================

# Typ callbacka do wykonywania narzędzi (injektowany przez ws.py)
ToolExecutorType = Callable[[str, dict], Awaitable[str]]


class DaemonBrain:
    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        log.info("DaemonBrain loaded (model=%s, url=%s)", server_settings.model_brain, server_settings.ollama_host)

    # ------------------------------------------------------------------
    # Prywatne metody pomocnicze
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Buduje tablicę messages[] dla Ollama /api/chat."""
        if system_prompt is None:
            system_prompt = server_settings.get_brain_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        msg: dict = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]
        messages.append(msg)
        return messages

    async def _probe_ollama(self) -> None:
        """Diagnostyczny GET /api/tags — loguje dostępne modele."""
        probe_url = f"{server_settings.ollama_host}/api/tags"
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

    async def _stream_ollama_raw(self, messages: list[dict]) -> AsyncIterator[str]:
        """
        Strumień surowych tokenów tekstu z Ollamy /api/chat (POST streaming).
        Zwraca surowe tokeny content — BEZ filtracji <think>/<tool_call>.
        """
        payload = {
            "model": server_settings.model_brain,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
            },
            "keep_alive": "15m",
        }

        url = f"{server_settings.ollama_host}/api/chat"
        ttft_timeout = getattr(server_settings, "brain_ttft_timeout", 60.0)
        timeout_cfg = httpx.Timeout(
            timeout=180.0,
            connect=5.0,
            read=ttft_timeout,
            write=10.0,
            pool=5.0,
        )

        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
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
                            yield content

                        if isinstance(chunk, dict) and chunk.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        log.error("JSON decode error w brain._stream_ollama_raw: %s (line: %s)", e, line)
                    except Exception as e:
                        log.error("Nieoczekiwany blad w brain._stream_ollama_raw: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Publiczne API — strumień z filtracją (kompatybilne wstecz)
    # ------------------------------------------------------------------

    async def stream_chat(self, prompt: str, image_b64: Optional[str] = None) -> AsyncIterator[str]:
        """
        Strumień czystego tekstu (bez <think>, bez <tool_call>).
        Kompatybilna wstecz z Sesją 23.
        """
        if not self._loaded:
            raise RuntimeError("LLM is not loaded")

        await self._probe_ollama()
        messages = self._build_messages(prompt, image_b64)

        log.debug("Brain: Rozpoczynam streaming odpowiedzi (model: %s)", server_settings.model_brain)
        full_text = ""

        try:
            tag_buf = TagAwareBuffer()
            async for raw_token in self._stream_ollama_raw(messages):
                clean, _tool_calls = tag_buf.feed(raw_token)
                if clean:
                    full_text += clean
                    yield clean

            remaining, _ = tag_buf.flush()
            if remaining:
                full_text += remaining
                yield remaining

            log.info("[DAEMON BRAIN ODPOWIEDŹ]: %s", full_text)

        except asyncio.CancelledError:
            log.info("Brain: Generowanie przerwane (barge-in / Cancelled).")
            raise
        except httpx.TimeoutException:
            log.warning("Brain: Timeout TTFT (%ss) przy zapytaniu do Ollamy.", getattr(server_settings, "brain_ttft_timeout", 60.0))
            yield "Przepraszam, model nie odpowiedział w wyznaczonym czasie."
        except httpx.ConnectError as e:
            log.error("Brain: Błąd połączenia z Ollamą: %s", e)
            yield "Przepraszam, brak połączenia z serwerem Ollama."
        except Exception as exc:
            log.error("LLM streaming failed: %s", exc, exc_info=True)
            yield "Przepraszam, wystąpił błąd generatora LLM."

    # ------------------------------------------------------------------
    # Publiczne API — ReAct loop z obsługą tool calling
    # ------------------------------------------------------------------

    async def stream_chat_with_tools(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        tool_executor: Optional[ToolExecutorType] = None,
        max_rounds: int = 2,
    ) -> AsyncIterator[str]:
        """
        Strumień czystego tekstu z pętlą ReAct (max_rounds iteracji).

        1. Strumień z Ollamy -> TagAwareBuffer filtruje <tool_call>/<think>.
        2. Czysty tekst jest yieldowany do TTS.
        3. Jeśli wykryto <tool_call>, wykonujemy narzędzie przez tool_executor.
        4. Wynik narzędzia trafia jako <tool_response> do kontekstu LLM (runda 2).
        5. Model w rundzie 2 generuje zwięzłe potwierdzenie głosowe.
        """
        if not self._loaded:
            raise RuntimeError("LLM is not loaded")

        await self._probe_ollama()
        messages = self._build_messages(prompt, image_b64)

        log.debug("Brain ReAct: Start (model=%s, max_rounds=%d)", server_settings.model_brain, max_rounds)

        try:
            for round_idx in range(max_rounds):
                tag_buf = TagAwareBuffer()
                tool_calls_found: list[str] = []
                round_text = ""

                async for raw_token in self._stream_ollama_raw(messages):
                    clean, tool_calls = tag_buf.feed(raw_token)
                    if clean:
                        round_text += clean
                        yield clean
                    tool_calls_found.extend(tool_calls)

                # Flush resztek z bufora
                remaining, final_tcs = tag_buf.flush()
                if remaining:
                    round_text += remaining
                    yield remaining
                tool_calls_found.extend(final_tcs)

                log.info("Brain ReAct runda %d: tekst=%d znaków, tool_calls=%d",
                         round_idx + 1, len(round_text), len(tool_calls_found))

                # Jeśli brak tool_calls lub brak executora — koniec pętli
                if not tool_calls_found or tool_executor is None:
                    break

                # Przetwarzanie każdego tool_call
                for tc_json_str in tool_calls_found:
                    try:
                        tc = json.loads(tc_json_str)
                    except json.JSONDecodeError:
                        log.warning("Brain ReAct: Nieprawidłowy JSON tool_call: %s", tc_json_str[:80])
                        continue

                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})

                    # Walidacja Safe API
                    if tool_name not in ALLOWED_TOOLS:
                        log.warning("Brain ReAct: Odrzucono nieznane narzędzie '%s'", tool_name)
                        result_json = json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})
                    elif tool_name == "desktop_action" and tool_args.get("action") not in ALLOWED_ACTIONS:
                        log.warning("Brain ReAct: Odrzucono nieznaną akcję '%s'", tool_args.get("action"))
                        result_json = json.dumps({"status": "error", "message": f"Unknown action: {tool_args.get('action')}"})
                    else:
                        # Wykonanie narzędzia przez callback z ws.py
                        try:
                            result_json = await asyncio.wait_for(
                                tool_executor(tool_name, tool_args),
                                timeout=10.0,
                            )
                        except asyncio.TimeoutError:
                            log.warning("Brain ReAct: Timeout executora narzędzia '%s'", tool_name)
                            result_json = json.dumps({"status": "error", "message": "Tool execution timeout"})
                        except Exception as exec_err:
                            log.error("Brain ReAct: Błąd executora '%s': %s", tool_name, exec_err)
                            result_json = json.dumps({"status": "error", "message": str(exec_err)})

                    # Jeśli narzędzie zwróciło obraz (np. request_frame), wyodrębnij go do pola images
                    extracted_b64 = None
                    try:
                        res_obj = json.loads(result_json)
                        if isinstance(res_obj, dict) and "image_b64" in res_obj:
                            extracted_b64 = res_obj.pop("image_b64")
                            result_json_for_content = json.dumps(res_obj)
                        else:
                            result_json_for_content = result_json
                    except Exception:
                        result_json_for_content = result_json

                    # Injektuj do kontekstu: asystent wysłał tool_call, odpowiedź narzędzia
                    messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{tc_json_str}\n</tool_call>",
                    })
                    tool_msg: dict = {
                        "role": "tool",
                        "content": f"<tool_response>\n{result_json_for_content}\n</tool_response>",
                    }
                    if extracted_b64:
                        tool_msg["images"] = [extracted_b64]
                    messages.append(tool_msg)
                    log.info("Brain ReAct: tool=%s, args=%s, result=%s (has_image=%s)",
                             tool_name, tool_args, result_json_for_content[:120], bool(extracted_b64))

                # Kontynuuj do następnej rundy — model wygeneruje potwierdzenie

            log.info("Brain ReAct: Zakończono po %d rundach.", round_idx + 1)

        except asyncio.CancelledError:
            log.info("Brain ReAct: Generowanie przerwane (barge-in / Cancelled).")
            raise
        except httpx.TimeoutException:
            log.warning("Brain ReAct: Timeout TTFT przy zapytaniu do Ollamy.")
            yield "Przepraszam, model nie odpowiedział w wyznaczonym czasie."
        except httpx.ConnectError as e:
            log.error("Brain ReAct: Błąd połączenia z Ollamą: %s", e)
            yield "Przepraszam, brak połączenia z serwerem Ollama."
        except Exception as exc:
            log.error("Brain ReAct streaming failed: %s", exc, exc_info=True)
            yield "Przepraszam, wystąpił błąd generatora LLM."
