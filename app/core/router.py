import httpx
from typing import Optional
from pydantic import TypeAdapter

from config import server_settings
from app.api.schemas import (
    IntentDecision, 
    VolumeControlIntent,
    AppControlIntent,
    SystemStatusIntent,
    VisionQueryIntent, 
    LLMQueryIntent
)
from app.core.logger import get_logger

log = get_logger("router")

class DaemonRouter:
    """
    Fast Intent Router. Decyduje, co zrobic z zapytaniem uzytkownika w <50 ms.
    Oparto o maly LLM (np. qwen2.5:0.5b).
    """

    def __init__(self, timeout: Optional[float] = None):
        # Odpytujemy natywne API Ollamy
        self.ollama_url = f"{server_settings.ollama_host}/api/chat"
        self.timeout = timeout if timeout is not None else getattr(server_settings, "router_timeout", 3.0)

        self._adapter = TypeAdapter(IntentDecision)

    def _heuristic_match(self, text: str) -> Optional[IntentDecision]:
        """Heurystyka poziomu 0 (0 ms). Proste wyłapywanie komend systemowych."""
        text_lower = text.lower().strip()
        
        commands = {
            "ciszej": ("VOLUME_CONTROL", "volume_down"),
            "głośniej": ("VOLUME_CONTROL", "volume_up"),
            "wycisz": ("VOLUME_CONTROL", "mute"),
            "odcisz": ("VOLUME_CONTROL", "unmute")
        }
        
        for kw, action_tuple in commands.items():
            if text_lower == kw:
                log.debug("Router: Heurystyka wyłapała komendę '%s' -> %s", kw, action_tuple)
                if action_tuple[0] == "VOLUME_CONTROL":
                    return VolumeControlIntent(action=action_tuple[1])
                    
        import re
        app_match = re.match(r'^(włącz|otwórz|odpal)\s+(.+)', text_lower)
        if app_match:
            app_name = app_match.group(2).strip()
            log.debug("Router: Heurystyka wyłapała APP_CONTROL -> %s", app_name)
            return AppControlIntent(app_name=app_name)
                
        return None

    async def route(self, text: str, session_id: str = "default") -> IntentDecision:
        """
        Zwraca type-safe IntentDecision.
        Najpierw heurystyka, potem strzał do Ollamy, a w razie bledu/timeoutu -> fallback.
        """
        # 1. Heurystyka poziomu 0
        heuristic_intent = self._heuristic_match(text)
        if heuristic_intent:
            return heuristic_intent

        # 2. Wywołanie API LLM (Poziom 1)
        system_prompt = server_settings.get_router_prompt()

        payload = {
            "model": server_settings.model_router,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "options": {"temperature": 0.0, "num_predict": 32},
            "format": "json",
            "stream": False,
            "keep_alive": "15m"
        }

        try:
            timeout_cfg = httpx.Timeout(self.timeout, connect=min(2.0, self.timeout))
            async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                if hasattr(data, "message") and hasattr(data.message, "content"):
                    content = data.message.content or ""
                elif isinstance(data, dict):
                    content = data.get("message", {}).get("content", "")
                else:
                    content = str(data)
                
                # Walidacja: pusta odpowiedź od modelu Ollamy
                if not content or not content.strip():
                    log.warning("Router: Otrzymano pustą odpowiedź od modelu Ollamy. Fallback do LLM_QUERY.")
                    return LLMQueryIntent(query=text)

                content_clean = content.strip()
                try:
                    # 3. Walidacja Type-Safe przez Pydantic
                    intent = self._adapter.validate_json(content_clean)
                    log.debug("Router: LLM zwrocil intent '%s' (session_id=%s)", intent.intent_type, session_id)
                    return intent
                except Exception as val_err:
                    log.warning("Router: Odpowiedź modelu nie spełnia schematu JSON (%s): %s. Fallback do LLM_QUERY.", val_err, content_clean[:80])
                    return LLMQueryIntent(query=text)

        except httpx.TimeoutException:
            log.warning("Router: Timeout %ss przy odpytywaniu Ollamy. Fallback do LLM_QUERY.", self.timeout)
        except httpx.ConnectError as e:
            log.warning("Router: Brak połączenia z Ollamą (%s). Fallback do LLM_QUERY.", e)
        except httpx.HTTPError as e:
            log.error("Router: Blad HTTP: %s. Fallback do LLM_QUERY.", e)
        except Exception as e:
            log.error("Router: Nieoczekiwany błąd routera: %s. Fallback do LLM_QUERY.", e)

        # 4. Deterministyczny Fallback
        return LLMQueryIntent(query=text)
