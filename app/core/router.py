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

    def __init__(self):
        # Odpytujemy natywne API Ollamy
        self.ollama_url = f"{server_settings.ollama_host}/api/chat"
        self.timeout = 1.2  # Zmniejszony timeout na 1.2s

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
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                if hasattr(data, "message") and hasattr(data.message, "content"):
                    content = data.message.content or ""
                elif isinstance(data, dict):
                    content = data.get("message", {}).get("content", "")
                else:
                    content = str(data)
                
                # 3. Walidacja Type-Safe przez Pydantic
                intent = self._adapter.validate_json(content)
                log.debug("Router: LLM zwrocil intent '%s' (session_id=%s)", intent.intent_type, session_id)
                return intent

        except httpx.TimeoutException:
            log.warning("Router: Timeout %ss. Fallback do LLM_QUERY.", self.timeout)
        except httpx.HTTPError as e:
            log.error("Router: Blad HTTP: %s. Fallback do LLM_QUERY.", e)
        except Exception as e:
            log.error("Router: Blad walidacji/JSON: %s. Fallback do LLM_QUERY.", e)

        # 4. Deterministyczny Fallback
        return LLMQueryIntent(query=text)
