import httpx
from typing import Optional
from pydantic import TypeAdapter

from app.core.config import settings
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
        self.ollama_url = f"{settings.ollama_url}/api/chat"
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
        system_prompt = (
            "You are a fast intent router. Classify the user's message into one of these categories:\n"
            "1. VOLUME_CONTROL - if user asks to change volume (volume_up, volume_down, mute, unmute).\n"
            "2. APP_CONTROL - if user asks to open an app (e.g. browser, calculator).\n"
            "3. SYSTEM_STATUS - if user asks about battery, time, or system status.\n"
            "4. VISION_QUERY - if user asks about what is currently on the screen.\n"
            "5. LLM_QUERY - for all other general questions, requests, or conversations.\n"
            "Respond ONLY with a valid JSON matching this schema:\n"
            "{\n"
            '  "intent_type": "VOLUME_CONTROL|APP_CONTROL|SYSTEM_STATUS|VISION_QUERY|LLM_QUERY",\n'
            '  "action": "volume_up|volume_down|mute|unmute",\n'
            '  "app_name": "name of the app (only for APP_CONTROL)",\n'
            '  "query": "original user text"\n'
            "}\n"
            "Do not include comments or markdown formatting, just the raw JSON object."
        )

        payload = {
            "model": settings.router_model,
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
                content = data.get("message", {}).get("content", "")
                
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
