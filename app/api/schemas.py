from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated

# =============================================================================
# DAEMON Core — API Schemas (serwer LXC)
# =============================================================================

# --- ZADANIA I ODPOWIEDZI REST ---

class SynthesizeRequest(BaseModel):
    text: str
    output: str = "daemon_out.wav"

class AssistantReplyRequest(BaseModel):
    text: str

class SynthesizeResponse(BaseModel):
    latency_first_chunk: float
    total_time: float
    audio_duration: float
    output: str

class AssistantReplyResponse(BaseModel):
    reply: str

class HealthResponse(BaseModel):
    status: str
    vox_loaded: bool
    device: str
    llm_model: str
    api_port: int

class StatusResponse(BaseModel):
    status: str
    vox_loaded: bool
    device: str
    llm_model: str
    language: str
    api_host: str
    api_port: int
    mem_used_mb: int
    mem_total_mb: int

class PublicConfigResponse(BaseModel):
    llm_model: str
    ollama_url: str
    language: str
    api_host: str
    api_port: int

# --- PROTOKÓŁ WEBSOCKET (Klient -> Serwer) ---

class UserPromptEvent(BaseModel):
    event_type: Literal["user_prompt"] = "user_prompt"
    text: str
    session_id: str

class FrameResponseEvent(BaseModel):
    event_type: Literal["frame_response"] = "frame_response"
    image_b64: str
    session_id: str

class AbortGenerationEvent(BaseModel):
    event_type: Literal["abort_generation"] = "abort_generation"
    session_id: str

# --- PROTOKÓŁ WEBSOCKET (Serwer -> Klient) ---

class RequestFrameEvent(BaseModel):
    event_type: Literal["request_frame"] = "request_frame"
    session_id: str

class ExecActionEvent(BaseModel):
    event_type: Literal["exec_action"] = "exec_action"
    action: str
    payload: Optional[Dict[str, Any]] = None
    session_id: str

class StateChangeEvent(BaseModel):
    event_type: Literal["state_change"] = "state_change"
    new_state: str
    session_id: str

class AssistantTextEvent(BaseModel):
    event_type: Literal["assistant_text"] = "assistant_text"
    text: str
    session_id: str

# --- INTENT ROUTER (Decyzje) ---

class VolumeControlIntent(BaseModel):
    intent_type: Literal["VOLUME_CONTROL"] = "VOLUME_CONTROL"
    action: Literal["volume_up", "volume_down", "mute", "unmute"] = Field(
        description="Akcja regulacji głośności."
    )

class AppControlIntent(BaseModel):
    intent_type: Literal["APP_CONTROL"] = "APP_CONTROL"
    app_name: str = Field(description="Nazwa aplikacji do uruchomienia/obsłużenia.")

class SystemStatusIntent(BaseModel):
    intent_type: Literal["SYSTEM_STATUS"] = "SYSTEM_STATUS"
    query: str = Field(description="Oryginalne zapytanie o status (bateria, godzina itp).")

class VisionQueryIntent(BaseModel):
    intent_type: Literal["VISION_QUERY"] = "VISION_QUERY"
    query: str = Field(description="Oryginalne zapytanie użytkownika wymagające spojrzenia na ekran.")

class LLMQueryIntent(BaseModel):
    intent_type: Literal["LLM_QUERY"] = "LLM_QUERY"
    query: str = Field(description="Oryginalne zapytanie użytkownika przeznaczone dla LLM (rozmowa, wiedza).")

# Dyskryminowana unia IntentDecision
IntentDecision = Annotated[
    Union[VolumeControlIntent, AppControlIntent, SystemStatusIntent, VisionQueryIntent, LLMQueryIntent],
    Field(discriminator="intent_type")
]
