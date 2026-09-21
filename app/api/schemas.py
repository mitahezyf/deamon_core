from pydantic import BaseModel

# =============================================================================
# DAEMON Core — API Schemas (serwer LXC)
# Schematy STT, Ears, WakeWord naleza do client_node_win/
# =============================================================================

# --- zadania REST ---


class SynthesizeRequest(BaseModel):
    # zadanie syntezy mowy przez Piper TTS ONNX
    text: str
    output: str = "daemon_out.wav"


class AssistantReplyRequest(BaseModel):
    text: str


# --- odpowiedzi REST ---


class SynthesizeResponse(BaseModel):
    # wynik syntezy z metrykami latencji
    latency_first_chunk: float
    total_time: float
    audio_duration: float
    output: str


class AssistantReplyResponse(BaseModel):
    reply: str


class HealthResponse(BaseModel):
    # minimalny health check
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
