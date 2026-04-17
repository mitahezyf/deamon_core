from pydantic import BaseModel

# --- żądania ---


class SynthesizeRequest(BaseModel):
    # żądanie syntezy mowy
    text: str
    output: str = "daemon_out.wav"


# --- odpowiedzi ---


class SynthesizeResponse(BaseModel):
    # wynik syntezy z metrykami latencji
    latency_first_chunk: float
    total_time: float
    audio_duration: float
    output: str


class HealthResponse(BaseModel):
    # stan serwera i załadowanych modeli
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


class PublicConfigResponse(BaseModel):
    llm_model: str
    ollama_url: str
    language: str
    whisper_model: str
    tts_model: str
    api_host: str
    api_port: int
    wake_word_model: str
