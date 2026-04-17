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
