from pydantic import BaseModel

# --- zadania ---


class SynthesizeRequest(BaseModel):
    # zadanie syntezy mowy
    text: str
    output: str = "daemon_out.wav"


class TranscribeRequest(BaseModel):
    # surowe PCM16 mono zakodowane base64
    audio_b64: str
    sample_rate: int = 16000


class TranscribeResponse(BaseModel):
    text: str
    sample_rate: int


# --- odpowiedzi ---


class SynthesizeResponse(BaseModel):
    # wynik syntezy z metrykami latencji
    latency_first_chunk: float
    total_time: float
    audio_duration: float
    output: str


class HealthResponse(BaseModel):
    # stan serwera i zaadowanych modeli
    status: str
    vox_loaded: bool
    ears_loaded: bool
    device: str
    llm_model: str
    api_port: int


class StatusResponse(BaseModel):
    status: str
    vox_loaded: bool
    ears_loaded: bool
    device: str
    llm_model: str
    language: str
    stt_enabled: bool
    stt_loaded: bool
    stt_sample_rate: int
    wake_word_enabled: bool
    wake_word_backend: str
    wake_word_label: str
    wake_word_threshold: float
    api_host: str
    api_port: int


class PublicConfigResponse(BaseModel):
    llm_model: str
    ollama_url: str
    language: str
    whisper_model: str
    tts_model: str
    stt_enabled: bool
    stt_sample_rate: int
    wake_word_enabled: bool
    wake_word_backend: str
    wake_word_label: str
    wake_word_threshold: float
    api_host: str
    api_port: int
    wake_word_model: str
    openwakeword_model_path: str
