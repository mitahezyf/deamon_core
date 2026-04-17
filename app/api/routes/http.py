import base64

import torch
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    HealthResponse,
    PublicConfigResponse,
    StatusResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("api.http")
router = APIRouter()


def _build_runtime_status(request: Request) -> dict:
    vox = request.app.state.vox
    ears = request.app.state.ears
    stt = request.app.state.stt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "vox_loaded": vox._tts is not None,
        "ears_loaded": ears.is_loaded,
        "device": device,
        "llm_model": settings.llm_model,
        "language": settings.language,
        "stt_enabled": settings.stt_enabled,
        "stt_loaded": stt.is_loaded,
        "stt_sample_rate": settings.stt_sample_rate,
        "wake_word_enabled": settings.wake_word_enabled,
        "wake_word_backend": settings.wake_word_backend,
        "wake_word_label": settings.wake_word_label,
        "wake_word_threshold": settings.wake_word_threshold,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
    }


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    # zwraca stan serwera, przydatne do debugowania i monitorowania przez GUI
    runtime = _build_runtime_status(request)
    log.debug(
        "Health check - vox_loaded=%s, device=%s",
        runtime["vox_loaded"],
        runtime["device"],
    )
    return HealthResponse(
        status=runtime["status"],
        vox_loaded=runtime["vox_loaded"],
        ears_loaded=runtime["ears_loaded"],
        device=runtime["device"],
        llm_model=runtime["llm_model"],
        api_port=runtime["api_port"],
    )


@router.get("/status", response_model=StatusResponse)
async def status(request: Request):
    runtime = _build_runtime_status(request)
    return StatusResponse(**runtime)


@router.get("/config/public", response_model=PublicConfigResponse)
async def public_config():
    return PublicConfigResponse(
        llm_model=settings.llm_model,
        ollama_url=settings.ollama_url,
        language=settings.language,
        whisper_model=settings.whisper_model,
        tts_model=settings.tts_model,
        stt_enabled=settings.stt_enabled,
        stt_sample_rate=settings.stt_sample_rate,
        wake_word_enabled=settings.wake_word_enabled,
        wake_word_backend=settings.wake_word_backend,
        wake_word_label=settings.wake_word_label,
        wake_word_threshold=settings.wake_word_threshold,
        api_host=settings.api_host,
        api_port=settings.api_port,
        wake_word_model=str(settings.wake_word_model),
        openwakeword_model_path=str(settings.openwakeword_model_path),
    )


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest, request: Request):
    # REST endpoint do jednorazowej syntezy, zapisuje WAV i zwraca metadane
    log.info("POST /synthesize | tekst: %r...", req.text[:60])
    vox = request.app.state.vox
    output_path = settings.output_dir / req.output  # type: ignore[operator]
    result = vox.synthesize_to_file(req.text, output_path)
    log.info("POST /synthesize zakończony | total=%.3fs", result["total_time"])
    return SynthesizeResponse(**result)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest, request: Request):
    stt = request.app.state.stt
    if not stt.is_loaded:
        return TranscribeResponse(text="", sample_rate=req.sample_rate)

    try:
        audio_bytes = base64.b64decode(req.audio_b64, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Invalid audio_b64 payload"
        ) from exc
    text = stt.transcribe_pcm16(audio_bytes, sample_rate=req.sample_rate)
    return TranscribeResponse(text=text, sample_rate=req.sample_rate)
