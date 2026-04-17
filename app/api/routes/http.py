import torch
from fastapi import APIRouter, Request

from app.api.schemas import (
    HealthResponse,
    PublicConfigResponse,
    StatusResponse,
    SynthesizeRequest,
    SynthesizeResponse,
)
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("api.http")
router = APIRouter()


def _build_runtime_status(request: Request) -> dict:
    vox = request.app.state.vox
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "vox_loaded": vox._tts is not None,
        "device": device,
        "llm_model": settings.llm_model,
        "language": settings.language,
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
        api_host=settings.api_host,
        api_port=settings.api_port,
        wake_word_model=str(settings.wake_word_model),
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
