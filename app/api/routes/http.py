import torch
from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse, SynthesizeRequest, SynthesizeResponse
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("api.http")
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    # zwraca stan serwera, przydatne do debugowania i monitorowania przez GUI
    vox = request.app.state.vox
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.debug("Health check - vox_loaded=%s, device=%s", vox._tts is not None, device)
    return HealthResponse(
        status="ok",
        vox_loaded=vox._tts is not None,
        device=device,
        llm_model=settings.llm_model,
        api_port=settings.api_port,
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
