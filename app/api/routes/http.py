import psutil
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    AssistantReplyRequest,
    AssistantReplyResponse,
    HealthResponse,
    PublicConfigResponse,
    StatusResponse,
    SynthesizeRequest,
    SynthesizeResponse,
)
from config import server_settings as settings
from app.core.logger import get_logger

log = get_logger("api.http")
router = APIRouter()


def _build_runtime_status(request: Request) -> dict:
    vox = request.app.state.vox
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "vox_loaded": vox.is_loaded,
        "device": "cpu",
        "llm_model": settings.model_brain,
        "language": settings.language,
        "api_host": settings.host,
        "api_port": settings.port,
        "mem_used_mb": round(mem.used / 1024 / 1024),
        "mem_total_mb": round(mem.total / 1024 / 1024),
    }


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    # zwraca stan serwera, przydatne do debugowania i monitorowania przez GUI
    runtime = _build_runtime_status(request)
    
    memgraph_connected = False
    if hasattr(request.app.state, "db") and request.app.state.db:
        memgraph_connected = await request.app.state.db.verify_connectivity()
        
    log.debug(
        "Health check - vox_loaded=%s, device=%s, memgraph_connected=%s",
        runtime["vox_loaded"],
        runtime["device"],
        memgraph_connected
    )
    return HealthResponse(
        status=runtime["status"],
        vox_loaded=runtime["vox_loaded"],
        device=runtime["device"],
        llm_model=runtime["llm_model"],
        api_port=runtime["api_port"],
        memgraph_connected=memgraph_connected
    )


@router.get("/status", response_model=StatusResponse)
async def status(request: Request):
    runtime = _build_runtime_status(request)
    return StatusResponse(**runtime)


@router.get("/config/public", response_model=PublicConfigResponse)
async def public_config():
    return PublicConfigResponse(
        llm_model=settings.model_brain,
        ollama_url=settings.ollama_host,
        language=settings.language,
        api_host=settings.host,
        api_port=settings.port,
    )


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest, request: Request):
    # REST endpoint do jednorazowej syntezy, zapisuje WAV i zwraca metadane
    log.info("POST /synthesize | tekst: %r...", req.text[:60])
    vox = request.app.state.vox
    output_path = settings.output_dir / req.output  # type: ignore[operator]
    result = vox.synthesize_to_file(req.text, output_path)
    log.info("POST /synthesize zakonczony | total=%.3fs", result["total_time"])
    return SynthesizeResponse(**result)


@router.post("/assistant/reply", response_model=AssistantReplyResponse)
async def assistant_reply(req: AssistantReplyRequest, request: Request):
    text = req.text.strip()
    if not text:
        return AssistantReplyResponse(reply="")

    brain = request.app.state.brain
    if not brain.is_loaded:
        raise HTTPException(status_code=503, detail="LLM is not ready")

    try:
        reply = brain.reply(text)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AssistantReplyResponse(reply=reply)

