from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.http import router as http_router
from app.api.routes.ui import router as ui_router
from app.api.routes.ws import router as ws_router
from app.core.config import settings
from app.core.ears import DaemonEars
from app.core.logger import get_logger
from app.core.stt import DaemonStt
from app.core.vox import DaemonVox

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # laduje modele przy starcie, potem wszystko jest gotowe
    log.info("Daemon startuje (debug_mode=%s)...", settings.debug_mode)
    vox = DaemonVox()
    vox.load()
    vox.warmup()
    app.state.vox = vox

    ears = DaemonEars()
    try:
        ears.load()
    except Exception as exc:
        log.warning("Ears load skipped: %s", exc)
    app.state.ears = ears

    stt = DaemonStt()
    try:
        stt.load()
    except Exception as exc:
        log.warning("STT load skipped: %s", exc)
    app.state.stt = stt

    log.info(
        "Daemon gotowy | LLM: %s | port: %d",
        settings.llm_model,
        settings.api_port,
    )
    yield
    # sprzatanie przy wyaczeniu
    log.info("Daemon zatrzymywany...")
    del app.state.stt
    del app.state.ears
    del app.state.vox
    log.info("Daemon zatrzymany.")


app = FastAPI(
    title="Daemon API",
    description="Lokalny asystent AI - backend WebSocket i REST",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - zezwala na polaczenia z GUI w sieci LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_router)
app.include_router(ws_router)
app.include_router(ui_router)

_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="debug" if settings.debug_mode else "info",
    )
