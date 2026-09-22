from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.http import router as http_router
from app.api.routes.ui import router as ui_router
from app.api.routes.ws import router as ws_router
from app.api.routes.graph import router as graph_router
from app.core.brain import DaemonBrain
from app.core.database import MemgraphClient
from config import server_settings as settings
from app.core.logger import get_logger
from app.core.vox import DaemonVox

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # laduje modele przy starcie, potem wszystko jest gotowe
    log.info("Daemon startuje (debug_mode=%s)...", settings.debug_mode)

    vox = DaemonVox()
    vox.load()
    app.state.vox = vox

    brain = DaemonBrain()
    try:
        brain.load()
    except Exception as exc:
        log.warning("Brain load skipped: %s", exc)
    app.state.brain = brain
    
    from app.core.router import DaemonRouter
    app.state.router = DaemonRouter()

    db = MemgraphClient()
    await db.connect()
    app.state.db = db

    log.info(
        "Daemon gotowy | LLM: %s | port: %s",
        settings.model_brain,
        settings.port,
    )
    yield
    # sprzatanie przy wylaczeniu
    log.info("Daemon zatrzymywany...")
    await app.state.db.close()
    del app.state.db
    del app.state.brain
    del app.state.vox
    log.info("Daemon zatrzymany.")


app = FastAPI(
    title="Daemon API",
    description="Lokalny asystent AI — backend WebSocket i REST (serwer LXC)",
    version="0.2.0",
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
app.include_router(graph_router)

_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="debug" if settings.debug_mode else "info",
    )
