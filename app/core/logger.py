import logging
import sys

from app.core.config import settings

# format logu: czas | poziom (wyrownany) | nazwa modulu | tresc
_LOG_FMT = "%(asctime)s [%(levelname)-8s] %(name)-24s - %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# flaga zapobiegajaca wielokrotnemu dodawaniu handlerow
_configured = False


def _konfiguruj_root_logger() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("daemon")
    root.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FMT, _DATE_FMT))
    root.addHandler(handler)

    # wycisza nadmiarowe logi z zewnetrznych bibliotek
    for noisy in ("uvicorn.access", "TTS", "numba", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    # zwraca logger w przestrzeni nazw daemon.<name>
    # debug_mode=True w .env wlacza logi DEBUG, domyslnie tylko INFO i wyzej
    _konfiguruj_root_logger()
    return logging.getLogger(f"daemon.{name}")
