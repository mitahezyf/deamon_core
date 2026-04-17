import logging

from app.core.vox import DaemonVox

# skrypt jednorazowy do przebudowania cache embeddingów głosu
# użyj gdy dodasz nowe próbki do voice_samples/
# uruchom: python scripts/build_voice_cache.py


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] build_voice_cache - %(message)s",
)
log = logging.getLogger("build_voice_cache")

if __name__ == "__main__":
    vox = DaemonVox()
    vox.rebuild_cache()
    log.info("Gotowe. Mozesz teraz uruchomic serwer normalnie.")
