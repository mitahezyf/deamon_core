import struct

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import get_logger

log = get_logger("api.ws")
router = APIRouter()

# format nagłówka paczki WebSocket:
# [4 bajty big-endian uint32: długość danych PCM] + [dane PCM float32]
# pusty pakiet (długość=0) sygnalizuje koniec strumienia
_HEADER_FMT = ">I"


@router.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket):
    # WebSocket endpoint do streamowania audio w czasie rzeczywistym
    # klient wysyła tekst, serwer odsyła chunki PCM na bieżąco bez czekania na całość
    await websocket.accept()
    log.info("Nowe połączenie WebSocket: %s", websocket.client)
    try:
        while True:
            text = await websocket.receive_text()
            if not text.strip():
                log.debug("Pusty tekst - pomijam")
                continue

            log.info("Synteza strumieniowa: %r...", text[:60])
            vox = websocket.app.state.vox
            chunk_count = 0

            for chunk in vox.stream_chunks(text):
                pcm_bytes = chunk.astype(np.float32).tobytes()
                header = struct.pack(_HEADER_FMT, len(pcm_bytes))
                await websocket.send_bytes(header + pcm_bytes)
                chunk_count += 1

            # sygnal konca strumienia - pusty pakiet z dlugoscia 0
            await websocket.send_bytes(struct.pack(_HEADER_FMT, 0))
            log.debug("Strumień zakończony, wysłano %d chunków", chunk_count)

    except WebSocketDisconnect:
        log.info("Klient WebSocket rozłączył się: %s", websocket.client)
    except Exception as e:
        log.error("Błąd WebSocket: %s", e, exc_info=True)
        await websocket.close(code=1011)
