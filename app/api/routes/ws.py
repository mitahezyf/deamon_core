import struct

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import get_logger

log = get_logger("api.ws")
router = APIRouter()

# format naglowka paczki WebSocket:
# [4 bajty big-endian uint32: dlugosc danych PCM] + [dane PCM float32]
# pusty pakiet (dlugosc=0) sygnalizuje koniec strumienia
_HEADER_FMT = ">I"

# TODO (Etap 6): Zastapic ponizszy stub centralnym hubem /api/v1/ws/agent
# zgodnym z pelnym protokolem z ai/03_PROTOCOLS_AND_STATE_MACHINE.md:
#   - user_prompt, frame_response, abort_generation (C->S)
#   - request_frame, exec_action, state_change, binary PCM (S->C)


@router.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket):
    # Tymczasowy WebSocket endpoint do streamowania audio w czasie rzeczywistym.
    # Klient wysyla tekst, serwer odsyla chunki PCM na biezaco.
    # STUB: Do zastapienia przez /api/v1/ws/agent w Etapie 6 roadmapy.
    await websocket.accept()
    log.info("Nowe polaczenie WebSocket (synthesize): %s", websocket.client)
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
            log.debug("Strumien zakonczony, wyslano %d chunkow", chunk_count)

    except WebSocketDisconnect:
        log.info("Klient WebSocket rozlaczyl sie: %s", websocket.client)
    except Exception as e:
        log.error("Blad WebSocket: %s", e, exc_info=True)
        await websocket.close(code=1011)

