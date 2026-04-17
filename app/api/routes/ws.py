import struct

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.audio_pipeline_v2 import EarsStreamSession
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("api.ws")
router = APIRouter()

# format nagowka paczki WebSocket:
# [4 bajty big-endian uint32: dugosc danych PCM] + [dane PCM float32]
# pusty pakiet (dugosc=0) sygnalizuje koniec strumienia
_HEADER_FMT = ">I"


@router.websocket("/ws/ears/listen")
async def ws_ears_listen(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"event": "ready", "message": "ears websocket ready"})
    log.info("Nowe polaczenie ears WS: %s", websocket.client)
    session = EarsStreamSession(
        ears=websocket.app.state.ears,
        stt=websocket.app.state.stt,
        stt_enabled=settings.stt_enabled,
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect

            if message.get("bytes") is not None:
                payload = message["bytes"]
                for event in session.on_audio_bytes(payload):
                    await websocket.send_json(event)
                continue

            text_payload = message.get("text")
            if text_payload is None:
                continue

            for event in session.on_command(text_payload):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        log.info("Klient ears WS rozlaczyl sie: %s", websocket.client)
    except Exception as exc:
        log.error("Blad ears WS: %s", exc, exc_info=True)
        await websocket.close(code=1011)


@router.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket):
    # WebSocket endpoint do streamowania audio w czasie rzeczywistym
    # klient wysya tekst, serwer odsya chunki PCM na biezaco bez czekania na caosc
    await websocket.accept()
    log.info("Nowe poaczenie WebSocket: %s", websocket.client)
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
            log.debug("Strumien zakonczony, wysano %d chunkow", chunk_count)

    except WebSocketDisconnect:
        log.info("Klient WebSocket rozaczy sie: %s", websocket.client)
    except Exception as e:
        log.error("Bad WebSocket: %s", e, exc_info=True)
        await websocket.close(code=1011)
