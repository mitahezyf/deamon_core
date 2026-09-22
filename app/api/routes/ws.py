import struct
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.schemas import (
    UserPromptEvent, FrameResponseEvent, AbortGenerationEvent,
    StateChangeEvent, RequestFrameEvent, ExecActionEvent,
    SystemCommandIntent, VisionQueryIntent, LLMQueryIntent
)
from app.core.logger import get_logger

log = get_logger("api.ws")
router = APIRouter()

# format naglowka paczki WebSocket:
# [4 bajty big-endian uint32: dlugosc danych PCM] + [dane PCM int16]
# pusty pakiet (dlugosc=0) sygnalizuje koniec strumienia
_HEADER_FMT = ">I"


@router.websocket("/api/v1/ws/agent")
async def ws_agent(websocket: WebSocket):
    await websocket.accept()
    log.info("Agent WS polaczony: %s", websocket.client)
    
    app_state = websocket.app.state
    vox = app_state.vox
    router_svc = getattr(app_state, "router", None)
    brain = app_state.brain

    if not router_svc:
        # Prowizorycznie inicjujemy router jesli nie ma w state
        from app.core.router import DaemonRouter
        router_svc = DaemonRouter()

    async def send_state(state: str, session_id: str):
        event = StateChangeEvent(new_state=state, session_id=session_id)
        await websocket.send_text(event.model_dump_json())

    # Na starcie STANDBY
    await send_state("STANDBY", "init")

    current_task: asyncio.Task = None

    try:
        while True:
            # Oczekujemy na ramki tekstowe z gniazda
            raw_msg = await websocket.receive_text()
            
            try:
                data = json.loads(raw_msg)
                event_type = data.get("event_type")
            except json.JSONDecodeError:
                log.warning("Otrzymano niepoprawny JSON: %s", raw_msg[:50])
                continue

            if event_type == "abort_generation":
                # Barge-in: przerywamy aktualne gadanie/myslenie
                if current_task and not current_task.done():
                    current_task.cancel()
                    log.info("Przerwano aktualne generowanie (barge-in).")
                abort_ev = AbortGenerationEvent.model_validate(data)
                await send_state("STANDBY", abort_ev.session_id)
                continue

            if event_type == "user_prompt":
                prompt_ev = UserPromptEvent.model_validate(data)
                session_id = prompt_ev.session_id
                
                # Zabezpieczenie przed podwójnym startem
                if current_task and not current_task.done():
                    current_task.cancel()

                async def process_prompt():
                    try:
                        await send_state("ROUTING", session_id)
                        intent = await router_svc.route(prompt_ev.text, session_id=session_id)
                        
                        if isinstance(intent, SystemCommandIntent):
                            exec_ev = ExecActionEvent(
                                action=intent.action, 
                                session_id=session_id
                            )
                            await websocket.send_text(exec_ev.model_dump_json())
                            await send_state("STANDBY", session_id)
                            return
                            
                        image_b64 = None
                        if isinstance(intent, VisionQueryIntent):
                            req_ev = RequestFrameEvent(session_id=session_id)
                            await websocket.send_text(req_ev.model_dump_json())
                            
                            # Czekamy na klatke (z krotkim timeoutem)
                            try:
                                frame_msg = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                                frame_data = json.loads(frame_msg)
                                if frame_data.get("event_type") == "frame_response":
                                    frame_ev = FrameResponseEvent.model_validate(frame_data)
                                    image_b64 = frame_ev.image_b64
                            except asyncio.TimeoutError:
                                log.warning("Timeout oczekiwania na frame_response")
                        
                        await send_state("INFERENCE_LLM", session_id)
                        query = intent.query
                        # LLM zwraca strumien tokenow
                        text_stream = brain.stream_chat(query, image_b64)
                        
                        # SentenceBuffer do sklejania tokenow w zdania
                        from app.core.brain import SentenceBuffer
                        async def sentence_generator():
                            sb = SentenceBuffer(min_length=15)
                            async for token in text_stream:
                                sentences = sb.add(token)
                                for sentence in sentences:
                                    yield sentence
                            remainder = sb.flush()
                            if remainder:
                                yield remainder

                        await send_state("STREAMING_TTS", session_id)
                        
                        chunk_count = 0
                        # Piper TTS zwraca wygenerowane binarne chunki audio (PCM 16-bit) z otrzymywanych zdan
                        async for pcm_bytes in vox.stream_sentences(sentence_generator()):
                            if chunk_count == 0:
                                await send_state("SPEAKING", session_id)
                            header = struct.pack(_HEADER_FMT, len(pcm_bytes))
                            await websocket.send_bytes(header + pcm_bytes)
                            chunk_count += 1
                            
                        # Koniec strumienia (pusty chunk)
                        await websocket.send_bytes(struct.pack(_HEADER_FMT, 0))
                        await send_state("STANDBY", session_id)

                    except asyncio.CancelledError:
                        log.info("Zadanie przerwane (Cancelled).")
                    except Exception as e:
                        log.error("Błąd w trakcie przetwarzania: %s", e, exc_info=True)
                        await send_state("STANDBY", session_id)

                current_task = asyncio.create_task(process_prompt())

    except WebSocketDisconnect:
        log.info("Klient WS rozłączony.")
    except Exception as e:
        log.error("Krytyczny błąd WS: %s", e, exc_info=True)

