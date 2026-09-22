import struct
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.api.schemas import (
    UserPromptEvent, FrameResponseEvent, AbortGenerationEvent,
    StateChangeEvent, RequestFrameEvent, ExecActionEvent,
    VolumeControlIntent, AppControlIntent, SystemStatusIntent,
    VisionQueryIntent, LLMQueryIntent, AssistantTextEvent
)
from app.core.logger import get_logger

log = get_logger("api.ws")
router = APIRouter()

# format naglowka paczki WebSocket:
# [4 bajty big-endian uint32: dlugosc danych PCM] + [dane PCM int16]
# pusty pakiet (dlugosc=0) sygnalizuje koniec strumienia
_HEADER_FMT = ">I"


async def safe_send_text(ws: WebSocket, text: str) -> bool:
    """Bezpieczne wysyłanie ramek tekstowych z weryfikacją stanu połączenia."""
    if ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_text(text)
            return True
        except (RuntimeError, WebSocketDisconnect):
            log.debug("Niepowodzenie wysłania tekstu - gniazdo zostało zamknięte.")
            return False
    return False


async def safe_send_bytes(ws: WebSocket, data: bytes) -> bool:
    """Bezpieczne wysyłanie ramek binarnych z weryfikacją stanu połączenia."""
    if ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_bytes(data)
            return True
        except (RuntimeError, WebSocketDisconnect):
            log.debug("Niepowodzenie wysłania bajtów - gniazdo zostało zamknięte.")
            return False
    return False


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
        if websocket.client_state == WebSocketState.CONNECTED:
            event = StateChangeEvent(new_state=state, session_id=session_id)
            await safe_send_text(websocket, event.model_dump_json())

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

            if event_type == "tts_request":
                text_to_say = data.get("text", "")
                session_id = data.get("session_id", "win_client")
                if text_to_say:
                    if current_task and not current_task.done():
                        current_task.cancel()

                    async def process_tts():
                        try:
                            await send_state("STREAMING_TTS", session_id)
                            await send_state("SPEAKING", session_id)
                            async def single_sentence_gen():
                                yield text_to_say
                            async for pcm_bytes in vox.stream_sentences(single_sentence_gen()):
                                if websocket.client_state != WebSocketState.CONNECTED:
                                    break
                                header = struct.pack(_HEADER_FMT, len(pcm_bytes))
                                if not await safe_send_bytes(websocket, header + pcm_bytes):
                                    break
                            await safe_send_bytes(websocket, struct.pack(_HEADER_FMT, 0))
                            await send_state("STANDBY", session_id)
                        except (asyncio.CancelledError, WebSocketDisconnect):
                            log.info("Zadanie TTS_REQUEST przerwane (barge-in / disconnect).")
                        except Exception as e:
                            log.error("Błąd podczas TTS_REQUEST: %s", e)
                            await send_state("STANDBY", session_id)
                    
                    current_task = asyncio.create_task(process_tts())
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
                        
                        async def quick_reply_and_execute(reply_text: str, action: str = None, payload: dict = None):
                            if action:
                                exec_ev = ExecActionEvent(action=action, payload=payload, session_id=session_id)
                                await safe_send_text(websocket, exec_ev.model_dump_json())
                            
                            await send_state("STREAMING_TTS", session_id)
                            await safe_send_text(websocket, AssistantTextEvent(text=reply_text, session_id=session_id).model_dump_json())
                            await send_state("SPEAKING", session_id)
                            
                            async def single_sentence_gen():
                                yield reply_text
                                
                            async for pcm_bytes in vox.stream_sentences(single_sentence_gen()):
                                if websocket.client_state != WebSocketState.CONNECTED:
                                    break
                                header = struct.pack(_HEADER_FMT, len(pcm_bytes))
                                if not await safe_send_bytes(websocket, header + pcm_bytes):
                                    break
                            await safe_send_bytes(websocket, struct.pack(_HEADER_FMT, 0))
                            await send_state("STANDBY", session_id)
                            
                        if isinstance(intent, VolumeControlIntent):
                            await quick_reply_and_execute("Jasne, modyfikuję głośność.", action=intent.action)
                            return
                        elif isinstance(intent, AppControlIntent):
                            await quick_reply_and_execute(f"Uruchamiam aplikację {intent.app_name}.", action="open_app", payload={"app_name": intent.app_name})
                            return
                        elif isinstance(intent, SystemStatusIntent):
                            await quick_reply_and_execute("Sprawdzam status.", action="check_status", payload={"query": intent.query})
                            return
                            
                        image_b64 = None
                        if isinstance(intent, VisionQueryIntent):
                            req_ev = RequestFrameEvent(session_id=session_id)
                            await safe_send_text(websocket, req_ev.model_dump_json())
                            
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
                            try:
                                async for token in text_stream:
                                    if websocket.client_state != WebSocketState.CONNECTED:
                                        break
                                    sentences = sb.add(token)
                                    for sentence in sentences:
                                        if not await safe_send_text(websocket, AssistantTextEvent(text=sentence, session_id=session_id).model_dump_json()):
                                            return
                                        yield sentence
                                remainder = sb.flush()
                                if remainder and websocket.client_state == WebSocketState.CONNECTED:
                                    await safe_send_text(websocket, AssistantTextEvent(text=remainder, session_id=session_id).model_dump_json())
                                    yield remainder
                            except (asyncio.CancelledError, WebSocketDisconnect):
                                log.info("sentence_generator przerwany (barge-in / disconnect).")
                                raise

                        await send_state("STREAMING_TTS", session_id)
                        
                        chunk_count = 0
                        # Piper TTS zwraca wygenerowane binarne chunki audio (PCM 16-bit) z otrzymywanych zdan
                        async for pcm_bytes in vox.stream_sentences(sentence_generator()):
                            if websocket.client_state != WebSocketState.CONNECTED:
                                log.info("Klient rozłączony podczas stream_sentences, przerywanie TTS.")
                                break
                            if chunk_count == 0:
                                await send_state("SPEAKING", session_id)
                            header = struct.pack(_HEADER_FMT, len(pcm_bytes))
                            if not await safe_send_bytes(websocket, header + pcm_bytes):
                                break
                            chunk_count += 1
                            
                        # Koniec strumienia (pusty chunk)
                        await safe_send_bytes(websocket, struct.pack(_HEADER_FMT, 0))
                        await send_state("STANDBY", session_id)

                    except (asyncio.CancelledError, WebSocketDisconnect):
                        log.info("Zadanie przetwarzania promptu przerwane (barge-in / disconnect).")
                    except Exception as e:
                        log.error("Błąd w trakcie przetwarzania: %s", e, exc_info=True)
                        await send_state("STANDBY", session_id)

                current_task = asyncio.create_task(process_prompt())

    except WebSocketDisconnect:
        log.info("Klient WS rozłączony.")
    except Exception as e:
        log.error("Krytyczny błąd WS: %s", e, exc_info=True)
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
            log.info("Anulowano aktywne zadanie po zamknięciu WebSocket.")
