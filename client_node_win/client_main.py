import os
import sys
from pathlib import Path

# Bezwzględnie na samym początku przed jakimkolwiek importem ctranslate2 / faster_whisper / torch:
if sys.platform == "win32":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    cublas_bin = site_packages / "nvidia" / "cublas" / "bin"
    cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"
    for p in [cublas_bin, cudnn_bin]:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

import asyncio
import websockets
import json
import logging
import struct

from client_mouth import ClientMouth
from client_eyes import capture_screen_base64
from client_ears import ClientEars
from client_actions import ActionExecutor

# Prosta konfiguracja logowania dla klienta Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-15s - %(message)s"
)
log = logging.getLogger("client.main")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import client_settings

DAEMON_URL = client_settings.daemon_server_url
_HEADER_FMT = ">I"

async def ws_loop():
    mouth = ClientMouth()
    mouth.start()
    
    ears = ClientEars()
    action_executor = ActionExecutor()
    
    active_ws = None
    input_task = None
    last_text = None
    last_text_time = 0.0

    while True:
        try:
            # 1. Anulowanie poprzedniego taska nasłuchu przed nową próbą
            if input_task and not input_task.done():
                input_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(input_task), timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
                input_task = None

            # 2. Bezpieczne zamknięcie poprzedniego gniazda jeśli nadal otwarte
            if active_ws and not active_ws.closed:
                try:
                    await active_ws.close()
                except Exception:
                    pass
                active_ws = None

            log.info("Próba połączenia z serwerem: %s", DAEMON_URL)
            async with websockets.connect(DAEMON_URL) as ws:
                active_ws = ws
                log.info("Połączono z DAEMON!")
                
                # Nasłuch z mikrofonu w pętli
                async def mic_input():
                    nonlocal last_text, last_text_time
                    try:
                        while not ws.closed:
                            text = await ears.listen_for_command(is_muted_func=lambda: mouth.is_playing)
                            if not text or ws.closed:
                                continue

                            # Debounce po stronie klienta (eliminacja podwójnego STT w krótkim oknie)
                            now = asyncio.get_event_loop().time()
                            if text == last_text and (now - last_text_time) < 1.5:
                                log.warning("Debounce STT klienta: zignorowano zduplikowany tekst '%s' w odstępie < 1.5s", text)
                                continue
                            last_text = text
                            last_text_time = now
                                
                            # Sprawdzamy barge-in
                            if mouth.is_playing:
                                log.info("Barge-in: Przerywam mowienie asystenta!")
                                mouth.stop()
                                await ws.send(json.dumps({
                                    "event_type": "abort_generation",
                                    "session_id": "win_client"
                                }))
                                
                            log.info(f"Otrzymano tekst z mikrofonu: {text}")
                            # Weryfikacja przez lokalny Action Matrix
                            was_handled, tts_text = await asyncio.to_thread(action_executor.execute, text)
                            
                            if was_handled:
                                log.info(f"Akcja wykonana lokalnie. Oczekiwany TTS: {tts_text}")
                                # Pomin LLM, zapytaj serwer wylacznie o szybki TTS
                                await ws.send(json.dumps({
                                    "event_type": "tts_request",
                                    "text": tts_text,
                                    "session_id": "win_client"
                                }))
                                continue

                            payload = {
                                "event_type": "user_prompt",
                                "text": text,
                                "session_id": "win_client"
                            }
                            await ws.send(json.dumps(payload))
                    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                        log.info("mic_input zakończony (rozłączenie / anulowanie).")

                input_task = asyncio.create_task(mic_input())
                
                # Odbiór wiadomości
                try:
                    while not ws.closed:
                        msg = await ws.recv()
                        
                        if isinstance(msg, str):
                            # Wiadomość tekstowa (JSON)
                            try:
                                data = json.loads(msg)
                                event_type = data.get("event_type")
                                
                                if event_type == "state_change":
                                    log.info("DAEMON Stan -> %s", data.get("new_state"))
                                    
                                elif event_type == "assistant_text":
                                    print(f"\n[DAEMON]: {data.get('text')}\n")
                                
                                elif event_type == "request_frame":
                                    log.info("DAEMON poprosił o obraz. Wykonywanie zrzutu...")
                                    b64_img = capture_screen_base64()
                                    await ws.send(json.dumps({
                                        "event_type": "frame_response",
                                        "image_b64": b64_img,
                                        "session_id": data.get("session_id", "win_client")
                                    }))
                                    
                                elif event_type == "exec_action":
                                    log.info("=== WYKONANIE AKCJI SYSTEMOWEJ: %s ===", data.get("action"))
                                    
                            except json.JSONDecodeError:
                                log.error("Nieprawidłowy JSON od serwera.")
                                
                        elif isinstance(msg, bytes):
                            # Wiadomość binarna (Audio PCM)
                            if len(msg) >= 4:
                                header = msg[:4]
                                length = struct.unpack(_HEADER_FMT, header)[0]
                                
                                if length == 0:
                                    log.debug("Koniec strumienia audio.")
                                    mouth.finish_stream()
                                else:
                                    pcm_data = msg[4:]
                                    mouth.play_chunk(pcm_data)
                finally:
                    if input_task and not input_task.done():
                        input_task.cancel()
                        try:
                            await asyncio.wait_for(asyncio.shield(input_task), timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                            pass
                        input_task = None
                        
        except websockets.exceptions.ConnectionClosedError:
            log.warning("Połączenie zerwane. Próba ponownego połączenia za 3s...")
            await asyncio.sleep(3)
        except ConnectionRefusedError:
            log.warning("Serwer niedostępny. Próba ponownego połączenia za 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("Nieoczekiwany błąd w pętli klienta: %s", e, exc_info=True)
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(ws_loop())
    except KeyboardInterrupt:
        log.info("Zamykanie klienta Windows...")
