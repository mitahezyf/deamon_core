import asyncio
import websockets
import json
import logging
import struct

from client_mouth import ClientMouth
from client_eyes import capture_screen_base64
from client_ears import ClientEars

# Prosta konfiguracja logowania dla klienta Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-15s - %(message)s"
)
log = logging.getLogger("client.main")

DAEMON_URL = "ws://192.168.0.103:8000/api/v1/ws/agent" # Zmień na IP swojego serwera LXC
_HEADER_FMT = ">I"

async def ws_loop():
    mouth = ClientMouth()
    mouth.start()
    
    ears = ClientEars()
    
    while True:
        try:
            log.info("Próba połączenia z serwerem: %s", DAEMON_URL)
            async with websockets.connect(DAEMON_URL) as ws:
                log.info("Połączono z DAEMON!")
                
                # Nasłuch z mikrofonu w pętli
                async def mic_input():
                    while True:
                        text = await ears.listen_for_command()
                        if not text:
                            continue
                            
                        # Sprawdzamy barge-in
                        if mouth.is_playing:
                            log.info("Barge-in: Przerywam mowienie asystenta!")
                            mouth.stop()
                            await ws.send(json.dumps({
                                "event_type": "abort_generation",
                                "session_id": "win_client"
                            }))
                            
                        payload = {
                            "event_type": "user_prompt",
                            "text": text,
                            "session_id": "win_client"
                        }
                        await ws.send(json.dumps(payload))
                
                input_task = asyncio.create_task(mic_input())
                
                # Odbiór wiadomości
                while True:
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
                                # Tutaj logika np. ściszenia systemu Windows
                                
                        except json.JSONDecodeError:
                            log.error("Nieprawidłowy JSON od serwera.")
                            
                    elif isinstance(msg, bytes):
                        # Wiadomość binarna (Audio PCM)
                        if len(msg) >= 4:
                            # Rozpakowanie nagłówka (4 bajty = długość)
                            header = msg[:4]
                            length = struct.unpack(_HEADER_FMT, header)[0]
                            
                            if length == 0:
                                log.debug("Koniec strumienia audio.")
                            else:
                                pcm_data = msg[4:]
                                mouth.play_chunk(pcm_data)
                        
        except websockets.exceptions.ConnectionClosedError:
            log.warning("Połączenie zerwane. Próba ponownego połączenia za 3s...")
            await asyncio.sleep(3)
        except ConnectionRefusedError:
            log.warning("Serwer niedostępny. Próba ponownego połączenia za 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("Nieoczekiwany błąd: %s", e, exc_info=True)
            await asyncio.sleep(3)
            
        finally:
            # Anulowanie taska input() przy rozłączeniu
            if 'input_task' in locals() and not input_task.done():
                input_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(ws_loop())
    except KeyboardInterrupt:
        log.info("Zamykanie klienta Windows...")
