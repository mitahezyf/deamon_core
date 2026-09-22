import base64
import io
from PIL import Image
import mss
import logging

log = logging.getLogger("client.eyes")

def capture_screen_base64() -> str:
    """
    Robi zrzut pierwszego aktywnego ekranu, zapisuje do RAM jako JPEG i koduje w Base64.
    Bez zapisywania pliku na dysku.
    """
    log.debug("Wykonywanie zrzutu ekranu...")
    with mss.mss() as sct:
        # Zrzut glownego (pierwszego) ekranu
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        
        # Konwersja na format PIL
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # Opcjonalne skalowanie (np. do max 1920x1080 zeby oszczedzac transfer i tokeny)
        # img.thumbnail((1920, 1080))
        
        # Zapis do bufora w pamieci
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        
        # Konwersja do base64
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        log.debug("Zrzut wykonany (rozmiar base64: %d bajtow)", len(base64_data))
        return base64_data
