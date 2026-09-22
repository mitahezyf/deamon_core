import logging

log = logging.getLogger("client.ears")

class ClientEars:
    """
    Szkielet dla modułu słuchającego na stacji Windows.
    Będzie korzystał z mikrofonu poprzez pyaudio/sounddevice,
    nasłuchiwał "WakeWord" (openWakeWord), 
    wycinał ciszę (Silero VAD) i transkrybował mowę (faster-whisper GPU/CPU).
    """
    
    def __init__(self):
        self._is_listening = False
        log.info("ClientEars zainicjalizowany (Szkielet).")
        
    def start_listening(self, callback):
        """
        Uruchamia nasłuch w tle. Callback będzie wywoływany z gotowym
        tekstem transkrypcji (stringiem), np. po wykryciu końca wypowiedzi.
        """
        self._is_listening = True
        log.info("Ears: Nasłuch uruchomiony. (Wymaga implementacji pętli audio)")
        
    def stop_listening(self):
        self._is_listening = False
        log.info("Ears: Nasłuch zatrzymany.")
