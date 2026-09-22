import asyncio
import numpy as np
import sounddevice as sd
import logging
import threading
import queue

log = logging.getLogger("client.mouth")

class ClientMouth:
    """Odtwarzacz PCM na stacji Windows."""
    def __init__(self, sample_rate=22050, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        # Kolejka wewnetrzna (watek-bezpieczna) do przekazywania PCM z asyncio do callbacku C
        self.audio_queue = queue.Queue()
        self._audio_buffer = bytearray()
        self.stream = None
        
    @property
    def is_playing(self) -> bool:
        """Zwraca True, jesli odtwarzacz ma dane w buforze."""
        return not self.audio_queue.empty()
        
    def start(self):
        """Uruchamia odtwarzanie."""
        if self.stream is not None:
            return
            
        def callback(outdata, frames, time, status):
            if status:
                log.warning("Sounddevice status: %s", status)
            
            outdata.fill(0)
            chunk_size_bytes = frames * self.channels * 2  # 16-bit = 2 bajty na próbkę
            
            # Pobieramy dane z kolejki aż wypełnimy potrzebny rozmiar
            while len(self._audio_buffer) < chunk_size_bytes:
                try:
                    data = self.audio_queue.get_nowait()
                    self._audio_buffer.extend(data)
                except queue.Empty:
                    break
            
            bytes_to_read = min(chunk_size_bytes, len(self._audio_buffer))
            
            if bytes_to_read > 0:
                # Nasze audio to 16-bit PCM, sounddevice oczekuje int16 przy tym dtype
                import numpy as np
                audio_data = np.frombuffer(self._audio_buffer[:bytes_to_read], dtype=np.int16)
                num_elements = len(audio_data)
                outdata[:num_elements] = audio_data.reshape(-1, self.channels)
                
                # Usuwamy zużyte bajty
                del self._audio_buffer[:bytes_to_read]

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=callback
        )
        self.stream.start()
        log.info("ClientMouth odtwarzacz gotowy (urzadzenie domyslne).")

    def play_chunk(self, pcm_bytes: bytes):
        """Wrzuca paczke PCM z WebSocketa do kolejki odtwarzania."""
        self.audio_queue.put(pcm_bytes)

    def stop(self):
        """Barge-in: czysci kolejke i przerywa aktualne odtwarzanie."""
        log.info("ClientMouth Barge-in: Czyszczenie bufora audio!")
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self._audio_buffer.clear()
            
    def shutdown(self):
        """Calkowite zamkniecie strumienia."""
        self.stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
