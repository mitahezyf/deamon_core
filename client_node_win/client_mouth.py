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
            
            chunk_size = len(outdata)
            data_to_write = bytearray()
            
            # Pobieramy dane z kolejki az wypelnimy bufor sounddevice
            while len(data_to_write) < chunk_size:
                try:
                    # Pobieramy 1 blok
                    data = self.audio_queue.get_nowait()
                    data_to_write.extend(data)
                except queue.Empty:
                    break
            
            # Jesli nie ma wystarczajaco danych, wypelniamy reszte zerami (cisza)
            if len(data_to_write) < chunk_size:
                data_to_write.extend(b'\x00' * (chunk_size - len(data_to_write)))
            elif len(data_to_write) > chunk_size:
                # W praktyce powinnismy obsluzyc nadmiar, ale dla uproszczenia
                # zakladamy chunk_size z API == chunk_size z stream
                pass
                
            # Konwersja do float/int i zapis do outdata (outdata oczekuje numpy buffer)
            # Nasze audio to 16-bit PCM, sounddevice oczekuje int16 przy tym dtype
            import numpy as np
            audio_data = np.frombuffer(data_to_write[:chunk_size], dtype=np.int16)
            outdata[:] = audio_data.reshape(-1, self.channels)

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
            
    def shutdown(self):
        """Calkowite zamkniecie strumienia."""
        self.stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
