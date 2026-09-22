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
        self._playback_thread_handle = None
        self._is_running = True
        
    @property
    def is_playing(self) -> bool:
        """Zwraca True, jesli odtwarzacz ma dane w buforze lub trwa generacja."""
        return not self.audio_queue.empty() or len(self._audio_buffer) > 0
        
    def start(self):
        """Uruchamia watek zarzadzajacy odtwarzaniem."""
        if self._playback_thread_handle is None:
            self._playback_thread_handle = threading.Thread(target=self._playback_worker, daemon=True)
            self._playback_thread_handle.start()
            log.info("ClientMouth odtwarzacz gotowy (dynamiczne zarzadzanie strumieniem).")
            
    def play_chunk(self, pcm_bytes: bytes):
        """Wrzuca paczke PCM z WebSocketa do kolejki odtwarzania."""
        self.audio_queue.put(pcm_bytes)

    def finish_stream(self):
        """Sygnalizuje koniec partii audio."""
        self.audio_queue.put(b"")

    def stop(self):
        """Barge-in: czysci kolejke i przerywa aktualne odtwarzanie."""
        log.info("ClientMouth Barge-in: Czyszczenie bufora audio!")
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self._audio_buffer.clear()
        self.audio_queue.put(b"") # Wymus przerwanie aktualnego strumienia
            
    def shutdown(self):
        """Calkowite zamkniecie strumienia."""
        self.stop()
        self._is_running = False
        self.audio_queue.put(None)
        if self._playback_thread_handle:
            self._playback_thread_handle.join(timeout=2.0)

    def _playback_worker(self):
        """Watek asynchroniczny: otwiera strumien tylko wtedy, gdy jest audio."""
        import numpy as np
        
        while self._is_running:
            item = self.audio_queue.get()
            if item is None:
                break
            if item == b"":
                continue
                
            self._audio_buffer.extend(item)
            
            event = threading.Event()
            end_of_stream = False
            
            def callback(outdata, frames, time, status):
                nonlocal end_of_stream
                if status:
                    log.warning("Sounddevice status: %s", status)
                outdata.fill(0)
                
                chunk_size_bytes = frames * self.channels * 2
                
                while len(self._audio_buffer) < chunk_size_bytes:
                    try:
                        data = self.audio_queue.get_nowait()
                        if data is None:
                            self._is_running = False
                            event.set()
                            raise sd.CallbackStop()
                        elif data == b"":
                            end_of_stream = True
                        else:
                            self._audio_buffer.extend(data)
                    except queue.Empty:
                        break
                        
                bytes_to_read = min(chunk_size_bytes, len(self._audio_buffer))
                if bytes_to_read > 0:
                    audio_data = np.frombuffer(self._audio_buffer[:bytes_to_read], dtype=np.int16)
                    outdata[:len(audio_data)] = audio_data.reshape(-1, self.channels)
                    del self._audio_buffer[:bytes_to_read]
                elif end_of_stream:
                    event.set()
                    raise sd.CallbackStop()
                    
            try:
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16',
                    callback=callback,
                    device=None  # Wymusza pobranie aktualnego domyslnego urzadzenia Windowsa
                ):
                    event.wait()
            except Exception as e:
                log.error("Blad uruchamiania strumienia audio: %s", e)
