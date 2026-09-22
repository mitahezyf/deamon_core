import asyncio
import logging
import queue
import numpy as np
import sounddevice as sd

try:
    from faster_whisper import WhisperModel
    import openwakeword
    from openwakeword.model import Model as OWWModel
    openwakeword.utils.download_models()
except ImportError:
    log = logging.getLogger("client.ears")
    log.warning("Brak openwakeword / faster-whisper! Uruchom pip install -r requirements_win.txt")
    OWWModel = None
    WhisperModel = None

log = logging.getLogger("client.ears")

class ClientEars:
    def __init__(self, wake_word="alexa", stt_model="base", device="cuda"):
        self.sample_rate = 16000
        self.chunk_size = 1280
        self.audio_queue = queue.Queue()
        self._is_listening = False
        self._stream = None
        self.wake_word_name = wake_word
        
        if OWWModel and WhisperModel:
            log.info("Ladowanie modelu openWakeWord...")
            self.oww_model = OWWModel(inference_framework="onnx")
            
            log.info(f"Ladowanie faster-whisper ({stt_model} na {device})...")
            self.stt_model = WhisperModel(stt_model, device=device, compute_type="float16")
            log.info("ClientEars gotowy.")
        else:
            log.error("Nie udalo sie zainicjalizowac modeli AI z powodu braku zaleznosci.")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            log.warning("Sounddevice status: %s", status)
        if self._is_listening:
            self.audio_queue.put(bytes(indata))

    def _get_audio_chunk(self):
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return b""

    async def listen_for_command(self) -> str:
        """
        Nasluch mikrofonu. Najpierw oczekuje Wake Word.
        Po aktywacji nagrywa az do wykrycia ciszy (prosty VAD).
        Nastepnie transkrybuje przez faster-whisper.
        """
        if not OWWModel:
            await asyncio.sleep(1)
            return ""

        self._is_listening = True
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        self._stream.start()

        command_text = ""
        try:
            log.info(f"Oczekiwanie na WakeWord '{self.wake_word_name}'...")
            ww_detected = False
            
            # Pętla nasłuchu na WakeWord
            while not ww_detected:
                chunk = await asyncio.to_thread(self._get_audio_chunk)
                if not chunk:
                    continue
                    
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(audio_data)
                
                for mdl, score in prediction.items():
                    if score > 0.5:
                        ww_detected = True
                        log.info(f"WakeWord WYKRYTY! (score: {score:.2f})")
                        break

            # Nagrywanie po aktywacji - prosty VAD (Voice Activity Detection)
            log.info("Nasluchiwanie komendy (mow teraz)...")
            command_audio = []
            silence_chunks = 0
            max_silence_chunks = int(1.2 * (self.sample_rate / self.chunk_size)) # 1.2 sekundy ciszy = koniec komendy
            rms_threshold = 300 # prog ciszy
            
            while True:
                chunk = await asyncio.to_thread(self._get_audio_chunk)
                if not chunk:
                    continue
                    
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                command_audio.append(audio_data)
                
                # Obliczanie glosnosci
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                if rms < rms_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0
                    
                if silence_chunks > max_silence_chunks:
                    log.info("Koniec komendy (wykryto ciszę).")
                    break
                    
                # Hard limit 15 sekund
                if len(command_audio) > (15.0 * self.sample_rate / self.chunk_size):
                    log.info("Osiągnieto maksymalny czas komendy (15s).")
                    break

            log.info("Transkrypcja audio...")
            full_audio = np.concatenate(command_audio)
            
            # faster-whisper oczekuje znormalizowanego float32 [-1.0, 1.0]
            audio_float32 = full_audio.astype(np.float32) / 32768.0
            
            def transcribe():
                segments, info = self.stt_model.transcribe(audio_float32, beam_size=5, language="pl")
                return " ".join([s.text for s in segments]).strip()
                
            command_text = await asyncio.to_thread(transcribe)
            log.info(f"STT Wynik: {command_text}")

        except Exception as e:
            log.error(f"Błąd podczas nasłuchu: {e}", exc_info=True)
        finally:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._is_listening = False
            
        return command_text

