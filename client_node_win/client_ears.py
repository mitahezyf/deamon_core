import asyncio
import logging
import queue
import threading
import numpy as np
import sounddevice as sd

import os
import sys

# Dynamiczne ladowanie DLL dla CUDA (CTranslate2 / faster-whisper) na Windows
if os.name == 'nt':
    try:
        import site
        # Zaleznosci moga byc w bin lub lib
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            for pkg in ["nvidia\\cublas\\bin", "nvidia\\cublas\\lib", "nvidia\\cudnn\\bin", "nvidia\\cudnn\\lib"]:
                dll_path = os.path.join(sp, pkg)
                if os.path.exists(dll_path):
                    os.add_dll_directory(dll_path)
    except Exception:
        pass

try:
    from faster_whisper import WhisperModel
    import openwakeword
    from openwakeword.model import Model as OWWModel
    openwakeword.utils.download_models()
    from silero_vad import load_silero_vad, VADIterator
except ImportError:
    log = logging.getLogger("client.ears")
    log.warning("Brak openwakeword / faster-whisper / silero-vad! Uruchom pip install -r requirements_win.txt")
    OWWModel = None
    WhisperModel = None
    load_silero_vad = None
    VADIterator = None

log = logging.getLogger("client.ears")

class ClientEars:
    def __init__(self, wake_word="alexa", stt_model="small", device="cuda"):
        self.sample_rate = 16000
        self.chunk_size = 1280
        self.audio_queue = queue.Queue()
        self._is_listening = False
        self._active_lock = threading.Lock()
        self._stream = None
        self.wake_word_name = wake_word
        
        if OWWModel and WhisperModel and load_silero_vad:
            log.info("Ladowanie modelu openWakeWord...")
            self.oww_model = OWWModel(inference_framework="onnx")
            
            log.info("Ladowanie modelu Silero VAD (ONNX CPU)...")
            self.vad_model = load_silero_vad(onnx=True)
            self.vad_iterator = VADIterator(self.vad_model, sampling_rate=16000, threshold=0.5, min_silence_duration_ms=700)
            
            self._stt_model_name = stt_model
            try:
                log.info(f"Ladowanie faster-whisper ({stt_model} na {device})...")
                self.stt_model = WhisperModel(stt_model, device=device, compute_type="float16", local_files_only=True)
                _ = self.stt_model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=5, language="pl", vad_filter=False)
            except Exception as e:
                log.warning(f"Błąd inicjalizacji/testu Whisper na {device}: {e}. Fallback na CPU (int8)...")
                self.stt_model = WhisperModel(stt_model, device="cpu", compute_type="int8")
                
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

    async def listen_for_command(self, is_muted_func=None) -> str:
        """
        Nasluch mikrofonu. Najpierw oczekuje Wake Word.
        Po aktywacji nagrywa az do wykrycia ciszy (prosty VAD).
        Nastepnie transkrybuje przez faster-whisper.
        """
        if not OWWModel:
            await asyncio.sleep(1)
            return ""

        if not self._active_lock.acquire(blocking=False):
            log.warning("Odrzucono nakladajacy sie nasluch (wyscig watkow).")
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
                    
                if is_muted_func and is_muted_func():
                    # Ignorujemy ramki (zabezpieczenie przed samowybudzeniem / fałszywym barge-in)
                    continue
                    
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(audio_data)
                
                for mdl, score in prediction.items():
                    if score > 0.5:
                        ww_detected = True
                        log.info(f"WakeWord WYKRYTY! (score: {score:.2f})")
                        break

            # Nagrywanie po aktywacji - Silero VAD
            log.info("Nasluchiwanie komendy (mow teraz)...")
            command_audio = []
            self.vad_iterator.reset_states()
            speech_started = False
            speech_ended = False
            
            vad_buffer = np.array([], dtype=np.float32)
            
            while not speech_ended:
                chunk = await asyncio.to_thread(self._get_audio_chunk)
                if not chunk:
                    continue
                    
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                command_audio.append(audio_data)
                
                audio_float32 = audio_data.astype(np.float32) / 32768.0
                vad_buffer = np.concatenate((vad_buffer, audio_float32))
                
                while len(vad_buffer) >= 512:
                    vad_chunk = vad_buffer[:512]
                    vad_buffer = vad_buffer[512:]
                    
                    speech_dict = self.vad_iterator(vad_chunk)
                    if speech_dict:
                        if 'start' in speech_dict:
                            speech_started = True
                        elif 'end' in speech_dict:
                            log.info("Koniec komendy (wykryto ciszę przez Silero VAD).")
                            speech_ended = True
                            break
                            
                if speech_ended:
                    break
                        
                # Timeout jesli przez 5s nic nie powiedziano
                if not speech_started and len(command_audio) > (5.0 * self.sample_rate / self.chunk_size):
                    log.info("Koniec komendy (brak mowy).")
                    break
                    
                # Hard limit 15 sekund
                if len(command_audio) > (15.0 * self.sample_rate / self.chunk_size):
                    log.info("Osiągnieto maksymalny czas komendy (15s).")
                    break
                    
            self.vad_iterator.reset_states()

            log.info("Transkrypcja audio...")
            full_audio = np.concatenate(command_audio)
            
            # faster-whisper oczekuje znormalizowanego float32 [-1.0, 1.0]
            audio_float32 = full_audio.astype(np.float32) / 32768.0
            
            def transcribe():
                try:
                    segments, info = self.stt_model.transcribe(audio_float32, beam_size=5, language="pl", vad_filter=False)
                    return " ".join([s.text for s in segments]).strip()
                except Exception:
                    # Natychmiastowy bezglosny fallback na CPU
                    self.stt_model = WhisperModel(self._stt_model_name, device="cpu", compute_type="int8", local_files_only=True)
                    segments, info = self.stt_model.transcribe(audio_float32, beam_size=5, language="pl", vad_filter=False)
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
            
            # Wyczyść resztki audio po nagraniu z kolejki, by wyeliminować fałszywe powtórzenia
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            # Oraz zresetuj stan detektora Wake Word
            if hasattr(self.oww_model, "reset"):
                self.oww_model.reset()
                
            self._active_lock.release()
            
        return command_text

