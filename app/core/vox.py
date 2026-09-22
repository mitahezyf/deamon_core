"""
app/core/vox.py — Silnik mowy DAEMON

ETAP AKTUALNY: Piper TTS ONNX CPU (Etap 2 wdrożony)

Wymagania:
  - Piper TTS z modelem pl_PL-darkman-medium.onnx
  - Asynchroniczny generator PCM 22050 Hz, 16-bit mono
  - Zero torch, zero CUDA, zero Coqui TTS
"""

import asyncio
import io
import re
import wave
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("vox")


class DaemonVox:
    """
    Silnik TTS wykorzystujący Piper ONNX na CPU.
    """

    def __init__(self) -> None:
        self._voice: Optional[PiperVoice] = None
        self._loaded: bool = False
        log.debug("DaemonVox zainicjalizowany (Piper TTS)")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """
        Laduje model Piper TTS ONNX synchronicznie.
        """
        if PiperVoice is None:
            log.error("Biblioteka 'piper' (piper-tts) nie jest zainstalowana!")
            return

        model_path = settings.piper_model_path
        config_path = settings.piper_config_path

        if not model_path or not model_path.exists():
            log.error("DaemonVox.load(): Model Piper nie znaleziony w %s.", model_path)
            return
            
        log.info("Ladowanie modelu Piper TTS z: %s", model_path)
        self._voice = PiperVoice.load(
            str(model_path), 
            config_path=str(config_path) if config_path else None, 
            use_cuda=False
        )
        self._loaded = True
        log.info("DaemonVox (Piper TTS) gotowy do pracy.")

    def synthesize_raw(self, text: str) -> bytes:
        """
        Synteza pojedynczego zdania do czystych bajtow PCM 16-bit mono 22050 Hz.
        Wykonuje obliczenia w obecnym watku (blokujaco). Do uzycia w thread poolu.
        """
        if not self._loaded or not self._voice:
            log.warning("DaemonVox nie jest zaladowany. Synteza anulowana.")
            return b""
            
        pcm_data = bytearray()
        # synthesize() dzieli tekst na zdania i generuje AudioChunk per zdanie
        for chunk in self._voice.synthesize(text):
            audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
            pcm_data.extend(audio_int16.tobytes())
            
        return bytes(pcm_data)

    async def stream_sentences(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Dzielenie strumienia na pelne zdania (Sentence Boundary: [.!?\n]) 
        i natychmiastowe wypychanie wygenerowanych chunkow PCM w locie.
        """
        if not self._loaded or not self._voice:
            log.warning("DaemonVox nie jest zaladowany.")
            return

        buffer = ""
        # Uproszczony boundary detector - dzieli po kropce, wykrzykniku, znaku zapytania lub nowej linii
        boundary_pattern = re.compile(r'([.!?\n]+)')
        
        async for text_chunk in text_stream:
            buffer += text_chunk
            
            while True:
                match = boundary_pattern.search(buffer)
                if not match:
                    break
                    
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                buffer = buffer[end_pos:]
                
                if len(sentence) > 1:
                    log.debug("Synteza zdania w tle: %r...", sentence[:30])
                    # Wykonujemy blokujaca synteze w oddzielnym watku (poza Event Loopem)
                    pcm_bytes = await asyncio.to_thread(self.synthesize_raw, sentence)
                    if pcm_bytes:
                        yield pcm_bytes

        # Jesli zostalo cos w buforze po zakonczeniu strumienia
        sentence = buffer.strip()
        if len(sentence) > 1:
            log.debug("Synteza koncowki: %r...", sentence[:30])
            pcm_bytes = await asyncio.to_thread(self.synthesize_raw, sentence)
            if pcm_bytes:
                yield pcm_bytes

    def synthesize_to_file(self, text: str, output_path: Path) -> dict:
        """
        Syntetyzuje tekst i zapisuje WAV (uzywane przez REST API).
        """
        import time
        t0 = time.perf_counter()
        
        if not self._loaded or not self._voice:
            log.warning("DaemonVox nie jest zaladowany.")
            return {
                "latency_first_chunk": 0.0,
                "total_time": 0.0,
                "audio_duration": 0.0,
                "output": str(output_path),
            }

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16 bit
            wav_file.setframerate(self._voice.config.sample_rate)
            
            t_first = None
            total_samples = 0
            
            for chunk in self._voice.synthesize(text):
                if t_first is None:
                    t_first = time.perf_counter() - t0
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
                total_samples += len(audio_int16)

        with open(output_path, "wb") as f:
            f.write(wav_io.getvalue())
            
        elapsed = time.perf_counter() - t0
        dur = total_samples / self._voice.config.sample_rate

        return {
            "latency_first_chunk": round(t_first or 0.0, 3),
            "total_time": round(elapsed, 3),
            "audio_duration": round(dur, 2),
            "output": str(output_path),
        }
