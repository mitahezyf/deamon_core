from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("audio.pipeline.v2")


class _EarsLike(Protocol):
    def process_audio_chunk(self, payload: bytes) -> dict: ...


class _SttLike(Protocol):
    is_loaded: bool

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate: int) -> str: ...


@dataclass
class EarsStreamSession:
    """State machine for WS `/ws/ears/listen` session.

    Keeps capture state and centralizes command handling to avoid endpoint spaghetti.
    """

    ears: _EarsLike
    stt: _SttLike
    is_capturing: bool = False
    sample_rate: int = settings.stt_sample_rate
    stt_enabled: bool = settings.stt_enabled
    audio_buffer: bytearray = field(default_factory=bytearray)
    last_voice_ts: float | None = None
    last_partial_ts: float = 0.0
    partial_text_last: str = ""

    def on_audio_bytes(self, payload: bytes) -> list[dict]:
        result = self.ears.process_audio_chunk(payload)
        events = [result]
        now = time.monotonic()

        if self.is_capturing:
            self.audio_buffer.extend(payload)
            if self._is_speech_chunk(payload):
                self.last_voice_ts = now

            events.extend(self._maybe_emit_partial(now))

            if self._should_auto_stop(now):
                self.is_capturing = False
                transcript_event = self.flush_stt()
                transcript_event["stop_reason"] = "silence_timeout"
                self.audio_buffer.clear()
                events.append(transcript_event)

        should_auto_capture = (
            self.stt_enabled
            and result.get("event") == "wake_word_detected"
            and not self.is_capturing
        )
        if should_auto_capture:
            self.is_capturing = True
            self.audio_buffer.clear()
            self.last_voice_ts = None
            self.last_partial_ts = 0.0
            self.partial_text_last = ""
            events.append({"event": "capture_started", "source": "wake_word"})

        return events

    def on_command(self, text_payload: str) -> list[dict]:
        command = text_payload.strip().lower()

        if command == "start_capture":
            self.is_capturing = True
            self.audio_buffer.clear()
            self.last_voice_ts = None
            self.last_partial_ts = 0.0
            self.partial_text_last = ""
            return [{"event": "capture_started", "source": "manual"}]

        if command == "stop_capture":
            self.is_capturing = False
            result = self.flush_stt()
            self.audio_buffer.clear()
            return [result]

        if command.startswith("set_sample_rate:"):
            raw_rate = command.split(":", maxsplit=1)[1]
            try:
                rate = int(raw_rate)
                if rate <= 0:
                    raise ValueError("sample_rate must be positive")
            except ValueError:
                return [{"event": "stt_error", "message": "invalid_sample_rate"}]

            self.sample_rate = rate
            return [{"event": "sample_rate_set", "sample_rate": self.sample_rate}]

        if command == "flush_stt":
            result = self.flush_stt()
            self.audio_buffer.clear()
            return [result]

        return [{"event": "unknown_command", "command": command}]

    def flush_stt(self) -> dict:
        if not self.audio_buffer:
            return {
                "event": "transcript",
                "text": "",
                "final": True,
                "reject_reason": "empty_audio",
            }

        if not self.stt.is_loaded:
            return {
                "event": "stt_error",
                "message": "stt_not_ready",
                "sample_rate": self.sample_rate,
            }

        try:
            text = self.stt.transcribe_pcm16(
                bytes(self.audio_buffer),
                sample_rate=self.sample_rate,
            )
            response: dict[str, object] = {
                "event": "transcript",
                "text": text,
                "final": True,
            }
            if not text:
                response["reject_reason"] = self._build_reject_reason()
            return response
        except Exception as exc:
            log.error("Blad STT websocket: %s", exc, exc_info=True)
            return {"event": "stt_error", "message": str(exc)}

    def _build_reject_reason(self) -> str:
        if self.sample_rate <= 0:
            return "invalid_sample_rate"

        pcm = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16)
        if pcm.size == 0:
            return "empty_audio"

        duration_seconds = pcm.size / float(self.sample_rate)
        if duration_seconds < settings.stt_min_seconds:
            return "too_short_audio"

        rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
        if rms < settings.stt_min_rms:
            return "too_quiet_audio"

        return "no_speech_detected"

    def _is_speech_chunk(self, payload: bytes) -> bool:
        pcm = np.frombuffer(payload, dtype=np.int16)
        if pcm.size == 0:
            return False
        rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
        return rms >= settings.stt_min_rms

    def _should_auto_stop(self, now: float) -> bool:
        if self.last_voice_ts is None:
            return False
        return (now - self.last_voice_ts) >= settings.stt_silence_stop_seconds

    def _maybe_emit_partial(self, now: float) -> list[dict]:
        if not settings.stt_partial_enabled:
            return []
        if not self.stt.is_loaded:
            return []
        if self.sample_rate <= 0:
            return []

        audio_duration = len(self.audio_buffer) / 2.0 / float(self.sample_rate)
        if audio_duration < settings.stt_partial_min_seconds:
            return []
        if (now - self.last_partial_ts) < settings.stt_partial_interval_seconds:
            return []

        self.last_partial_ts = now
        try:
            text = self.stt.transcribe_pcm16(
                bytes(self.audio_buffer), sample_rate=self.sample_rate
            )
        except Exception:
            return []

        if text and text != self.partial_text_last:
            self.partial_text_last = text
            return [{"event": "partial_transcript", "text": text, "final": False}]
        return []
