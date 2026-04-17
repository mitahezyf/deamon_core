from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

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

    def on_audio_bytes(self, payload: bytes) -> list[dict]:
        result = self.ears.process_audio_chunk(payload)
        events = [result]

        if self.is_capturing:
            self.audio_buffer.extend(payload)

        should_auto_capture = (
            self.stt_enabled
            and result.get("event") == "wake_word_detected"
            and not self.is_capturing
        )
        if should_auto_capture:
            self.is_capturing = True
            self.audio_buffer.clear()
            events.append({"event": "capture_started", "source": "wake_word"})

        return events

    def on_command(self, text_payload: str) -> list[dict]:
        command = text_payload.strip().lower()

        if command == "start_capture":
            self.is_capturing = True
            self.audio_buffer.clear()
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
            return {"event": "transcript", "text": "", "final": True}

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
            return {"event": "transcript", "text": text, "final": True}
        except Exception as exc:
            log.error("Blad STT websocket: %s", exc, exc_info=True)
            return {"event": "stt_error", "message": str(exc)}
