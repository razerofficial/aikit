"""
Energy VAD state machine.

Accept raw PCM chunks, emit structured events when speech starts or stops.
No WebSocket, no asyncio, no I/O — purely synchronous. The caller is
responsible for generating item IDs and routing the emitted events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import protocol

# Audio format advertised to clients (PCM16 mono, OpenAI Realtime default)
AUDIO_RATE = 24_000
CHUNK_MS = 100
CHUNK_BYTES = AUDIO_RATE * 2 * CHUNK_MS // 1000  # 4800 bytes = 100 ms

# Energy VAD (same approach as omni_agent.py, tuned for 24 kHz)
VAD_THRESHOLD = 200       # RMS amplitude 0–32767
VAD_ONSET_CHUNKS = 3      # 300 ms of speech to trigger start
VAD_OFFSET_CHUNKS = 20    # 2 s of silence to trigger end


@dataclass
class SpeechStarted:
    item_id: str


@dataclass
class SpeechStopped:
    item_id: str
    audio: bytes          # accumulated PCM for this utterance
    duration_ms: int


VADEvent = SpeechStarted | SpeechStopped


class VADBuffer:
    """
    Accumulate raw PCM bytes, slice into CHUNK_BYTES-sized frames, run
    energy VAD on each frame, and return a list of VADEvent objects.

    Caller is responsible for generating item IDs and routing events.
    No asyncio, no side-effects — purely synchronous.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._speech_buf = bytearray()
        self._state: Literal["silence", "speech"] = "silence"
        self._onset = 0
        self._offset = 0
        self._item_id: str | None = None

    def push(self, pcm: bytes) -> list[VADEvent]:
        """
        Feed raw PCM bytes. Returns a (possibly empty) list of VADEvents.
        Call this for every input_audio_buffer.append payload.
        """
        events: list[VADEvent] = []
        self._buf.extend(pcm)
        while len(self._buf) >= CHUNK_BYTES:
            chunk = bytes(self._buf[:CHUNK_BYTES])
            del self._buf[:CHUNK_BYTES]
            events.extend(self._process_chunk(chunk))
        return events

    def _process_chunk(self, chunk: bytes) -> list[VADEvent]:
        """Classify one CHUNK_BYTES frame; update state; return 0–1 events."""
        events: list[VADEvent] = []
        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

        if rms >= VAD_THRESHOLD:
            self._onset += 1
            self._offset = 0
        else:
            self._offset += 1
            self._onset = 0

        # Transition: silence → speech
        if self._state == "silence" and self._onset >= VAD_ONSET_CHUNKS:
            self._state = "speech"
            self._speech_buf.clear()
            self._item_id = protocol.iid()
            events.append(SpeechStarted(item_id=self._item_id))

        if self._state == "speech":
            self._speech_buf.extend(chunk)

            # Transition: speech → silence
            if self._offset >= VAD_OFFSET_CHUNKS:
                self._state = "silence"
                self._onset = 0
                self._offset = 0
                audio = bytes(self._speech_buf)
                self._speech_buf.clear()
                ms = len(audio) // 2 * 1000 // AUDIO_RATE
                events.append(SpeechStopped(
                    item_id=self._item_id, audio=audio, duration_ms=ms))

        return events

    @property
    def speech_buf(self) -> bytearray:
        """Current in-progress speech segment (for manual commit)."""
        return self._speech_buf

    def take_speech(self) -> bytes:
        """Return and clear the current speech segment (manual commit path)."""
        audio = bytes(self._speech_buf)
        self._speech_buf.clear()
        return audio

    def clear(self) -> None:
        """Reset all buffers and state (input_audio_buffer.clear)."""
        self._buf.clear()
        self._speech_buf.clear()
        self._state = "silence"
        self._onset = 0
        self._offset = 0
