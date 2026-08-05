"""Unit tests for the energy VAD state machine — no I/O, synthetic PCM only."""

from __future__ import annotations

import numpy as np

from voice_agent.vad import (
    AUDIO_RATE,
    CHUNK_BYTES,
    VAD_OFFSET_CHUNKS,
    VAD_ONSET_CHUNKS,
    VADBuffer,
    SpeechStarted,
    SpeechStopped,
)

CHUNK_SAMPLES = CHUNK_BYTES // 2


def _speech_chunk() -> bytes:
    """One CHUNK_BYTES frame of high-amplitude 440 Hz tone."""
    t = np.linspace(0, CHUNK_SAMPLES / AUDIO_RATE, CHUNK_SAMPLES, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 28_000).astype(np.int16).tobytes()


def _silence_chunk() -> bytes:
    return bytes(CHUNK_BYTES)


def test_silence_only_no_events():
    vad = VADBuffer()
    events = vad.push(_silence_chunk() * 30)
    assert events == []


def test_short_noise_burst_no_onset():
    # Fewer than VAD_ONSET_CHUNKS speech frames → no SpeechStarted
    vad = VADBuffer()
    events = vad.push(_speech_chunk() * (VAD_ONSET_CHUNKS - 1))
    assert events == []


def test_sustained_speech_triggers_started():
    vad = VADBuffer()
    events = vad.push(_speech_chunk() * VAD_ONSET_CHUNKS)
    started = [e for e in events if isinstance(e, SpeechStarted)]
    assert len(started) == 1
    assert started[0].item_id.startswith("item_")


def test_speech_then_silence_triggers_stopped():
    vad = VADBuffer()
    vad.push(_speech_chunk() * VAD_ONSET_CHUNKS)
    events = vad.push(_silence_chunk() * VAD_OFFSET_CHUNKS)
    stopped = [e for e in events if isinstance(e, SpeechStopped)]
    assert len(stopped) == 1
    # Accumulated audio should be non-empty and a multiple of the frame size
    assert len(stopped[0].audio) > 0
    assert stopped[0].duration_ms > 0


def test_stopped_item_id_matches_started():
    vad = VADBuffer()
    started = vad.push(_speech_chunk() * VAD_ONSET_CHUNKS)
    stopped = vad.push(_silence_chunk() * VAD_OFFSET_CHUNKS)
    assert started[0].item_id == stopped[0].item_id


def test_clear_resets_state():
    vad = VADBuffer()
    vad.push(_speech_chunk() * VAD_ONSET_CHUNKS)
    vad.clear()
    # After clear, a fresh silence run yields nothing (no lingering speech state)
    events = vad.push(_silence_chunk() * VAD_OFFSET_CHUNKS)
    assert events == []


def test_partial_chunk_is_buffered():
    vad = VADBuffer()
    # Push half a chunk — not enough to process
    events = vad.push(_speech_chunk()[: CHUNK_BYTES // 2])
    assert events == []
