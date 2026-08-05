"""
Voice agent bridge between the Gradio UI and the realtime server.

Adapts voice_agent/examples/openai_client.py for use inside the Gradio process.
The CLI client reads a microphone and writes to speakers with pyaudio; here the
browser does both, so the two hardware threads are replaced by queues that the
Gradio event handlers fill and drain:

    Gradio mic .stream()  → mic_q  → _send_audio()     ─┐
                                                        ├─ WS → realtime_server
    Gradio audio out      ← play_q ← _receive_events() ─┘

Everything else — the session.update call, the event table, and the barge-in
drain — is the same protocol handling as the CLI client. The realtime server is
untouched; this is only a client.

One VoiceBridge per browser session, held in a gr.State.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

# PCM16 mono at 24 kHz, matching voice_agent/vad.py and the OpenAI Realtime default.
AUDIO_RATE = 24_000
# Gradio streams audio as HLS: every value the playback generator yields becomes
# one independent AAC segment. AAC codes in 1024-sample frames, so each segment
# carries about a frame of encoder priming silence (~43 ms at 24 kHz) plus a
# padded trailing frame. Forwarding the server's 4096-byte (~85 ms) TTS chunks
# one-for-one inserts more silence than audio, so they are coalesced into
# frame-aligned segments instead.
AAC_FRAME_SAMPLES = 1024
FRAME_BYTES = AAC_FRAME_SAMPLES * 2  # PCM16 mono

# Segment sizes in AAC frames, applied in order with the last value repeating.
# The first segment is deliberately tiny so speech starts as soon as the first
# TTS chunk lands (~85 ms) instead of waiting for a big buffer to fill; later
# segments grow to amortise the per-segment padding. Measured on a 6 s sample,
# this ramp is just as smooth as starting large (1.06x either way) but gets first
# audio out 170 ms sooner. Capped at 24 frames (~1.02 s) to stay near Gradio's
# hls.js maxBufferLength of 1 second.
SEGMENT_RAMP_FRAMES = (2, 12, 24)

# Control markers passed through play_q alongside PCM bytes.
TURN_FLUSH = object()  # reply finished — release the partial buffer
TURN_RESET = object()  # server dropped its audio — discard the partial buffer

# The realtime server's own config, so the UI never has to restate the model.
# The server reads the same file; anything it can't find there it defaults itself.
CONFIG_PATH = Path("/var/aikit/voice_agent/voice_agent_config.json")
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def load_model_name() -> str:
    """
    Read llm_model from the voice agent config, falling back to the default.

    The value is only echoed back in session.update — the server uses its own
    config regardless — so a stale name here is harmless.
    """
    try:
        return json.loads(CONFIG_PATH.read_text()).get("llm_model") or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def _drain_queue(q: queue.Queue) -> None:
    """Discard all items currently sitting in a Queue (non-blocking)."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def to_pcm16_24k(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a Gradio mic chunk to the raw PCM16 mono 24 kHz bytes the server wants.

    Browsers capture at 44.1 or 48 kHz, but the server's energy VAD is tuned for
    24 kHz (VAD_THRESHOLD in voice_agent/vad.py) and STT expects that rate too,
    so resampling is required rather than cosmetic.
    """
    if audio.ndim > 1:  # mixdown to mono
        audio = audio.mean(axis=1)

    # Gradio hands over int16 for type="numpy", but normalize defensively.
    if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0) * 32767
    audio = audio.astype(np.float32)

    if sample_rate != AUDIO_RATE:
        from scipy.signal import resample_poly

        g = np.gcd(int(sample_rate), AUDIO_RATE)
        audio = resample_poly(audio, AUDIO_RATE // g, int(sample_rate) // g)

    return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


def pcm_to_wav(pcm: bytes, sample_rate: int = AUDIO_RATE) -> bytes:
    """Wrap raw PCM16 in a WAV container for gr.Audio(streaming=True)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class PlaybackSegmenter:
    """
    Accumulate small TTS chunks into AAC-frame-aligned HLS segments.

    See the SEGMENT_RAMP_FRAMES comment: one segment per TTS chunk makes Gradio
    emit mostly encoder padding, so chunks are coalesced here instead. Sizes are
    whole multiples of AAC_FRAME_SAMPLES so a segment never straddles a frame
    boundary, which would push the remainder into another padded frame.

    Used from the playback generator only — not thread-safe, and doesn't need to
    be, since one generator owns one instance.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._index = 0  # how many segments emitted, indexes into the ramp

    def _target_bytes(self) -> int:
        frames = SEGMENT_RAMP_FRAMES[min(self._index, len(SEGMENT_RAMP_FRAMES) - 1)]
        return frames * FRAME_BYTES

    def push(self, pcm: bytes) -> list[bytes]:
        """Add a TTS chunk; return the WAV segments it completed."""
        self._buf += pcm
        out = []
        while len(self._buf) >= self._target_bytes():
            n = self._target_bytes()
            out.append(pcm_to_wav(bytes(self._buf[:n])))
            del self._buf[:n]
            self._index += 1
        return out

    def flush(self) -> list[bytes]:
        """
        Emit the partial tail at the end of a reply.

        Without this the last fraction of a segment would sit in the buffer,
        cutting the final word off every reply.
        """
        if not self._buf:
            return []
        tail = pcm_to_wav(bytes(self._buf))
        self._buf.clear()
        self._index = 0  # next reply restarts the ramp for a fast first segment
        return [tail]

    def reset(self) -> None:
        """Drop buffered audio the server has told us to discard."""
        self._buf.clear()
        self._index = 0


class VoiceBridge:
    """
    A realtime-server client driven by Gradio events instead of pyaudio.

    Owns a background thread running its own asyncio loop, because the Gradio
    server's loop is busy handling requests. Audio crosses the thread boundary
    through mic_q / play_q; the transcript crosses through a plain list guarded
    by a lock and polled by a gr.Timer.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8081",
        model: str | None = None,
    ) -> None:
        self._url = server_url.rstrip("/")
        self._model = model or load_model_name()

        self.mic_q: queue.Queue = queue.Queue()
        self.play_q: queue.Queue = queue.Queue()

        self.stop = threading.Event()
        self.status = "idle"
        self.error: str | None = None

        self._lock = threading.Lock()
        self._transcript: list[dict] = []
        self._assistant_open = False  # is the last entry an in-progress reply?
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background connection thread. Returns immediately."""
        if self._thread is not None:
            return
        self.status = "connecting"
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Signal the background thread to shut down and drop queued audio."""
        self.stop.set()
        self.status = "idle"
        _drain_queue(self.mic_q)
        _drain_queue(self.play_q)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.error = str(exc)
            self.status = "error"
        finally:
            self.stop.set()

    # ------------------------------------------------------------------
    # Transcript (read by the UI thread)
    # ------------------------------------------------------------------

    def transcript(self) -> list[dict]:
        """Snapshot of the conversation as gr.Chatbot message dicts."""
        with self._lock:
            return [dict(m) for m in self._transcript]

    def _add_user(self, text: str) -> None:
        with self._lock:
            self._transcript.append({"role": "user", "content": text})
            self._assistant_open = False

    def _append_assistant(self, delta: str) -> None:
        with self._lock:
            if not self._assistant_open:
                self._transcript.append({"role": "assistant", "content": ""})
                self._assistant_open = True
            self._transcript[-1]["content"] += delta

    def _close_assistant(self, text: str) -> None:
        with self._lock:
            if self._assistant_open:
                self._transcript[-1]["content"] = text  # authoritative
            elif text:
                self._transcript.append({"role": "assistant", "content": text})
            self._assistant_open = False

    # ------------------------------------------------------------------
    # Connection — the OpenAI SDK realtime client, pointed at our server
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        client = AsyncOpenAI(
            api_key="vllm",
            base_url=self._url,
            # The SDK defaults to wss:// (TLS). Our local server is plain ws://,
            # so set websocket_base_url explicitly with the ws:// scheme.
            websocket_base_url=self._url.replace("http://", "ws://"),
        )

        async with client.beta.realtime.connect(model=self._model) as conn:
            # No instructions sent — the server keeps its configured system
            # prompt, which is what openai_client.py would override but we don't.
            await conn.session.update(session={
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
            })

            self.status = "listening"

            send_task = asyncio.create_task(self._send_audio(conn))
            recv_task = asyncio.create_task(self._receive_events(conn))
            try:
                await asyncio.gather(send_task, recv_task)
            except asyncio.CancelledError:
                pass
            finally:
                send_task.cancel()
                recv_task.cancel()

    def _interrupt(self) -> None:
        """
        Discard audio that hasn't been published to the browser yet.

        This cannot stop audio already handed to Gradio: MediaStream.segments is
        append-only and the playlist is EXT-X-PLAYLIST-TYPE:EVENT, so anything
        the player can already see plays to the end. Since TTS runs faster than
        realtime, most of a reply is usually already published — true barge-in
        would need a fresh playlist per reply, which costs about a second of
        added latency on every response, so it is deliberately not done here.
        """
        _drain_queue(self.play_q)
        self.play_q.put(TURN_RESET)

    async def _send_audio(self, conn) -> None:
        """Forward everything queued by the Gradio mic handler to the server."""
        while not self.stop.is_set():
            chunks = []
            try:
                while True:
                    chunks.append(self.mic_q.get_nowait())
            except queue.Empty:
                pass
            for chunk in chunks:
                await conn.input_audio_buffer.append(
                    audio=base64.b64encode(chunk).decode()
                )
            # Short poll: this delay lands in front of the server's VAD, so it
            # postpones the end-of-speech detection that starts the whole reply.
            await asyncio.sleep(0.01)

    async def _receive_events(self, conn) -> None:
        """Same event table as the CLI client; prints become transcript writes."""
        async for event in conn:
            if self.stop.is_set():
                return

            t = event.type

            if t == "input_audio_buffer.speech_started":
                self.status = "listening"
                self._interrupt()

            elif t == "output_audio_buffer.cleared":
                self._interrupt()

            elif t == "input_audio_buffer.speech_stopped":
                self.status = "thinking"

            elif t == "conversation.item.input_audio_transcription.completed":
                self._add_user(event.transcript)

            elif t == "response.output_audio_transcript.delta":
                self._append_assistant(event.delta)

            elif t == "response.output_audio.delta":
                self.status = "speaking"
                self.play_q.put(base64.b64decode(event.delta))

            elif t == "response.output_audio.done":
                # End of this turn's audio — release the buffered tail so the
                # last words aren't held back waiting for a full segment.
                self.play_q.put(TURN_FLUSH)

            elif t == "response.output_audio_transcript.done":
                self._close_assistant(event.transcript)

            elif t == "response.done":
                self.status = "listening"

            elif t == "error":
                self.error = str(getattr(event, "error", event))
                self.status = "error"
