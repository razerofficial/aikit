"""
SDK compatibility test for realtime_server.py.

Uses the official OpenAI Python SDK's realtime client pointed at the local
server instead of OpenAI's servers. This is the direct equivalent of:

    client = AsyncOpenAI()                          # normally points to OpenAI
    async with client.beta.realtime.connect(...):   # uses wss://api.openai.com

Here we just change base_url to point at our server. Everything else is
identical to how you would use the real OpenAI Realtime API.

Usage:
    # Server must be running first:
    #   python websocket/realtime_server.py
    python websocket/openai_client.py
"""

from __future__ import annotations

import asyncio
import base64
import threading
import queue

import numpy as np
import pyaudio
from openai import AsyncOpenAI

SERVER_URL = "http://127.0.0.1:8081"
MODEL     = "Qwen/Qwen3.5-4B"   # value is sent in session.update; server uses config

AUDIO_RATE   = 24_000   # rate the server speaks (pcm16 at 24 kHz)
CHUNK_FRAMES = AUDIO_RATE * 100 // 1000   # 100 ms worth of server-rate frames

# Candidate rates to probe when 24 kHz is unsupported by the hardware.
_FALLBACK_RATES = [48_000, 44_100, 16_000, 8_000]


def _find_device_rate(pa: pyaudio.PyAudio, *, is_input: bool) -> int:
    """Return the best sample rate the default device accepts, starting with 24 kHz."""
    for rate in [AUDIO_RATE] + _FALLBACK_RATES:
        try:
            kw = dict(format=pyaudio.paInt16, channels=1, rate=rate, frames_per_buffer=256)
            kw["input" if is_input else "output"] = True
            s = pa.open(**kw)
            s.stop_stream()
            s.close()
            return rate
        except OSError:
            continue
    raise RuntimeError("No supported audio sample rate found on this device")


def _resample(data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resample of raw int16 PCM bytes."""
    if from_rate == to_rate:
        return data
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    n_out = max(1, int(round(len(samples) * to_rate / from_rate)))
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, n_out),
        np.arange(len(samples)),
        samples,
    ).astype(np.int16)
    return resampled.tobytes()


def _drain_queue(q: queue.Queue) -> None:
    """Discard all items currently sitting in a Queue (non-blocking)."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _mic_thread(out_q: queue.Queue, stop: threading.Event, pa: pyaudio.PyAudio) -> None:
    dev_rate = _find_device_rate(pa, is_input=True)
    dev_chunk = dev_rate * 100 // 1000   # 100 ms at device rate
    if dev_rate != AUDIO_RATE:
        print(f"[mic] device rate {dev_rate} Hz → resampling to {AUDIO_RATE} Hz")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=dev_rate,
                     input=True, frames_per_buffer=dev_chunk)
    try:
        while not stop.is_set():
            raw = stream.read(dev_chunk, exception_on_overflow=False)
            out_q.put(_resample(raw, dev_rate, AUDIO_RATE))
    finally:
        stream.stop_stream()
        stream.close()


def _play_thread(in_q: queue.Queue, stop: threading.Event, pa: pyaudio.PyAudio) -> None:
    dev_rate = _find_device_rate(pa, is_input=False)
    dev_chunk = dev_rate * 100 // 1000   # 100 ms at device rate
    silence = b'\x00' * (dev_chunk * 2)  # 2 bytes per int16 sample
    if dev_rate != AUDIO_RATE:
        print(f"[play] device rate {dev_rate} Hz → resampling from {AUDIO_RATE} Hz")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=dev_rate,
                     output=True, frames_per_buffer=dev_chunk)
    try:
        while not stop.is_set():
            try:
                chunk = in_q.get_nowait()
            except queue.Empty:
                chunk = silence
            else:
                chunk = _resample(chunk, AUDIO_RATE, dev_rate)
            stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()


async def main() -> None:
    # ── This is identical to how you'd connect to OpenAI ──────────────────────
    client = AsyncOpenAI(
        api_key="vllm",
        base_url=SERVER_URL,
        # The SDK defaults to wss:// (TLS). Our local server is plain ws://,
        # so we set websocket_base_url explicitly with the ws:// scheme.
        websocket_base_url=SERVER_URL.replace("http://", "ws://"),
    )

    pa = pyaudio.PyAudio()
    stop = threading.Event()
    mic_q: queue.Queue = queue.Queue()
    play_q: queue.Queue = queue.Queue()

    threading.Thread(target=_mic_thread,  args=(mic_q,  stop, pa), daemon=True).start()
    threading.Thread(target=_play_thread, args=(play_q, stop, pa), daemon=True).start()

    print(f"Connecting to {SERVER_URL} via OpenAI SDK realtime client…")

    async with client.beta.realtime.connect(model=MODEL) as conn:
        print("Connected. Speak into your mic. Ctrl+C to stop.\n")

        # Configure the session — same call you'd make against OpenAI
        await conn.session.update(session={
            "instructions": "You are a helpful voice assistant. Keep answers brief.",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        })

        async def send_audio() -> None:
            while True:
                try:
                    chunks = []
                    while True:
                        chunks.append(mic_q.get_nowait())
                except queue.Empty:
                    pass
                for chunk in chunks:
                    await conn.input_audio_buffer.append(
                        audio=base64.b64encode(chunk).decode()
                    )
                await asyncio.sleep(0.05)

        async def receive_events() -> None:
            async for event in conn:
                t = event.type

                if t == "input_audio_buffer.speech_started":
                    print("[speech detected]")
                    # Discard queued TTS audio so the old response stops playing immediately
                    _drain_queue(play_q)

                elif t == "output_audio_buffer.cleared":
                    # Server explicitly signals: drop everything in the play buffer
                    _drain_queue(play_q)

                elif t == "input_audio_buffer.speech_stopped":
                    print("[speech ended — generating…]")

                elif t == "conversation.item.input_audio_transcription.completed":
                    print(f"[You]: {event.transcript}")

                elif t == "response.output_audio_transcript.delta":
                    print(event.delta, end="", flush=True)

                elif t == "response.output_audio.delta":
                    play_q.put(base64.b64decode(event.delta))

                elif t == "response.output_audio.done":
                    print()

                elif t == "error":
                    print(f"\n[error]: {event}")

        send_task   = asyncio.create_task(send_audio())
        recv_task   = asyncio.create_task(receive_events())
        try:
            await asyncio.gather(send_task, recv_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            send_task.cancel()
            recv_task.cancel()

    stop.set()
    pa.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
