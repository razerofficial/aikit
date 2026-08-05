"""
Interruption detection test — no microphone required.

Tests that when the user speaks mid-response, TTS audio stops promptly:
  - The server cancels the response (server-side)
  - The client discards any queued audio (client-side, verified via event timing)

Requires a running realtime_server.py:
    cd /home/cestest/Projects/voice_agent
    source .venv/bin/activate
    python websocket/realtime_server.py

Then in another terminal:
    python tests/test_interruption.py

Pass criteria
-------------
  audio_after_speech  ≤ MAX_AUDIO_AFTER_SPEECH  (server stops sending quickly)
  cancel_latency_ms   ≤ MAX_CANCEL_LATENCY_MS   (response.done/cancelled arrives quickly)
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field

import numpy as np
import websockets

# ── config ────────────────────────────────────────────────────────────────────
SERVER_URL  = "ws://127.0.0.1:8081/v1/realtime"
AUDIO_RATE  = 24_000
CHUNK_BYTES = AUDIO_RATE * 2 * 100 // 1000   # 100 ms of PCM16

# Pass thresholds
MAX_AUDIO_AFTER_SPEECH   = 5    # audio deltas allowed after speech_started
MAX_CANCEL_LATENCY_MS    = 800  # ms from speech_started to response.done/cancelled


# ── helpers ───────────────────────────────────────────────────────────────────

def _speech_pcm(duration_s: float = 0.5) -> bytes:
    """440 Hz tone at high amplitude — reliably triggers the energy VAD."""
    n = int(AUDIO_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 28_000).astype(np.int16)
    return samples.tobytes()


def _pad_chunk(data: bytes) -> bytes:
    """Zero-pad to exactly CHUNK_BYTES if shorter."""
    if len(data) < CHUNK_BYTES:
        data = data + bytes(CHUNK_BYTES - len(data))
    return data


async def _send(ws, obj: dict) -> None:
    await ws.send(json.dumps(obj))


# ── event collector ───────────────────────────────────────────────────────────

@dataclass
class RunStats:
    label: str
    audio_deltas_before: int = 0
    audio_deltas_after:  int = 0
    speech_started_at:   float | None = None
    cancel_latency_ms:   float | None = None
    new_response_done:   bool = False
    passed:              bool = False
    log: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        ts = time.monotonic()
        self.log.append(f"  {ts:.3f}  {msg}")
        print(msg)


async def run_interruption_test(label: str) -> RunStats:
    stats = RunStats(label=label)

    async with websockets.connect(SERVER_URL) as ws:
        done_event = asyncio.Event()

        async def receive_loop() -> None:
            async for raw in ws:
                evt = json.loads(raw)
                t   = evt.get("type", "")

                if t == "response.output_audio.delta":
                    if stats.speech_started_at is None:
                        stats.audio_deltas_before += 1
                    else:
                        stats.audio_deltas_after += 1

                elif t == "input_audio_buffer.speech_started":
                    stats.speech_started_at = time.monotonic()
                    stats.note(f"[speech_started] after {stats.audio_deltas_before} audio deltas")

                elif t == "output_audio_buffer.cleared":
                    stats.note("[output_audio_buffer.cleared] server told client to drop buffer")

                elif t == "response.done":
                    status = evt.get("response", {}).get("status", "")
                    if status == "cancelled" and stats.speech_started_at is not None:
                        elapsed = (time.monotonic() - stats.speech_started_at) * 1000
                        stats.cancel_latency_ms = elapsed
                        stats.note(f"[response.done/cancelled] {elapsed:.0f} ms after speech_started, "
                                   f"{stats.audio_deltas_after} audio deltas leaked")
                    elif status == "completed" and stats.cancel_latency_ms is not None:
                        stats.new_response_done = True
                        stats.note("[response.done/completed] new response finished — interruption cycle complete")
                        done_event.set()

                elif t == "error":
                    stats.note(f"[error] {evt}")
                    done_event.set()

        recv_task = asyncio.create_task(receive_loop())

        # 1. Configure session — ask for a deliberately long answer
        await _send(ws, {
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a verbose assistant. "
                    "Always answer with very long, detailed paragraphs. "
                    "Never use bullet points. "
                    "Speak in complete flowing sentences."
                ),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
            },
        })

        # 2. Inject a text turn (bypasses STT — no real audio transcription needed)
        await _send(ws, {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "Please give me an extremely detailed and lengthy explanation "
                        "of the entire history of computing from the abacus to modern GPUs. "
                        "Be very verbose and cover every era in great depth."
                    ),
                }],
            },
        })
        await _send(ws, {"type": "response.create"})

        # 3. Wait for TTS to start streaming (need at least 3 audio deltas)
        stats.note("Waiting for TTS to begin streaming…")
        deadline = time.monotonic() + 30.0
        while stats.audio_deltas_before < 3:
            if time.monotonic() > deadline:
                stats.note("TIMEOUT waiting for first audio deltas")
                recv_task.cancel()
                return stats
            await asyncio.sleep(0.05)

        stats.note(f"TTS streaming confirmed ({stats.audio_deltas_before} deltas). Injecting speech…")

        # 4. Inject high-energy synthetic speech to trigger server VAD
        #    VAD_ONSET_CHUNKS = 3  →  need ≥ 3 × 100 ms = 300 ms above threshold
        #    We send 500 ms to be safe.
        speech = _speech_pcm(0.5)
        for i in range(0, len(speech), CHUNK_BYTES):
            chunk = _pad_chunk(speech[i : i + CHUNK_BYTES])
            await _send(ws, {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            })
            await asyncio.sleep(0.1)   # match real-time pacing so VAD sees separate chunks

        # 5. Also follow with silence so VAD eventually fires speech_stopped → new response
        silence = bytes(CHUNK_BYTES)
        for _ in range(22):   # VAD_OFFSET_CHUNKS = 20  →  send 22 × 100 ms = 2.2 s silence
            await _send(ws, {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(silence).decode(),
            })
            await asyncio.sleep(0.1)

        # 6. Wait up to 15 s for the full interruption + new response cycle
        try:
            await asyncio.wait_for(done_event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            stats.note("TIMEOUT waiting for interruption cycle to complete")

        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass

    # 7. Evaluate
    leaked = stats.audio_deltas_after
    lat    = stats.cancel_latency_ms

    stats.passed = (
        leaked is not None
        and leaked <= MAX_AUDIO_AFTER_SPEECH
        and lat is not None
        and lat <= MAX_CANCEL_LATENCY_MS
        and stats.new_response_done
    )
    return stats


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 60)
    print("  Interruption detection test")
    print(f"  Server : {SERVER_URL}")
    print(f"  Thresholds: ≤{MAX_AUDIO_AFTER_SPEECH} audio leaks, "
          f"≤{MAX_CANCEL_LATENCY_MS} ms cancel latency")
    print("=" * 60)

    results: list[RunStats] = []

    for i in range(1, 3):   # run twice to check consistency
        print(f"\n─── Run {i}/2 ───────────────────────────────────────────")
        stats = await run_interruption_test(label=f"run-{i}")
        results.append(stats)
        await asyncio.sleep(1.0)   # brief pause between runs

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in results:
        lat_str = f"{r.cancel_latency_ms:.0f} ms" if r.cancel_latency_ms else "N/A"
        status  = "PASS" if r.passed else "FAIL"
        all_pass = all_pass and r.passed
        print(f"  [{status}] {r.label}:")
        print(f"         audio deltas after speech : {r.audio_deltas_after}  "
              f"(limit {MAX_AUDIO_AFTER_SPEECH})")
        print(f"         cancel latency            : {lat_str}  "
              f"(limit {MAX_CANCEL_LATENCY_MS} ms)")
        print(f"         new response completed    : {r.new_response_done}")

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — check output above")
        print()
        print("  Hint: if audio_deltas_after is high, the server is not cancelling")
        print("  fast enough. If cancel_latency_ms is high, VAD onset is slow.")
        print("  If new_response_done is False, the silence tail was too short or")
        print("  the post-interruption response timed out.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
