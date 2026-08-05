"""
OpenAI Realtime-compatible WebSocket server backed by the cascade pipeline.

Exposes ws://<host>:<port>/v1/realtime speaking the OpenAI Realtime WebSocket
protocol. Internally drives three services:

    Energy VAD → Whisper STT → Qwen LLM (streaming) → Qwen TTS (streaming)

No Pipecat pipeline — uses the openai Python client directly for each service.
Multi-client: every connection gets its own session with its own conversation
history and VAD state.

Thin entry point: config loading, connection dispatch, and the server loop.
The Session orchestrator lives in session.py; VAD, pipeline, and protocol
concerns live in their own modules.

Config: voice_agent_config.json (same directory).

Usage:
    python -m voice_agent.realtime_server [--host 0.0.0.0] [--port 8081]

Client connection:
    ws://127.0.0.1:8081/v1/realtime   (path is accepted but not enforced)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets

from . import protocol
from .session import Session

CONFIG_PATH = Path(__file__).parent / "voice_agent_config.json"


async def handle_connection(ws, cfg: dict) -> None:
    session = Session(ws, cfg)
    await session._send(protocol.session_created(session._session_obj()))
    try:
        async for message in ws:
            if isinstance(message, str):
                await session.on_event(message)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as exc:
        print(f"[session {session.id}] error: {exc}")
    finally:
        await session._cancel_response()


async def main() -> None:
    host = "0.0.0.0"
    port = 8081

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
        elif arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])

    cfg = json.loads(CONFIG_PATH.read_text())

    print(f"Realtime server  →  ws://{host}:{port}/v1/realtime")
    print(f"  STT : {cfg['stt_model']}  @ {cfg['stt_base_url']}")
    print(f"  LLM : {cfg['llm_model']}  @ {cfg['llm_base_url']}")
    print(f"  TTS : {cfg['tts_model']}  @ {cfg['tts_base_url']}")
    print("Ready — Ctrl+C to stop.\n")

    async with websockets.serve(lambda ws: handle_connection(ws, cfg), host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
