"""
Session orchestrator.

Maintain conversation history, dispatch inbound protocol events, coordinate the
VADBuffer and InferencePipeline, and manage the cancellation task. One
WebSocket connection = one Session.
"""

from __future__ import annotations

import asyncio
import base64
import json

from . import protocol
from .pipeline import AudioDelta, InferencePipeline, PipelineDone, TextDelta
from .vad import VAD_THRESHOLD, VADBuffer, SpeechStarted, SpeechStopped


class Session:
    """One WebSocket connection = one session with independent state."""

    def __init__(self, ws, cfg: dict) -> None:
        self.ws = ws
        self.id = f"sess_{protocol.iid()}"
        self.instructions: str = cfg.get("llm_system_prompt", "You are a helpful assistant.")

        # Conversation history — system message always at index 0
        self.history: list[dict] = [{"role": "system", "content": self.instructions}]

        self._vad = VADBuffer()
        self._pipeline = InferencePipeline(cfg)
        self._cfg = cfg

        # In-progress response
        self._response_task: asyncio.Task | None = None
        self._resp_cancel: asyncio.Event = asyncio.Event()  # set when current response is interrupted

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def _send(self, event: dict) -> None:
        await self.ws.send(json.dumps(event))

    def _session_obj(self) -> dict:
        return {
            "id": self.id,
            "object": "realtime.session",
            "model": self._cfg.get("llm_model", ""),
            "modalities": ["text", "audio"],
            "instructions": self.instructions,
            "voice": self._cfg.get("tts_voice", ""),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": self._cfg.get("stt_model", "")},
            "turn_detection": {"type": "server_vad", "threshold": VAD_THRESHOLD},
        }

    # ------------------------------------------------------------------
    # Inbound event dispatch
    # ------------------------------------------------------------------

    async def on_event(self, raw: str) -> None:
        try:
            evt = json.loads(raw)
        except Exception:
            return

        t = evt.get("type", "")

        if t == "session.update":
            body = evt.get("session", {})
            if "instructions" in body:
                self.instructions = body["instructions"]
                self.history[0]["content"] = self.instructions
            await self._send(protocol.session_updated(self._session_obj()))

        elif t == "input_audio_buffer.append":
            raw_b64 = evt.get("audio", "")
            if raw_b64:
                await self._audio_append(base64.b64decode(raw_b64))

        elif t == "input_audio_buffer.commit":
            # Manual commit (client using turn_detection: none)
            audio = self._vad.take_speech()
            if audio:
                await self._commit(protocol.iid(), audio)

        elif t == "input_audio_buffer.clear":
            self._vad.clear()
            await self._send(protocol.audio_buffer_cleared())

        elif t == "conversation.item.create":
            await self._handle_text_item(evt.get("item", {}))

        elif t == "response.create":
            await self._cancel_response()
            self._response_task = asyncio.create_task(
                self._run_response(user_item_id=None, audio=None)
            )

        elif t == "response.cancel":
            await self._cancel_response()

    # ------------------------------------------------------------------
    # VAD integration
    # ------------------------------------------------------------------

    async def _audio_append(self, pcm: bytes) -> None:
        """Feed PCM to VADBuffer; translate emitted events to WebSocket sends."""
        for vad_event in self._vad.push(pcm):
            if isinstance(vad_event, SpeechStarted):
                await self._send(protocol.speech_started(vad_event.item_id))
            elif isinstance(vad_event, SpeechStopped):
                await self._send(protocol.speech_stopped(
                    vad_event.item_id, vad_event.duration_ms))
                await self._commit(vad_event.item_id, vad_event.audio)

    # ------------------------------------------------------------------
    # Commit → trigger response
    # ------------------------------------------------------------------

    async def _commit(self, item_id: str, audio: bytes) -> None:
        await self._send(protocol.audio_buffer_committed(item_id))
        await self._send(protocol.conversation_item_created(
            item_id, "user", [{"type": "input_audio", "audio": ""}]))
        await self._cancel_response()
        self._response_task = asyncio.create_task(
            self._run_response(user_item_id=item_id, audio=audio)
        )

    async def _handle_text_item(self, item: dict) -> None:
        if item.get("role") != "user":
            return
        text = "".join(
            c.get("text", "") for c in item.get("content", [])
            if c.get("type") == "input_text"
        )
        if not text:
            return
        item_id = item.get("id") or protocol.iid()
        self.history.append({"role": "user", "content": text})
        await self._send(protocol.conversation_item_created(
            item_id, "user", item.get("content", [])))
        await self._cancel_response()
        self._response_task = asyncio.create_task(
            self._run_response(user_item_id=item_id, audio=None)
        )

    async def _cancel_response(self) -> None:
        if self._response_task and not self._response_task.done():
            self._resp_cancel.set()          # fast signal checked inside TTS loop
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass
        self._resp_cancel = asyncio.Event()  # reset for next response

    # ------------------------------------------------------------------
    # STT → LLM → TTS orchestration
    # ------------------------------------------------------------------

    async def _run_response(self, user_item_id: str | None, audio: bytes | None) -> None:
        resp_id = protocol.rid()
        out_item = protocol.iid()
        cancel = self._resp_cancel

        try:
            # 1. Transcribe audio if this turn came from the mic
            if audio is not None:
                transcript = await self._pipeline.transcribe(audio)
                if not transcript:
                    return
                self.history.append({"role": "user", "content": transcript})
                await self._send(protocol.transcription_completed(user_item_id, transcript))

            # 2. Scaffolding events
            await self._send(protocol.response_created(resp_id))
            await self._send(protocol.output_item_added(resp_id, out_item))
            await self._send(protocol.content_part_added(resp_id, out_item))

            # 3. Stream LLM → sentence-chunked TTS → audio deltas
            full_text = ""
            async for event in self._pipeline.run(self.history, cancel):
                if isinstance(event, TextDelta):
                    full_text += event.token
                    await self._send(protocol.audio_transcript_delta(
                        resp_id, out_item, event.token))
                elif isinstance(event, AudioDelta):
                    await self._send(protocol.audio_delta(resp_id, out_item, event.pcm_b64))
                elif isinstance(event, PipelineDone):
                    full_text = event.full_text   # authoritative

            # 4. Save assistant turn in history
            self.history.append({"role": "assistant", "content": full_text})

            # 5. Completion events
            await self._send(protocol.output_audio_done(resp_id, out_item))
            await self._send(protocol.audio_transcript_done(resp_id, out_item, full_text))
            await self._send(protocol.content_part_done(resp_id, out_item, full_text))
            await self._send(protocol.output_item_done(resp_id, out_item, full_text))
            await self._send(protocol.response_done(resp_id, out_item, full_text))

        except asyncio.CancelledError:
            # Tell the client to discard its audio play-buffer immediately
            await self._send(protocol.output_audio_buffer_cleared())
            await self._send(protocol.response_cancelled(resp_id))
        except Exception as exc:
            await self._send(protocol.error_event(str(exc)))
