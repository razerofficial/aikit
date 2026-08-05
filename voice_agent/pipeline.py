"""
Inference pipeline: STT → LLM (streaming) → TTS (streaming).

Owns the three AsyncOpenAI clients but holds no conversation state. Exposes the
inference flow as an async generator that yields typed events.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import wave
from dataclasses import dataclass

from openai import AsyncOpenAI

# Sentence boundary for TTS chunking (flush at end of sentence)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# Leading tone marker emitted by the LLM at the start of each response
_LEADING_TONE = re.compile(r"^\s*\[([^\]]{1,40})\]\s*")

AUDIO_RATE = 24_000


def _pcm_to_wav(pcm: bytes, sample_rate: int = AUDIO_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# --- pipeline event types ---------------------------------------------------

@dataclass
class TextDelta:
    token: str            # LLM text token (tone tag already stripped)


@dataclass
class AudioDelta:
    pcm_b64: str          # base64-encoded PCM chunk ready to forward


@dataclass
class PipelineDone:
    full_text: str        # complete assistant turn text (tone tag stripped)


PipelineEvent = TextDelta | AudioDelta | PipelineDone


class InferencePipeline:
    """
    Stateless inference: transcribe audio, stream LLM tokens, stream TTS audio.
    Owns the three OpenAI clients but holds no conversation state.
    """

    def __init__(self, cfg: dict) -> None:
        self._stt = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["stt_base_url"])
        self._llm = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["llm_base_url"])
        self._tts = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["tts_base_url"])
        self._cfg = cfg

    async def transcribe(self, pcm: bytes) -> str:
        """Convert raw PCM bytes → transcript string via the STT endpoint."""
        wav_bytes = _pcm_to_wav(pcm)
        result = await self._stt.audio.transcriptions.create(
            model=self._cfg["stt_model"],
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
        )
        return (result.text or "").strip()

    async def run(self, history: list[dict], cancel: asyncio.Event):
        """
        Async generator. Streams: TextDelta* → AudioDelta* → PipelineDone.

        Caller iterates with `async for event in pipeline.run(history, cancel)`.
        Yields TextDelta and AudioDelta interleaved as they become available.
        Yields exactly one PipelineDone as the final event.
        Stops early (yields PipelineDone with partial text) if cancel is set.
        """
        # Kick off the LLM request immediately so it can warm up while the caller
        # sends scaffolding events.
        llm_task = asyncio.create_task(self._llm_stream(history))
        stream = await llm_task

        sentence_buf = ""
        full_text = ""
        instruct = self._cfg.get("tts_instruct", "")

        # Buffer the very start of the response to extract the single leading
        # [tone] tag. Stop buffering once the tag is found or >60 chars have
        # accumulated without one.
        lead_buf = ""
        tone_parsed = False

        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token:
                continue

            if not tone_parsed:
                lead_buf += token
                m = _LEADING_TONE.match(lead_buf)
                if m:
                    instruct = m.group(1).strip()
                    token = lead_buf[m.end():]   # remainder after the tag
                    tone_parsed = True
                elif len(lead_buf) > 60 or ("[" not in lead_buf and lead_buf.strip()):
                    # No tag found - flush accumulated buffer as normal text
                    token = lead_buf
                    tone_parsed = True
                else:
                    continue   # still accumulating the tag

            if not token:
                continue

            sentence_buf += token
            full_text += token

            # Emit text transcript delta (tone tag already stripped)
            yield TextDelta(token=token)

            # Flush complete sentences to TTS immediately for low latency
            parts = _SENTENCE_END.split(sentence_buf)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    if sentence.strip():
                        async for audio in self._tts_sentence(sentence.strip(), instruct, cancel):
                            yield AudioDelta(pcm_b64=audio)
                sentence_buf = parts[-1]

        # Flush any remaining text
        if sentence_buf.strip():
            async for audio in self._tts_sentence(sentence_buf.strip(), instruct, cancel):
                yield AudioDelta(pcm_b64=audio)

        yield PipelineDone(full_text=full_text)

    async def _llm_stream(self, history: list[dict]):
        """Open streaming chat completion. Called as a Task for parallelism."""
        return await self._llm.chat.completions.create(
            model=self._cfg["llm_model"],
            messages=history,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _tts_sentence(self, text: str, instruct: str, cancel: asyncio.Event):
        """
        Async generator. Yields base64-encoded PCM chunks for one sentence.
        Returns immediately if cancel is set.
        """
        extra_body = {"instruct": instruct} if instruct else None
        async with self._tts.audio.speech.with_streaming_response.create(
            model=self._cfg["tts_model"],
            voice=self._cfg["tts_voice"],
            input=text,
            response_format="pcm",
            extra_body=extra_body,
        ) as resp:
            async for chunk in resp.iter_bytes(4096):
                if cancel.is_set():          # fast bail-out; task cancel fires at next await
                    return
                if chunk:
                    yield base64.b64encode(chunk).decode()
