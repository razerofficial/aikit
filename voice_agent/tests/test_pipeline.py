"""
Unit tests for the inference pipeline.

The three OpenAI clients are constructed in __init__ but never actually reached
here: we stub `_llm_stream` and `_tts_sentence` so the sentence-chunking,
tone-tag stripping, and cancel bail-out logic can be exercised in isolation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from voice_agent.pipeline import (
    AudioDelta,
    InferencePipeline,
    PipelineDone,
    TextDelta,
)

_CFG = {
    "api_key": "test",
    "stt_base_url": "http://x/v1",
    "llm_base_url": "http://x/v1",
    "tts_base_url": "http://x/v1",
    "stt_model": "stt",
    "llm_model": "llm",
    "tts_model": "tts",
    "tts_voice": "v",
}


def _chunk(text: str):
    """Mimic an OpenAI streaming chat chunk carrying a content delta."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


async def _fake_stream(tokens):
    async def _gen():
        for tok in tokens:
            yield _chunk(tok)
    return _gen()


def _make_pipeline(tokens, tts_chunks_per_sentence=1):
    """Build a pipeline whose LLM emits `tokens` and TTS emits fixed audio."""
    pipe = InferencePipeline(_CFG)

    async def fake_llm_stream(history):
        return await _fake_stream(tokens)

    async def fake_tts_sentence(text, instruct, cancel):
        for i in range(tts_chunks_per_sentence):
            if cancel.is_set():
                return
            yield f"audio::{instruct}::{text}::{i}"

    pipe._llm_stream = fake_llm_stream
    pipe._tts_sentence = fake_tts_sentence
    return pipe


async def _collect(pipe, cancel=None):
    cancel = cancel or asyncio.Event()
    events = []
    async for ev in pipe.run([{"role": "system", "content": "x"}], cancel):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_tone_tag_stripped_from_text():
    pipe = _make_pipeline(["[cheerful] ", "Hello there. "])
    events = await _collect(pipe)
    text = "".join(e.token for e in events if isinstance(e, TextDelta))
    assert "[cheerful]" not in text
    assert "Hello there." in text


@pytest.mark.asyncio
async def test_tone_tag_passed_to_tts_instruct():
    pipe = _make_pipeline(["[excited] ", "Hi. "])
    events = await _collect(pipe)
    audio = [e.pcm_b64 for e in events if isinstance(e, AudioDelta)]
    assert audio and all("::excited::" in a for a in audio)


@pytest.mark.asyncio
async def test_no_tone_tag_empty_instruct():
    pipe = _make_pipeline(["Plain text without a tag. "])
    events = await _collect(pipe)
    audio = [e.pcm_b64 for e in events if isinstance(e, AudioDelta)]
    assert audio and all("::::" in a for a in audio)


@pytest.mark.asyncio
async def test_sentence_chunking_splits_on_punctuation():
    pipe = _make_pipeline(["[calm] ", "One. ", "Two! ", "Three? ", "Four "])
    events = await _collect(pipe)
    sentences = [
        e.pcm_b64.split("::")[2]
        for e in events if isinstance(e, AudioDelta)
    ]
    assert "One." in sentences
    assert "Two!" in sentences
    assert "Three?" in sentences
    # Trailing "Four" (no terminator) flushed at the end
    assert "Four" in sentences


@pytest.mark.asyncio
async def test_pipeline_done_has_full_text():
    pipe = _make_pipeline(["[warm] ", "Alpha. ", "Beta."])
    events = await _collect(pipe)
    done = [e for e in events if isinstance(e, PipelineDone)]
    assert len(done) == 1
    assert done[0].full_text == "Alpha. Beta."


@pytest.mark.asyncio
async def test_cancel_stops_tts_early():
    cancel = asyncio.Event()
    cancel.set()   # cancelled before any TTS runs
    pipe = _make_pipeline(["[calm] ", "One. ", "Two. "], tts_chunks_per_sentence=3)
    events = await _collect(pipe, cancel=cancel)
    audio = [e for e in events if isinstance(e, AudioDelta)]
    assert audio == []   # TTS bailed out immediately
    # Still emits a terminal PipelineDone with the accumulated text
    assert any(isinstance(e, PipelineDone) for e in events)
