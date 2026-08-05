"""
OpenAI Realtime protocol event builders.

Pure functions: generate IDs and build the event-dict payloads that the OpenAI
Realtime WebSocket protocol requires. No I/O, no state. Every builder returns a
plain dict with "event_id" already populated so protocol compliance is
auditable in one place.
"""

from __future__ import annotations

import uuid


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------

def eid() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


def iid() -> str:
    return f"item_{uuid.uuid4().hex[:8]}"


def rid() -> str:
    return f"resp_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------

def session_created(session_obj: dict) -> dict:
    return {"event_id": eid(), "type": "session.created", "session": session_obj}


def session_updated(session_obj: dict) -> dict:
    return {"event_id": eid(), "type": "session.updated", "session": session_obj}


# ---------------------------------------------------------------------------
# Input audio buffer / VAD events
# ---------------------------------------------------------------------------

def speech_started(item_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": 0,
        "item_id": item_id,
    }


def speech_stopped(item_id: str, audio_end_ms: int) -> dict:
    return {
        "event_id": eid(),
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": audio_end_ms,
        "item_id": item_id,
    }


def audio_buffer_committed(item_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "input_audio_buffer.committed",
        "previous_item_id": None,
        "item_id": item_id,
    }


def audio_buffer_cleared() -> dict:
    return {"event_id": eid(), "type": "input_audio_buffer.cleared"}


def conversation_item_created(item_id: str, role: str, content: list) -> dict:
    return {
        "event_id": eid(),
        "type": "conversation.item.created",
        "item": {
            "id": item_id,
            "type": "message",
            "role": role,
            "content": content,
        },
    }


def transcription_completed(item_id: str, transcript: str) -> dict:
    return {
        "event_id": eid(),
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Response lifecycle events
# ---------------------------------------------------------------------------

def response_created(resp_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.created",
        "response": {"id": resp_id, "status": "in_progress", "output": []},
    }


def output_item_added(resp_id: str, out_item_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.output_item.added",
        "response_id": resp_id,
        "output_index": 0,
        "item": {"id": out_item_id, "type": "message", "role": "assistant", "content": []},
    }


def content_part_added(resp_id: str, out_item_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.content_part.added",
        "response_id": resp_id,
        "item_id": out_item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "audio", "transcript": "", "audio": ""},
    }


def audio_transcript_delta(resp_id: str, out_item_id: str, delta: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.output_audio_transcript.delta",
        "response_id": resp_id,
        "item_id": out_item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def audio_delta(resp_id: str, out_item_id: str, delta_b64: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.output_audio.delta",
        "response_id": resp_id,
        "item_id": out_item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta_b64,
    }


def output_audio_done(resp_id: str, out_item_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.output_audio.done",
        "response_id": resp_id,
        "item_id": out_item_id,
        "output_index": 0,
        "content_index": 0,
    }


def audio_transcript_done(resp_id: str, out_item_id: str, transcript: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.output_audio_transcript.done",
        "response_id": resp_id,
        "item_id": out_item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": transcript,
    }


def content_part_done(resp_id: str, out_item_id: str, transcript: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.content_part.done",
        "response_id": resp_id,
        "item_id": out_item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "audio", "transcript": transcript, "audio": ""},
    }


def output_item_done(resp_id: str, out_item_id: str, transcript: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.output_item.done",
        "response_id": resp_id,
        "output_index": 0,
        "item": {
            "id": out_item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "audio", "transcript": transcript}],
        },
    }


def response_done(resp_id: str, out_item_id: str, transcript: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.done",
        "response": {
            "id": resp_id,
            "status": "completed",
            "output": [{
                "id": out_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "audio", "transcript": transcript}],
            }],
        },
    }


def response_cancelled(resp_id: str) -> dict:
    return {
        "event_id": eid(),
        "type": "response.done",
        "response": {"id": resp_id, "status": "cancelled", "output": []},
    }


def output_audio_buffer_cleared() -> dict:
    return {"event_id": eid(), "type": "output_audio_buffer.cleared"}


def error_event(message: str) -> dict:
    return {
        "event_id": eid(),
        "type": "error",
        "error": {"type": "server_error", "message": message},
    }
