"""Snapshot-ish tests for the protocol event builders."""

from __future__ import annotations

from voice_agent import protocol

# (builder callable, args, expected "type")
_CASES = [
    (protocol.session_created, ({"id": "s"},), "session.created"),
    (protocol.session_updated, ({"id": "s"},), "session.updated"),
    (protocol.speech_started, ("item_1",), "input_audio_buffer.speech_started"),
    (protocol.speech_stopped, ("item_1", 500), "input_audio_buffer.speech_stopped"),
    (protocol.audio_buffer_committed, ("item_1",), "input_audio_buffer.committed"),
    (protocol.audio_buffer_cleared, (), "input_audio_buffer.cleared"),
    (protocol.conversation_item_created, ("item_1", "user", []), "conversation.item.created"),
    (protocol.transcription_completed, ("item_1", "hi"),
     "conversation.item.input_audio_transcription.completed"),
    (protocol.response_created, ("resp_1",), "response.created"),
    (protocol.output_item_added, ("resp_1", "out_1"), "response.output_item.added"),
    (protocol.content_part_added, ("resp_1", "out_1"), "response.content_part.added"),
    (protocol.audio_transcript_delta, ("resp_1", "out_1", "x"),
     "response.output_audio_transcript.delta"),
    (protocol.audio_delta, ("resp_1", "out_1", "b64"), "response.output_audio.delta"),
    (protocol.output_audio_done, ("resp_1", "out_1"), "response.output_audio.done"),
    (protocol.audio_transcript_done, ("resp_1", "out_1", "t"),
     "response.output_audio_transcript.done"),
    (protocol.content_part_done, ("resp_1", "out_1", "t"), "response.content_part.done"),
    (protocol.output_item_done, ("resp_1", "out_1", "t"), "response.output_item.done"),
    (protocol.response_done, ("resp_1", "out_1", "t"), "response.done"),
    (protocol.response_cancelled, ("resp_1",), "response.done"),
    (protocol.output_audio_buffer_cleared, (), "output_audio_buffer.cleared"),
    (protocol.error_event, ("boom",), "error"),
]


def test_builders_have_type_and_event_id():
    for builder, args, expected_type in _CASES:
        event = builder(*args)
        assert isinstance(event, dict), builder.__name__
        assert event.get("type") == expected_type, builder.__name__
        assert event.get("event_id", "").startswith("evt_"), builder.__name__


def test_id_generators_prefixes():
    assert protocol.eid().startswith("evt_")
    assert protocol.iid().startswith("item_")
    assert protocol.rid().startswith("resp_")


def test_ids_are_unique():
    assert protocol.eid() != protocol.eid()
    assert protocol.iid() != protocol.iid()
    assert protocol.rid() != protocol.rid()


def test_response_cancelled_status():
    event = protocol.response_cancelled("resp_1")
    assert event["response"]["status"] == "cancelled"


def test_error_event_carries_message():
    event = protocol.error_event("boom")
    assert event["error"]["message"] == "boom"
