from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from typing import Any

from builtin_interfaces.msg import Time
from reception_interfaces.msg import ConversationTrace


ROLE_LABELS = {
    int(ConversationTrace.ROLE_UNKNOWN): 'unknown',
    int(ConversationTrace.ROLE_USER): 'user',
    int(ConversationTrace.ROLE_ASSISTANT): 'assistant',
    int(ConversationTrace.ROLE_SYSTEM): 'system',
}


def build_conversation_trace_message(
    *,
    timestamp: Time,
    session_id: str,
    turn_seq: int,
    role: int,
    text: str,
    dialog_act: str = '',
    phase: str = '',
    utterance_id: str = '',
    asr_confidence: float = 0.0,
    event_type: str = '',
    event_payload: str = '',
    payload_json: str = '',
) -> ConversationTrace:
    msg = ConversationTrace()
    msg.timestamp = timestamp
    msg.session_id = str(session_id)
    msg.turn_seq = int(turn_seq)
    msg.role = int(role)
    msg.text = str(text)
    msg.dialog_act = str(dialog_act)
    msg.phase = str(phase)
    msg.utterance_id = str(utterance_id)
    msg.asr_confidence = float(asr_confidence)
    msg.event_type = str(event_type)
    msg.event_payload = str(event_payload)
    msg.payload_json = str(payload_json)
    return msg


def conversation_trace_to_dict(msg: ConversationTrace) -> dict[str, Any]:
    return {
        'timestamp': ros_time_to_iso8601(msg.timestamp),
        'session_id': msg.session_id,
        'turn_seq': int(msg.turn_seq),
        'role': ROLE_LABELS.get(int(msg.role), 'unknown'),
        'role_code': int(msg.role),
        'text': msg.text,
        'dialog_act': msg.dialog_act,
        'phase': msg.phase,
        'utterance_id': msg.utterance_id,
        'asr_confidence': round(float(msg.asr_confidence), 3),
        'event_type': msg.event_type,
        'event_payload': msg.event_payload,
        'event_payload_decoded': _parse_payload(msg.event_payload),
        'payload_json': msg.payload_json,
        'payload': _parse_payload(msg.payload_json),
    }


def _parse_payload(raw: str) -> Any:
    raw = str(raw or '').strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {'raw': raw}


def ros_time_to_iso8601(stamp: Time) -> str:
    seconds = float(int(stamp.sec)) + (float(int(stamp.nanosec)) / 1_000_000_000.0)
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat(timespec='milliseconds')
