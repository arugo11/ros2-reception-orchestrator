from __future__ import annotations

from pathlib import Path

from builtin_interfaces.msg import Time
from reception_interfaces.msg import ConversationTrace

from ros2_reception_orchestrator.conversation_log import ConversationLogWriter
from ros2_reception_orchestrator.conversation_trace import build_conversation_trace_message


def _stamp(sec: int, nanosec: int = 0) -> Time:
    stamp = Time()
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


def _read_single_log(output_dir: Path) -> str:
    files = sorted(output_dir.glob('*.txt'))
    assert len(files) == 1
    return files[0].read_text(encoding='utf-8')


def test_conversation_log_writer_persists_user_and_assistant_lines(tmp_path: Path) -> None:
    writer = ConversationLogWriter(enabled=True, output_dir=str(tmp_path))

    writer.record(
        build_conversation_trace_message(
            timestamp=_stamp(10),
            session_id='session-1',
            turn_seq=1,
            role=ConversationTrace.ROLE_USER,
            text='こんにちは',
            event_type='UTTERANCE_RECEIVED',
        )
    )
    writer.record(
        build_conversation_trace_message(
            timestamp=_stamp(11),
            session_id='session-1',
            turn_seq=1,
            role=ConversationTrace.ROLE_ASSISTANT,
            text='お名前を教えてください。',
            event_type='TTS_REQUESTED',
        )
    )
    writer.flush_all()

    content = _read_single_log(tmp_path)
    assert 'session_id: session-1' in content
    assert '[1970-01-01T00:00:10.000+00:00] user: こんにちは' in content
    assert '[1970-01-01T00:00:11.000+00:00] assistant: お名前を教えてください。' in content


def test_conversation_log_writer_ignores_internal_trace_events_by_default(tmp_path: Path) -> None:
    writer = ConversationLogWriter(enabled=True, output_dir=str(tmp_path))

    writer.record(
        build_conversation_trace_message(
            timestamp=_stamp(12),
            session_id='session-2',
            turn_seq=2,
            role=ConversationTrace.ROLE_SYSTEM,
            text='',
            event_type='TURN_PARSED',
            payload_json='{"speech_act":"inform"}',
        )
    )
    writer.flush_all()

    assert list(tmp_path.glob('*.txt')) == []


def test_conversation_log_writer_flushes_on_session_switch(tmp_path: Path) -> None:
    writer = ConversationLogWriter(enabled=True, output_dir=str(tmp_path))

    writer.record(
        build_conversation_trace_message(
            timestamp=_stamp(20),
            session_id='session-a',
            turn_seq=1,
            role=ConversationTrace.ROLE_SYSTEM,
            text='こんにちは。',
            event_type='TTS_REQUESTED',
        )
    )
    writer.record(
        build_conversation_trace_message(
            timestamp=_stamp(30),
            session_id='session-b',
            turn_seq=1,
            role=ConversationTrace.ROLE_USER,
            text='こんにちは',
            event_type='UTTERANCE_RECEIVED',
        )
    )
    writer.flush_all()

    files = sorted(tmp_path.glob('*.txt'))
    assert len(files) == 2
    contents = [path.read_text(encoding='utf-8') for path in files]
    assert any('session_id: session-a' in content for content in contents)
    assert any('session_id: session-b' in content for content in contents)


def test_conversation_log_writer_disabled_writes_nothing(tmp_path: Path) -> None:
    writer = ConversationLogWriter(enabled=False, output_dir=str(tmp_path))

    writer.record(
        build_conversation_trace_message(
            timestamp=_stamp(40),
            session_id='session-3',
            turn_seq=1,
            role=ConversationTrace.ROLE_USER,
            text='島中です',
            event_type='UTTERANCE_RECEIVED',
        )
    )
    writer.flush_all()

    assert list(tmp_path.glob('*.txt')) == []
