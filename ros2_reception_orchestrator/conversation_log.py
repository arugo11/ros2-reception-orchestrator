from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from reception_interfaces.msg import ConversationTrace

from .conversation_trace import ROLE_LABELS
from .conversation_trace import ros_time_to_iso8601


_UTTERANCE_EVENT_TYPES = {'UTTERANCE_RECEIVED', 'TTS_REQUESTED'}


@dataclass(slots=True)
class _SessionLogBuffer:
    session_id: str
    started_at: str
    lines: list[str]


class ConversationLogWriter:
    """Persist human-readable per-session conversation logs."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        log_format: str = 'text',
        scope: str = 'utterances',
        flush_on_session_switch: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._output_dir = Path(output_dir).expanduser()
        self._format = str(log_format or 'text').strip().lower()
        self._scope = str(scope or 'utterances').strip().lower()
        self._flush_on_session_switch = bool(flush_on_session_switch)
        self._lock = threading.RLock()
        self._buffers: dict[str, _SessionLogBuffer] = {}
        self._active_session_id = ''

    def record(self, msg: ConversationTrace) -> None:
        if not self._enabled or self._format != 'text':
            return
        if not self._should_record(msg):
            return

        session_id = str(msg.session_id or '').strip()
        if not session_id:
            return

        with self._lock:
            if (
                self._flush_on_session_switch
                and self._active_session_id
                and session_id != self._active_session_id
            ):
                self._flush_session_locked(self._active_session_id)

            buffer = self._buffers.get(session_id)
            if buffer is None:
                buffer = _SessionLogBuffer(
                    session_id=session_id,
                    started_at=ros_time_to_iso8601(msg.timestamp),
                    lines=[],
                )
                self._buffers[session_id] = buffer

            buffer.lines.append(self._format_line(msg))
            self._active_session_id = session_id

    def flush_all(self) -> None:
        if not self._enabled or self._format != 'text':
            return
        with self._lock:
            for session_id in list(self._buffers.keys()):
                self._flush_session_locked(session_id)
            self._active_session_id = ''

    def _should_record(self, msg: ConversationTrace) -> bool:
        role = ROLE_LABELS.get(int(msg.role), 'unknown')
        if role not in {'user', 'assistant', 'system'}:
            return False
        if not str(msg.text or '').strip():
            return False
        if self._scope == 'utterances':
            return str(msg.event_type or '').strip() in _UTTERANCE_EVENT_TYPES
        return True

    def _format_line(self, msg: ConversationTrace) -> str:
        timestamp = ros_time_to_iso8601(msg.timestamp)
        role = ROLE_LABELS.get(int(msg.role), 'unknown')
        text = str(msg.text or '').strip().replace('\n', ' ')
        return f'[{timestamp}] {role}: {text}'

    def _flush_session_locked(self, session_id: str) -> None:
        buffer = self._buffers.pop(session_id, None)
        if buffer is None or not buffer.lines:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'{self._filename_prefix(buffer.started_at)}_{buffer.session_id[:8]}.txt'
        path = self._output_dir / filename
        content = (
            f'session_id: {buffer.session_id}\n'
            f'started_at: {buffer.started_at}\n'
            '\n'
            + '\n'.join(buffer.lines)
            + '\n'
        )
        path.write_text(content, encoding='utf-8')

    @staticmethod
    def _filename_prefix(timestamp: str) -> str:
        sanitized = str(timestamp).replace('+00:00', 'Z')
        return sanitized.replace(':', '-')
