from __future__ import annotations

from dataclasses import dataclass
import time

from .v2_types import TurnEnvelopeData


@dataclass(slots=True)
class _PendingTurn:
    utterance_id: str
    text: str
    confidence: float
    captured_during_tts: bool
    due_monotonic: float


class TurnIngestor:
    """Normalize ASR utterances into deterministic turn envelopes."""

    def __init__(self, *, merge_window_sec: float = 1.2) -> None:
        self._merge_window_sec = float(merge_window_sec)
        self._pending: _PendingTurn | None = None
        self._last_text = ''
        self._last_text_at = 0.0
        self._turn_seq = 0

    def accept(
        self,
        *,
        utterance_id: str,
        text: str,
        confidence: float,
        captured_during_tts: bool,
        session_id: str,
        now_monotonic: float | None = None,
    ) -> list[TurnEnvelopeData]:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        cleaned = text.strip()
        if not cleaned:
            return []

        if cleaned == self._last_text and now - self._last_text_at < 1.0:
            return []
        self._last_text = cleaned
        self._last_text_at = now

        out: list[TurnEnvelopeData] = []
        pending = self._pending
        if pending is None:
            self._pending = _PendingTurn(
                utterance_id=utterance_id,
                text=cleaned,
                confidence=float(confidence),
                captured_during_tts=bool(captured_during_tts),
                due_monotonic=now + self._merge_window_sec,
            )
            return out

        if now <= pending.due_monotonic:
            merged = ' '.join(chunk for chunk in (pending.text, cleaned) if chunk).strip()
            self._pending = _PendingTurn(
                utterance_id=utterance_id,
                text=merged,
                confidence=max(float(confidence), pending.confidence),
                captured_during_tts=(pending.captured_during_tts or bool(captured_during_tts)),
                due_monotonic=now + self._merge_window_sec,
            )
            return out

        out.append(self._finalize_pending(session_id=session_id))
        self._pending = _PendingTurn(
            utterance_id=utterance_id,
            text=cleaned,
            confidence=float(confidence),
            captured_during_tts=bool(captured_during_tts),
            due_monotonic=now + self._merge_window_sec,
        )
        return out

    def flush_due(self, *, session_id: str, now_monotonic: float | None = None) -> TurnEnvelopeData | None:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        if self._pending is None or now < self._pending.due_monotonic:
            return None
        return self._finalize_pending(session_id=session_id)

    def reset(self) -> None:
        self._pending = None
        self._last_text = ''
        self._last_text_at = 0.0
        self._turn_seq = 0

    def _finalize_pending(self, *, session_id: str) -> TurnEnvelopeData:
        pending = self._pending
        if pending is None:
            raise RuntimeError('no pending turn to finalize')
        self._pending = None
        self._turn_seq += 1
        return TurnEnvelopeData(
            session_id=session_id,
            turn_seq=self._turn_seq,
            utterance_id=pending.utterance_id,
            text=pending.text,
            captured_during_tts=pending.captured_during_tts,
            asr_confidence=pending.confidence,
        )
