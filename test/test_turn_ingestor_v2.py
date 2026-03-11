from __future__ import annotations

from ros2_reception_orchestrator.turn_ingestor_v2 import TurnIngestor


def test_turn_ingestor_merges_within_window() -> None:
    ingestor = TurnIngestor(merge_window_sec=1.2)

    out1 = ingestor.accept(
        utterance_id='u1',
        text='こんにちは',
        confidence=0.9,
        captured_during_tts=False,
        session_id='s1',
        now_monotonic=0.0,
    )
    out2 = ingestor.accept(
        utterance_id='u2',
        text='島中です',
        confidence=0.95,
        captured_during_tts=False,
        session_id='s1',
        now_monotonic=0.5,
    )
    flushed = ingestor.flush_due(session_id='s1', now_monotonic=2.0)

    assert out1 == []
    assert out2 == []
    assert flushed is not None
    assert flushed.turn_seq == 1
    assert flushed.text == 'こんにちは 島中です'


def test_turn_ingestor_deduplicates_fast_repeats() -> None:
    ingestor = TurnIngestor(merge_window_sec=0.1)
    ingestor.accept(
        utterance_id='u1',
        text='同じ文です',
        confidence=0.9,
        captured_during_tts=False,
        session_id='s1',
        now_monotonic=0.0,
    )
    out = ingestor.accept(
        utterance_id='u2',
        text='同じ文です',
        confidence=0.9,
        captured_during_tts=False,
        session_id='s1',
        now_monotonic=0.2,
    )
    flushed = ingestor.flush_due(session_id='s1', now_monotonic=2.0)

    assert out == []
    assert flushed is not None
    assert flushed.turn_seq == 1
