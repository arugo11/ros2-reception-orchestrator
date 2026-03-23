from __future__ import annotations

from datetime import UTC
from datetime import datetime

from ros2_reception_orchestrator.session_reducer_v2 import SessionReducer
from ros2_reception_orchestrator.v2_types import BeliefOperationData
from ros2_reception_orchestrator.v2_types import SemanticDecisionData


def test_collecting_updates_only_name_and_asks_affiliation() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)

    outcome = reducer.apply(
        turn_seq=1,
        utterance_text='島中です',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='inform',
            target_slot='name',
            ambiguity='low',
            confidence=0.9,
            operations=[
                BeliefOperationData(
                    op='set_slot',
                    slot='name',
                    value='島中',
                    grounded_text='島中です',
                    confidence=0.9,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'ask_affiliation'
    assert reducer.state.phase == 'collecting'
    assert reducer.state.working_info.name == '島中'
    assert reducer.state.working_info.affiliation == ''
    assert reducer.state.committed_info.name == ''
    assert outcome.outbox_items == []


def test_affiliation_focus_replacement_does_not_pollute_other_slots() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)
    reducer.state.working_info.name = '島中'
    reducer.state.focus_slot = 'affiliation'
    reducer.state.last_system_act = 'ask_affiliation'

    outcome = reducer.apply(
        turn_seq=2,
        utterance_text='菅屋研究室です',
        decision=SemanticDecisionData(
            turn_seq=2,
            speech_act='inform',
            target_slot='affiliation',
            ambiguity='low',
            confidence=0.92,
            operations=[
                BeliefOperationData(
                    op='set_slot',
                    slot='affiliation',
                    value='菅屋研究室',
                    grounded_text='菅屋研究室です',
                    confidence=0.92,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'ask_purpose'
    assert reducer.state.working_info.name == '島中'
    assert reducer.state.working_info.affiliation == '菅屋研究室'
    assert reducer.state.working_info.purpose == ''


def test_same_value_multi_slot_operations_are_rejected_and_clarified() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)
    reducer.state.working_info.name = '島中'
    reducer.state.focus_slot = 'affiliation'

    outcome = reducer.apply(
        turn_seq=3,
        utterance_text='それじゃなくて、菅屋研究室です',
        decision=SemanticDecisionData(
            turn_seq=3,
            speech_act='correction',
            target_slot='affiliation',
            ambiguity='medium',
            requires_confirmation=True,
            confidence=0.88,
            operations=[
                BeliefOperationData(op='replace_slot', slot='name', value='菅屋研究室', confidence=0.88),
                BeliefOperationData(op='replace_slot', slot='affiliation', value='菅屋研究室', confidence=0.88),
                BeliefOperationData(op='replace_slot', slot='purpose', value='菅屋研究室', confidence=0.88),
            ],
        ),
    )

    assert outcome.dialog_act == 'clarify_affiliation'
    assert reducer.state.working_info.name == '島中'
    assert reducer.state.working_info.affiliation == ''
    assert reducer.state.pending_clarification_slot == 'affiliation'


def test_confirm_snapshot_commits_and_enqueues_outbox() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)
    reducer.state.working_info.name = '島中'
    reducer.state.working_info.affiliation = '菅屋研究室'
    reducer.state.working_info.purpose = '打ち合わせです'
    reducer.state.phase = 'confirming'

    outcome = reducer.apply(
        turn_seq=4,
        utterance_text='はい',
        decision=SemanticDecisionData(
            turn_seq=4,
            speech_act='affirm',
            ambiguity='low',
            confidence=0.95,
            operations=[
                BeliefOperationData(
                    op='confirm_working_state',
                    slot='none',
                    confidence=0.95,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'notify_waiting'
    assert reducer.state.phase == 'notified_waiting'
    assert reducer.state.committed_info.name == '島中'
    assert len(outcome.outbox_items) == 1
    assert outcome.outbox_items[0].event_type == 'confirmed_snapshot'


def test_mark_tts_completed_refreshes_last_activity_timestamp() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)
    reducer.state.latest_applied_turn = 3
    reducer.state.last_activity_at = datetime(2026, 1, 1, tzinfo=UTC)

    reducer.mark_tts_completed(turn_seq=3, dialog_act='ask_affiliation')

    assert reducer.state.last_activity_at > datetime(2026, 1, 1, tzinfo=UTC)


def test_low_confidence_language_does_not_flip_response_language() -> None:
    reducer = SessionReducer(confidence_threshold=0.55)
    reducer.state.response_language = 'ja'

    reducer.apply(
        turn_seq=1,
        utterance_text='hello',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='greeting',
            detected_language='en',
            ambiguity='medium',
            confidence=0.2,
        ),
    )

    assert reducer.state.response_language == 'ja'


def test_greeting_with_bogus_slot_write_is_ignored_and_keeps_name_prompt() -> None:
    reducer = SessionReducer(confidence_threshold=0.55)

    outcome = reducer.apply(
        turn_seq=1,
        utterance_text='こんにちは',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='greeting',
            target_slot='name',
            ambiguity='low',
            confidence=0.95,
            operations=[
                BeliefOperationData(
                    op='replace_slot',
                    slot='name',
                    value='おはようございます',
                    grounded_text='こんにちは',
                    confidence=0.95,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'ask_name'
    assert reducer.state.working_info.name == ''


def test_confirming_affirm_rescues_invalid_slot_operation_and_commits() -> None:
    reducer = SessionReducer(confidence_threshold=0.55)
    reducer.state.working_info.name = '島中'
    reducer.state.working_info.affiliation = '菅屋研究室'
    reducer.state.working_info.purpose = '打ち合わせ'
    reducer.state.phase = 'confirming'

    outcome = reducer.apply(
        turn_seq=2,
        utterance_text='はい',
        decision=SemanticDecisionData(
            turn_seq=2,
            speech_act='affirm',
            ambiguity='low',
            confidence=0.95,
            operations=[
                BeliefOperationData(
                    op='replace_slot',
                    slot='none',
                    value='none',
                    grounded_text='はい',
                    confidence=0.95,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'notify_waiting'
    assert reducer.state.committed_info.name == '島中'
    assert len(outcome.outbox_items) == 1


def test_ungrounded_slot_write_is_rejected_and_clarified() -> None:
    reducer = SessionReducer(confidence_threshold=0.55)
    reducer.state.working_info.name = '島中'
    reducer.state.working_info.affiliation = '菅屋研究室'
    reducer.state.focus_slot = 'purpose'

    outcome = reducer.apply(
        turn_seq=3,
        utterance_text='Det.',
        decision=SemanticDecisionData(
            turn_seq=3,
            speech_act='inform',
            target_slot='purpose',
            ambiguity='low',
            confidence=0.95,
            operations=[
                BeliefOperationData(
                    op='replace_slot',
                    slot='purpose',
                    value='研究',
                    grounded_text='Det.',
                    confidence=0.95,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'clarify_purpose'
    assert reducer.state.working_info.purpose == ''


def test_prompt_leak_slot_write_is_rejected_and_clarified() -> None:
    reducer = SessionReducer(confidence_threshold=0.55)
    reducer.state.focus_slot = 'name'

    outcome = reducer.apply(
        turn_seq=1,
        utterance_text='.',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='inform',
            target_slot='name',
            ambiguity='low',
            confidence=1.0,
            operations=[
                BeliefOperationData(
                    op='set_slot',
                    slot='name',
                    value='島中です。',
                    grounded_text='working_name=島中です。',
                    confidence=1.0,
                )
            ],
        ),
    )

    assert outcome.dialog_act == 'clarify_name'
    assert reducer.state.working_info.name == ''
