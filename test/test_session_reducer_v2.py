from __future__ import annotations

from reception_interfaces.msg import ExecutionCommand

from ros2_reception_orchestrator.session_reducer_v2 import SessionReducer
from ros2_reception_orchestrator.v2_types import SemanticDecisionData
from ros2_reception_orchestrator.v2_types import VisitorInfoData


def test_collecting_transitions_to_missing_slot_prompt() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)

    outcome = reducer.apply(
        turn_seq=1,
        utterance_text='島中です',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='inform',
            slot_patch=VisitorInfoData(name='島中'),
            confidence=0.9,
        ),
    )

    assert outcome.dialog_act == 'ask_affiliation'
    assert reducer.state.phase == 'collecting'
    assert reducer.state.visitor_info.name == '島中'
    assert any(cmd.command_type == ExecutionCommand.COMMAND_DISCORD_CREATE for cmd in outcome.commands)


def test_collecting_to_confirming_when_all_slots_present() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)

    outcome = reducer.apply(
        turn_seq=1,
        utterance_text='島中です。菅谷研究室です。学長に会いに来ました。',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='inform',
            slot_patch=VisitorInfoData(
                name='島中',
                affiliation='菅谷研究室',
                purpose='学長に会いに来ました',
            ),
            confidence=0.95,
        ),
    )

    assert outcome.dialog_act == 'confirm'
    assert reducer.state.phase == 'confirming'
    assert reducer.state.pending_confirmation.purpose == '学長に会いに来ました'


def test_confirming_affirm_moves_to_notified_waiting() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)

    reducer.apply(
        turn_seq=1,
        utterance_text='島中です。菅谷研究室です。学長に会いに来ました。',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='inform',
            slot_patch=VisitorInfoData(
                name='島中',
                affiliation='菅谷研究室',
                purpose='学長に会いに来ました',
            ),
            confidence=0.95,
        ),
    )

    outcome = reducer.apply(
        turn_seq=2,
        utterance_text='はい',
        decision=SemanticDecisionData(
            turn_seq=2,
            speech_act='affirm',
            confidence=0.9,
        ),
    )

    assert outcome.dialog_act == 'notify_waiting'
    assert reducer.state.phase == 'notified_waiting'


def test_low_confidence_does_not_commit_slots() -> None:
    reducer = SessionReducer(confidence_threshold=0.8)

    reducer.apply(
        turn_seq=1,
        utterance_text='島中です',
        decision=SemanticDecisionData(
            turn_seq=1,
            speech_act='inform',
            slot_patch=VisitorInfoData(name='島中'),
            confidence=0.4,
        ),
    )

    assert reducer.state.visitor_info.name == ''


def test_stale_turn_is_not_applied() -> None:
    reducer = SessionReducer(confidence_threshold=0.5)
    reducer.apply(
        turn_seq=2,
        utterance_text='島中です',
        decision=SemanticDecisionData(
            turn_seq=2,
            speech_act='inform',
            slot_patch=VisitorInfoData(name='島中'),
            confidence=0.9,
        ),
    )

    outcome = reducer.apply(
        turn_seq=1,
        utterance_text='古い発話',
        decision=SemanticDecisionData(turn_seq=1, speech_act='inform', confidence=0.9),
    )

    assert outcome.should_render_response is False
    assert reducer.state.latest_applied_turn == 2
