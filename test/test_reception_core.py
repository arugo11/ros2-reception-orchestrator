from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

from ros2_reception_orchestrator.session_manager import ReceptionOrchestratorCore
from ros2_reception_orchestrator.state_models import SupervisorDecision
from ros2_reception_orchestrator.state_models import ThreadCreationResult


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _make_core() -> ReceptionOrchestratorCore:
    return ReceptionOrchestratorCore(inactivity_reset_sec=60)


def test_begin_turn_starts_session_and_requests_thread():
    core = _make_core()

    turn = core.begin_turn(utterance_id='u1', text='こんにちは', now=_now())

    assert turn is not None
    assert turn.create_thread is False
    assert turn.initial_thread_text == ''
    assert core.session is not None
    assert core.session.phase == 'collecting'


def test_supervisor_reducer_collects_slots_in_any_order():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です', now=_now())
    assert turn is not None

    outcome = core.reduce_supervisor_turn(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        utterance_text='島中です',
        decision=SupervisorDecision(
            speech_act='inform',
            extracted_name='島中',
            missing_fields=['affiliation', 'purpose'],
            next_dialog_act='ask_affiliation',
        ),
        now=_now(),
    )

    assert outcome is not None
    assert core.session is not None
    assert core.session.visitor_info.name == '島中'
    assert outcome.dialog_request is not None
    assert outcome.dialog_request.dialog_act == 'ask_affiliation'
    assert outcome.create_thread is True
    assert outcome.initial_thread_text


def test_reducer_transitions_to_confirm_when_all_slots_present():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です。菅谷研究室です。面会です。', now=_now())
    assert turn is not None

    outcome = core.reduce_supervisor_turn(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        utterance_text='島中です。菅谷研究室です。面会です。',
        decision=SupervisorDecision(
            speech_act='inform',
            extracted_name='島中',
            extracted_affiliation='菅谷研究室',
            extracted_purpose='面会',
            should_confirm=True,
            missing_fields=[],
            next_dialog_act='confirm',
        ),
        now=_now(),
    )

    assert outcome is not None
    assert outcome.dialog_request is not None
    assert outcome.dialog_request.dialog_act == 'confirm'
    assert core.session is not None
    assert core.session.phase == 'confirming'


def test_confirming_affirm_moves_to_waiting_and_confirms_discord():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です。菅谷研究室です。面会です。', now=_now())
    assert turn is not None
    core.reduce_supervisor_turn(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        utterance_text='島中です。菅谷研究室です。面会です。',
        decision=SupervisorDecision(
            speech_act='inform',
            extracted_name='島中',
            extracted_affiliation='菅谷研究室',
            extracted_purpose='面会',
            should_confirm=True,
            missing_fields=[],
            next_dialog_act='confirm',
        ),
        now=_now(),
    )

    turn2 = core.begin_turn(utterance_id='u2', text='はい', now=_now())
    assert turn2 is not None
    outcome = core.reduce_supervisor_turn(
        session_id=turn2.session_id,
        turn_id=turn2.turn_id,
        utterance_text='はい',
        decision=SupervisorDecision(
            speech_act='affirm',
            should_confirm=True,
            missing_fields=[],
            next_dialog_act='notify_waiting',
            discord_update_kind='confirmed',
        ),
        now=_now(),
    )

    assert outcome is not None
    assert outcome.dialog_request is not None
    assert outcome.dialog_request.dialog_act == 'notify_waiting'
    assert outcome.discord_update_kind == 'confirmed'
    assert core.session is not None
    assert core.session.phase == 'notified_waiting'


def test_secretary_reply_is_deduplicated():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です', now=_now())
    assert turn is not None
    assert core.session is not None
    core.session.phase = 'notified_waiting'
    core.session.discord.thread_id = 'thread-1'

    request1 = core.handle_secretary_reply(
        thread_id='thread-1',
        message_id='m1',
        text='担当者が向かいます。',
        now=_now(),
    )
    request2 = core.handle_secretary_reply(
        thread_id='thread-1',
        message_id='m1',
        text='担当者が向かいます。',
        now=_now(),
    )

    assert request1 is not None
    assert request1.dialog_act == 'relay_secretary'
    assert request2 is None


def test_handle_thread_created_returns_canonical_post():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です', now=_now())
    assert turn is not None
    assert core.session is not None
    core.session.visitor_info.name = '島中'

    text = core.handle_thread_created(
        ThreadCreationResult(
            session_id=core.session.session_id,
            success=True,
            thread_id='thread-1',
            channel_id='channel-1',
        )
    )

    assert text is not None
    assert '島中' in text
    assert core.session.discord.thread_id == 'thread-1'


def test_stale_supervisor_result_is_ignored():
    core = _make_core()
    turn1 = core.begin_turn(utterance_id='u1', text='島中です', now=_now())
    assert turn1 is not None
    turn2 = core.begin_turn(utterance_id='u2', text='菅谷研究室です', now=_now())
    assert turn2 is not None

    outcome = core.reduce_supervisor_turn(
        session_id=turn1.session_id,
        turn_id=turn1.turn_id,
        utterance_text='島中です',
        decision=SupervisorDecision(
            speech_act='inform',
            extracted_name='島中',
            next_dialog_act='ask_affiliation',
            missing_fields=['affiliation', 'purpose'],
        ),
        now=_now(),
    )

    assert outcome is None


def test_discord_update_deduplicates_same_text():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です', now=_now())
    assert turn is not None
    assert core.session is not None
    core.session.discord.thread_id = 'thread-1'

    outcome1 = core.reduce_supervisor_turn(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        utterance_text='島中です',
        decision=SupervisorDecision(
            speech_act='inform',
            extracted_name='島中',
            next_dialog_act='ask_affiliation',
            missing_fields=['affiliation', 'purpose'],
        ),
        now=_now(),
    )

    turn2 = core.begin_turn(utterance_id='u2', text='島中です', now=_now())
    assert turn2 is not None
    outcome2 = core.reduce_supervisor_turn(
        session_id=turn2.session_id,
        turn_id=turn2.turn_id,
        utterance_text='島中です',
        decision=SupervisorDecision(
            speech_act='inform',
            extracted_name='島中',
            next_dialog_act='ask_affiliation',
            missing_fields=['affiliation', 'purpose'],
        ),
        now=_now(),
    )

    assert outcome1 is not None
    assert outcome1.discord_text
    assert outcome2 is not None
    assert outcome2.discord_text == ''


def test_inactivity_resets_session():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='島中です', now=_now())
    assert turn is not None
    assert core.session is not None

    timed_out = core.handle_inactivity(
        now=core.session.last_activity_at + timedelta(seconds=61)
    )

    assert timed_out is True
    assert core.session is None


def test_noise_input_does_not_create_thread_or_dialog():
    core = _make_core()
    turn = core.begin_turn(utterance_id='u1', text='うーん', now=_now())
    assert turn is not None

    outcome = core.reduce_supervisor_turn(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        utterance_text='うーん',
        decision=SupervisorDecision(
            speech_act='unknown',
            ignore_input=True,
            next_dialog_act='retry',
            missing_fields=['name', 'affiliation', 'purpose'],
        ),
        now=_now(),
    )

    assert outcome is not None
    assert outcome.create_thread is False
    assert outcome.dialog_request is None
