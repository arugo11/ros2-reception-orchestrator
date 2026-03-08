from __future__ import annotations

from ros2_reception_orchestrator.state_models import SessionSnapshot
from ros2_reception_orchestrator.state_models import VisitorInfo
from ros2_reception_orchestrator.supervisor_adapter import SupervisorAdapter


def _snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        session_id='session-1',
        phase='collecting',
        visitor_info=VisitorInfo(),
        last_user_utterance='',
        last_dialog_act=None,
        pending_confirmation=None,
        latest_turn_id=1,
    )


def test_supervisor_adapter_parses_valid_json():
    calls = []

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        calls.append((session_id, user_message, system_prompt, temperature, max_tokens, stateless))
        return (
            '{"speech_act":"inform","extracted_name":"島中","extracted_affiliation":null,'
            '"extracted_purpose":null,"slot_confidence":0.9,"missing_fields":["affiliation","purpose"],'
            '"next_dialog_act":"ask_affiliation","should_confirm":false,'
            '"correction_target":"none","discord_update_kind":"initial","ignore_input":false}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(_snapshot(), '島中です')

    assert decision.extracted_name == '島中'
    assert decision.next_dialog_act == 'ask_affiliation'
    assert len(calls) == 1
    assert calls[0][-1] is True


def test_supervisor_adapter_repairs_invalid_first_response():
    responses = iter(
        [
            '以下は例です```json {"utterance":"島中です"} ```',
            (
                '{"speech_act":"inform","extracted_name":"島中","extracted_affiliation":null,'
                '"extracted_purpose":null,"slot_confidence":0.8,"missing_fields":["affiliation","purpose"],'
                '"next_dialog_act":"ask_affiliation","should_confirm":false,'
                '"correction_target":"none","discord_update_kind":"initial","ignore_input":false}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(_snapshot(), '島中です')

    assert decision.extracted_name == '島中'
    assert decision.ignore_input is False


def test_supervisor_adapter_normalizes_inconsistent_llm_decision():
    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        return (
            '{"speech_act":"inform","extracted_name":"島中","extracted_affiliation":null,'
            '"extracted_purpose":null,"slot_confidence":0.95,"missing_fields":[],"next_dialog_act":"ask_purpose",'
            '"should_confirm":true,"correction_target":"all","discord_update_kind":"initial","ignore_input":false}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(_snapshot(), '名前は島中です')

    assert decision.extracted_name == '島中'
    assert decision.missing_fields == ['affiliation', 'purpose']
    assert decision.next_dialog_act == 'ask_affiliation'
    assert decision.should_confirm is False
    assert decision.correction_target == 'none'


def test_supervisor_adapter_treats_unknown_placeholder_as_missing():
    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        return (
            '{"speech_act":"greeting","extracted_name":"unknown","extracted_affiliation":"unknown",'
            '"extracted_purpose":"unknown","slot_confidence":0.0,"missing_fields":[],"next_dialog_act":"confirm",'
            '"should_confirm":true,"correction_target":"none","discord_update_kind":"none","ignore_input":false}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(_snapshot(), 'こんにちは')

    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['name', 'affiliation', 'purpose']
    assert decision.next_dialog_act == 'ask_name'
    assert decision.should_confirm is False


def test_supervisor_adapter_accepts_partial_json_without_fallback():
    calls = []

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        calls.append('called')
        return '{"speech_act":"inform","extracted_name":"島中","next_dialog_act":"ask_affiliation"}'

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(_snapshot(), '名前は島中です')

    assert decision.extracted_name == '島中'
    assert decision.missing_fields == ['affiliation', 'purpose']
    assert decision.next_dialog_act == 'ask_affiliation'
    assert len(calls) == 1


def test_supervisor_adapter_enriches_missing_purpose_with_second_pass():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'
    snapshot.visitor_info.affiliation = '菅谷研究室'
    snapshot.last_dialog_act = 'ask_purpose'

    responses = iter(
        [
            (
                '{"speech_act":"inform","extracted_name":"島中","extracted_affiliation":"菅谷研究室",'
                '"extracted_purpose":null,"slot_confidence":0.8,"missing_fields":["purpose"],'
                '"next_dialog_act":"ask_purpose","should_confirm":false,'
                '"correction_target":"none","discord_update_kind":"update","ignore_input":false}'
            ),
            '{"name":null,"affiliation":null,"purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(snapshot, '学長に会いに来ました')

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == []
    assert decision.next_dialog_act == 'confirm'
    assert decision.should_confirm is True


def test_supervisor_adapter_promotes_deny_with_new_name_to_correction():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '下中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            '{"speech_act":"deny","next_dialog_act":"ask_name"}',
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(snapshot, '名前が違います。島中です。')

    assert decision.speech_act == 'correction'
    assert decision.extracted_name == '島中'
    assert decision.correction_target == 'name'
    assert decision.next_dialog_act == 'ask_affiliation'


def test_supervisor_adapter_repair_does_not_overwrite_useful_primary_fields():
    responses = iter(
        [
            (
                '{"speech_act":"inform","extracted_name":"島中","extracted_affiliation":"菅谷研究室",'
                '"extracted_purpose":"学長に会いに来ました","slot_confidence":0.95}'
            ),
            (
                '{"speech_act":"inform","extracted_name":null,"extracted_affiliation":null,'
                '"extracted_purpose":null,"missing_fields":[],"next_dialog_act":"confirm",'
                '"should_confirm":true,"correction_target":"none","discord_update_kind":"update","ignore_input":false}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless
        return next(responses)

    snapshot = _snapshot()
    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(snapshot, '島中です。菅谷研究室です。学長に会いに来ました。')

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.next_dialog_act == 'confirm'
    assert decision.should_confirm is True


def test_supervisor_adapter_targets_name_only_for_name_correction():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '下中'
    snapshot.last_dialog_act = 'ask_affiliation'
    invocations = []
    responses = iter(
        [
            '{"speech_act":"deny","next_dialog_act":"ask_name"}',
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless):
        del system_prompt, temperature, max_tokens, stateless
        invocations.append((session_id, user_message))
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(snapshot, '名前が違います。島中です。')

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation is None
    assert decision.correction_target == 'name'
    assert any('target_fields=name' in user_message for _session_id, user_message in invocations[1:])
