from __future__ import annotations

from ros2_reception_orchestrator.state_models import SessionSnapshot
from ros2_reception_orchestrator.state_models import VisitorInfo
from ros2_reception_orchestrator.supervisor_adapter import SupervisorAdapter
from ros2_reception_orchestrator.supervisor_adapter import _target_fields_for_slot_rescue


def _snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        session_id='session-1',
        phase='collecting',
        visitor_info=VisitorInfo(),
        last_user_utterance='',
        last_dialog_act=None,
        last_spoken_text='',
        pending_confirmation=None,
        latest_turn_id=1,
    )


def test_supervisor_adapter_parses_valid_json():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":null,"purpose":null},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"ご所属を教えていただけますか。"}'
            ),
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '島中です',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.missing_fields == ['affiliation', 'purpose']
    assert decision.spoken_response == 'ご所属を教えていただけますか。'


def test_supervisor_adapter_repairs_invalid_first_response():
    responses = iter(
        [
            '以下は例です```json {"utterance":"島中です"} ```',
            (
                '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":null,"purpose":null},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.8,"spoken_response":"ご所属を教えていただけますか。"}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '島中です',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.spoken_response == 'ご所属を教えていただけますか。'


def test_supervisor_adapter_treats_unknown_placeholder_as_missing():
    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return (
            '{"speech_act":"greeting","slot_updates":{"name":"unknown","affiliation":"unknown","purpose":"unknown"},'
            '"correction":{"target":"none","overwrite":false},'
            '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
            '"confidence":0.0,"spoken_response":"お名前を伺ってもよろしいでしょうか。"}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        'こんにちは',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['name', 'affiliation', 'purpose']


def test_supervisor_adapter_promotes_deny_with_new_name_to_correction():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '下中'
    snapshot.last_dialog_act = 'ask_affiliation'

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return (
            '{"speech_act":"deny","slot_updates":{"name":"島中","affiliation":null,"purpose":null},'
            '"correction":{"target":"name","overwrite":true},'
            '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
            '"confidence":0.9,"spoken_response":"失礼しました。ご所属を教えていただけますか。"}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '名前が違います。島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.speech_act == 'correction'
    assert decision.extracted_name == '島中'
    assert decision.correction_target == 'name'


def test_supervisor_adapter_accepts_affiliation_and_purpose_together():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.95,"spoken_response":"お名前は島中様、ご所属は菅谷研究室、ご用件は学長との面会でお間違いないでしょうか。"}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == []
    assert decision.should_confirm is True


def test_supervisor_adapter_ignores_inform_with_stale_name_and_fake_correction_flag():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '下中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"下中","affiliation":null,"purpose":null},'
                '"correction":{"target":"name","overwrite":true},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.7,"spoken_response":"ご所属を教えていただけますか。"}'
            ),
            '{"name":null,"affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '名前が違います。島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.speech_act == 'inform'
    assert decision.extracted_name == '下中'
    assert decision.correction_target == 'none'


def test_supervisor_adapter_slot_rescue_fills_missing_affiliation_and_purpose():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":null,"affiliation":null,"purpose":null},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.6,"spoken_response":"ご用件を教えていただけますか。"}'
            ),
            '{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'


def test_supervisor_adapter_recovers_name_when_duplicate_slot_values_are_rejected():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":null,"purpose":"打ち合わせで参りました"},'
                '"correction":{"target":"name","overwrite":true},"confirmation":{"ready":false,"accepted":false},'
                '"ignore_input":false,"confidence":0.9,"spoken_response":"ご所属を教えていただけますか。"}'
            ),
            '{"name":"島中","affiliation":"島中","purpose":null}',
            '{"name":"島中","affiliation":null,"purpose":null}',
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation is None
    assert decision.missing_fields == ['affiliation', 'purpose']


def test_supervisor_adapter_slot_rescue_preserves_existing_purpose_when_affiliation_is_recovered():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":null,"affiliation":null,"purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.6,"spoken_response":"内容を確認します。"}'
            ),
            '{"name":null,"affiliation":"菅谷研究室","purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'


def test_supervisor_adapter_unusable_confirm_turn_recovers_affirmation():
    snapshot = _snapshot()
    snapshot.phase = 'confirming'
    snapshot.visitor_info.name = '島中'
    snapshot.visitor_info.affiliation = '菅谷研究室'
    snapshot.visitor_info.purpose = '学長に会いに来ました'
    snapshot.pending_confirmation = snapshot.visitor_info.copy()

    responses = iter(
        [
            '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
            '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
            (
                '{"speech_act":"affirm","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":true}}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        'はい',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.speech_act == 'affirm'
    assert decision.should_confirm is True
    assert decision.missing_fields == []
    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None


def test_supervisor_adapter_unusable_primary_falls_back_to_slot_recovery():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            '{"speech_act":"inform","slot_updates":{"name":"菅谷","affiliation":"情報科学研究室","purpose":"学長に会いに参りました"}',
            '```json {"speech_act":"inform","slot_updates":{"name":"島中","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}}',
            '{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'


def test_supervisor_adapter_unusable_primary_can_retry_missing_fields_individually():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            '{"speech_act":"inform","slot_updates":{"name":"菅谷","affiliation":"情報科学研究室","purpose":"学長に会いに参りました"}',
            '```json {"utterance":"菅谷研究室に所属しており、学長に会いに来ました"}',
            '{"name":null,"affiliation":null,"purpose":null}',
            '{"affiliation":"菅谷研究室","name":null,"purpose":null}',
            '{"purpose":"学長に会いに来ました","name":null,"affiliation":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.should_confirm is True


def test_supervisor_adapter_unusable_primary_refines_lossy_combined_slots_with_single_field_retry():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            '{"speech_act":"inform","slot_updates":{"name":"菅谷","affiliation":"情報科学研究室","purpose":"学長に会いに参りました"}',
            '```json {"utterance":"菅谷研究室に所属しており、学長に会いに来ました"}',
            '{"name":"菅谷","affiliation":"研究室","purpose":"会いに来ました"}',
            '{"affiliation":"菅谷研究室","name":null,"purpose":null}',
            '{"purpose":"学長に会いに来ました","name":null,"affiliation":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'


def test_supervisor_adapter_rejects_meeting_target_as_name_but_keeps_purpose():
    snapshot = _snapshot()
    snapshot.last_dialog_act = 'ask_name'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"学長","affiliation":null,'
                '"purpose":"学長に会いに来ました"},"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.95,"spoken_response":"お名前を伺ってもよろしいでしょうか。"}'
            ),
            '{"name":"学長","affiliation":null,"purpose":"学長に会いに来ました"}',
            '{"name":null,"affiliation":null,"purpose":"学長に会いに来ました"}',
            '{"value":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == ['name', 'affiliation']


def test_supervisor_adapter_ignores_polite_filler_even_when_model_marks_inform():
    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return (
            '{"speech_act":"inform","slot_updates":{"name":null,"affiliation":null,'
            '"purpose":"よろしくお願いします"},"correction":{"target":"none","overwrite":false},'
            '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
            '"confidence":0.6,"spoken_response":"お名前を伺ってもよろしいでしょうか。"}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        'えっと、よろしくお願いします。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.ignore_input is True
    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['name', 'affiliation', 'purpose']


def test_slot_rescue_stays_on_primary_field_for_affiliation_prompt():
    snapshot = _snapshot()
    snapshot.last_dialog_act = 'ask_affiliation'
    snapshot.visitor_info.name = '島中'

    fields = _target_fields_for_slot_rescue(
        snapshot,
        latest_utterance='菅谷研究室です。',
        speech_act='inform',
        slot_confidence=0.2,
        correction_target='none',
        ignore_input=False,
        extracted_name=None,
        extracted_affiliation=None,
        extracted_purpose=None,
    )

    assert fields == ['affiliation']


def test_supervisor_adapter_slot_rescue_can_clear_hallucinated_missing_slots():
    snapshot = _snapshot()

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"unknown","affiliation":"unknown","purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.5,"spoken_response":"お名前を教えてください。"}'
            ),
            '{"name":null,"affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        'こんにちは',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None


def test_supervisor_adapter_ignores_unrelated_name_overwrite_without_correction():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"菅谷研究室","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.95,"spoken_response":"内容を確認します。"}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'


def test_supervisor_adapter_purpose_first_does_not_commit_meeting_target_as_name():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":"学長","affiliation":null,"purpose":"学長に会いに来ました"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,"confidence":0.9}'
            ),
            '{"name":null,"affiliation":null,"purpose":"学長に会いに来ました"}',
            '{"value":null}',
            '{"value":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == ['name', 'affiliation']


def test_supervisor_adapter_name_turn_does_not_commit_affiliation_or_purpose():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":"島中","affiliation":"未知","purpose":"self-introduction"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,"confidence":0.8}'
            ),
            '{"name":"島中","affiliation":null,"purpose":null}',
            '{"value":"島中"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['affiliation', 'purpose']


def test_supervisor_adapter_ignores_filler_without_slot_recovery():
    responses = iter(
        [
            (
                '{"speech_act":"greeting","slot_candidates":{"name":null,"affiliation":null,"purpose":null},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":true,"confidence":0.7}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        'えっと、よろしくお願いします。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.ignore_input is True
    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None


def test_supervisor_adapter_confirming_new_information_is_not_forced_to_affirm():
    snapshot = _snapshot()
    snapshot.phase = 'confirming'
    snapshot.visitor_info.name = '山田'
    snapshot.visitor_info.affiliation = '総務部'
    snapshot.visitor_info.purpose = None
    snapshot.pending_confirmation = snapshot.visitor_info.copy()
    snapshot.last_dialog_act = 'confirm'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":null,"affiliation":null,"purpose":"書類提出です"},'
                '"correction_scope":"purpose","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,"confidence":0.95}'
            ),
            '{"name":null,"affiliation":null,"purpose":"書類提出です"}',
            '{"value":"書類提出です"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '書類提出です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.speech_act != 'affirm'
    assert decision.extracted_purpose == '書類提出です'


def test_supervisor_adapter_preserves_explicit_name_correction():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '下中'

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return (
            '{"speech_act":"correction","slot_updates":{"name":"島中","affiliation":null,"purpose":null},'
            '"correction":{"target":"name","overwrite":true},'
            '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
            '"confidence":0.98,"spoken_response":"失礼しました。"}'
        )

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '名前が違います。島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.correction_target == 'name'


def test_supervisor_adapter_ignores_fake_correction_flag_on_inform_turn():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"菅谷","affiliation":"研究室","purpose":"学長に会いに来ました"},'
                '"correction":{"target":"name","overwrite":true},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"確認します。"}'
            ),
            '{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.correction_target == 'none'


def test_supervisor_adapter_rescues_name_correction_from_deny_turn():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '下中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            '{"speech_act":"deny","next_dialog_act":"ask_name"}',
            '{"speech_act":"correction","slot_updates":{"name":"島中","affiliation":null,"purpose":null},"correction":{"target":"name","overwrite":true},"confirmation":{"ready":false,"accepted":false},"ignore_input":false,"confidence":0.9,"spoken_response":"失礼しました。"}',
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '名前が違います。島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.speech_act == 'correction'
    assert decision.extracted_name == '島中'
    assert decision.correction_target == 'name'


def test_supervisor_adapter_semantic_normalization_clears_name_only_purpose_pollution():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":null,"purpose":"打ち合わせで参りました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"ご所属を教えていただけますか。"}'
            ),
            '{"name":null,"affiliation":null,"purpose":"島中です"}',
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['affiliation', 'purpose']


def test_supervisor_adapter_semantic_normalization_preserves_composite_affiliation_and_purpose():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":null,"affiliation":"研究室","purpose":"会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.85,"spoken_response":"確認します。"}'
            ),
            '{"affiliation":"菅谷研究室","purpose":"学長に会いに来ました","name":null}',
            '{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'


def test_supervisor_adapter_confirmation_rescue_promotes_affirm():
    snapshot = _snapshot()
    snapshot.phase = 'confirming'
    snapshot.visitor_info.name = '島中'
    snapshot.visitor_info.affiliation = '菅谷研究室'
    snapshot.visitor_info.purpose = '学長に会いに来ました'
    snapshot.pending_confirmation = snapshot.visitor_info.copy()

    responses = iter(
        [
            (
                '{"speech_act":"unknown","slot_updates":{"name":null,"affiliation":null,"purpose":null},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.4,"spoken_response":"承知しました。"}'
            ),
            (
                '{"speech_act":"affirm","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":true}}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        'はい',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.speech_act == 'affirm'
    assert decision.should_confirm is True


def test_supervisor_adapter_purpose_first_turn_commits_only_purpose():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":"学長","affiliation":"大学","purpose":"学長に会いに来ました"},'
                '"slot_updates":{"name":"学長","affiliation":"大学","purpose":"学長に会いに来ました"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"お名前を教えてください。"}'
            ),
            '{"name":"学長","affiliation":"大学","purpose":"学長に会いに来ました"}',
            '{"name":null,"affiliation":null,"purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == ['name', 'affiliation']


def test_supervisor_adapter_purpose_first_accepts_minor_purpose_inflection_difference():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":null,"affiliation":null,"purpose":"学長に会いに来た"},'
                '"slot_updates":{"name":null,"affiliation":null,"purpose":"学長に会いに来た"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"お名前を教えてください。"}'
            ),
            '{"name":null}',
            '{"name":null,"affiliation":null,"purpose":"学長に会いに来た"}',
            '{"name":null,"affiliation":null,"purpose":"学長に会いに来た"}',
            '{"value":"学長に会いに来た"}',
            '{"purpose":"学長に会いに来ました"}',
            '{"purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == ['name', 'affiliation']


def test_supervisor_adapter_confirming_new_name_reopens_with_name_update():
    snapshot = _snapshot()
    snapshot.phase = 'confirming'
    snapshot.last_dialog_act = 'confirm'
    snapshot.visitor_info.name = '学長'
    snapshot.visitor_info.affiliation = '大学'
    snapshot.visitor_info.purpose = '学長に会いに来ました'
    snapshot.pending_confirmation = snapshot.visitor_info.copy()

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":"島中","affiliation":null,"purpose":null},'
                '"slot_updates":{"name":"島中","affiliation":null,"purpose":null},'
                '"correction_scope":"name","correction":{"target":"name","overwrite":true},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"失礼しました。"}'
            ),
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.correction_target == 'name'


def test_supervisor_adapter_rejects_hallucinated_purpose_from_name_only_turn():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"田中","affiliation":null,"purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.92,"spoken_response":"確認します。"}'
            ),
            '{"name":"田中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '私の名前は田中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '田中'
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['affiliation', 'purpose']


def test_supervisor_adapter_rejects_hallucinated_affiliation_not_in_utterance():
    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"田中","affiliation":"菅谷研究室","purpose":null},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"ご用件を教えていただけますか。"}'
            ),
            '{"name":"田中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        '私の名前は田中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '田中'
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['affiliation', 'purpose']


def test_supervisor_adapter_rejects_duplicate_name_that_matches_affiliation():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"菅谷研究室","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.92,"spoken_response":"確認します。"}'
            ),
            '{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'


def test_supervisor_adapter_commit_policy_drops_inferred_purpose_during_affiliation_turn():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":"菅谷研究室","purpose":"研究"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.9,"spoken_response":"確認します。"}'
            ),
            '{"name":"島中","affiliation":"菅谷研究室","purpose":"研究"}',
            '{"name":null,"affiliation":"菅谷研究室","purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name is None
    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['purpose']


def test_supervisor_adapter_commit_policy_keeps_explicit_secondary_slot_when_user_volunteers_it():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_updates":{"name":"島中","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":true,"accepted":false},"ignore_input":false,'
                '"confidence":0.95,"spoken_response":"確認します。"}'
            ),
            '{"name":"島中","affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
            '{"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室に所属しており、学長に会いに来ました。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose == '学長に会いに来ました'
    assert decision.missing_fields == []


def test_supervisor_adapter_greeting_turn_is_ignored_without_slot_rescue():
    responses = iter(
        [
            (
                '{"speech_act":"greeting","slot_candidates":{"name":"未知の先生","affiliation":"未知の研究所","purpose":"来訪理由不明"},'
                '"slot_updates":{"name":"未知の先生","affiliation":"未知の研究所","purpose":"来訪理由不明"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":true,'
                '"confidence":0.2,"spoken_response":"こんにちは。"}'
            ),
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        _snapshot(),
        'こんにちは。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.ignore_input is True
    assert decision.extracted_name is None
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None


def test_supervisor_adapter_name_turn_does_not_rescue_affiliation_or_purpose():
    snapshot = _snapshot()
    snapshot.last_dialog_act = 'ask_name'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":"島中","affiliation":"未知","purpose":"self-introduction"},'
                '"slot_updates":{"name":"島中","affiliation":"未知","purpose":"self-introduction"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.7,"spoken_response":"承知しました。"}'
            ),
            '{"name":"島中","affiliation":null,"purpose":null}',
            '{"name":"島中","affiliation":null,"purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '島中です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_name == '島中'
    assert decision.extracted_affiliation is None
    assert decision.extracted_purpose is None


def test_supervisor_adapter_affiliation_turn_does_not_commit_inferred_purpose():
    snapshot = _snapshot()
    snapshot.visitor_info.name = '島中'
    snapshot.last_dialog_act = 'ask_affiliation'

    responses = iter(
        [
            (
                '{"speech_act":"inform","slot_candidates":{"name":"島中","affiliation":"菅谷研究室","purpose":"研究"},'
                '"slot_updates":{"name":"島中","affiliation":"菅谷研究室","purpose":"研究"},'
                '"correction_scope":"none","correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},"ignore_input":false,'
                '"confidence":0.85,"spoken_response":"確認します。"}'
            ),
            '{"name":null,"affiliation":"菅谷研究室","purpose":null}',
            '{"name":null,"affiliation":"菅谷研究室","purpose":null}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = SupervisorAdapter(_invoke, temperature=0.0, max_tokens=96)
    decision = adapter.analyze(
        snapshot,
        '菅谷研究室です。',
        currently_speaking=False,
        captured_during_tts=False,
    )

    assert decision.extracted_affiliation == '菅谷研究室'
    assert decision.extracted_purpose is None
