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
    assert decision.missing_fields == []


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
    assert decision.extracted_purpose is None
    assert decision.missing_fields == ['name', 'affiliation', 'purpose']


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
