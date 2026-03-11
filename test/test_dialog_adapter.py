from __future__ import annotations

from ros2_reception_orchestrator.dialog_adapter import DialogAdapter
from ros2_reception_orchestrator.state_models import DialogRenderRequest
from ros2_reception_orchestrator.state_models import VisitorInfo


def test_dialog_adapter_rejects_chitchat_for_ask_affiliation():
    responses = iter(
        [
            '{"spoken_response":"こんにちは、島中さん。お元気ですか？"}',
            '{"accept":false,"spoken_response":"島中さん、ご所属を教えていただけますか。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=2,
            dialog_act='ask_affiliation',
            phase='collecting',
            latest_utterance='島中です。',
            visitor_info=VisitorInfo(name='島中'),
        )
    )

    assert text == '島中さん、ご所属を教えていただけますか。'


def test_dialog_adapter_rejects_wrong_slot_question():
    responses = iter(
        [
            '{"spoken_response":"こんにちは、島中さん。お名前はどのようにお呼びいただいていますか？"}',
            '{"accept":false,"spoken_response":"島中さん、ご所属を教えていただけますか。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=3,
            dialog_act='ask_affiliation',
            phase='collecting',
            latest_utterance='はい。',
            visitor_info=VisitorInfo(name='島中'),
        )
    )

    assert text == '島中さん、ご所属を教えていただけますか。'


def test_dialog_adapter_rejects_ask_affiliation_with_gozonji_wording():
    responses = iter(
        [
            '{"spoken_response":"こんにちは、島中さん、ご所属をご存知でしょうか？"}',
            '{"accept":false,"spoken_response":"島中さん、ご所属を教えていただけますか。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=3,
            dialog_act='ask_affiliation',
            phase='collecting',
            latest_utterance='島中です。',
            visitor_info=VisitorInfo(name='島中'),
        )
    )

    assert text == '島中さん、ご所属を教えていただけますか。'


def test_dialog_adapter_rejects_waiting_state_restart():
    responses = iter(
        [
            '{"spoken_response":"恐れ入りますが、お名前を伺ってもよろしいでしょうか。"}',
            '{"accept":false,"spoken_response":"承知しました。担当者への連絡は継続しておりますので、少々お待ちください。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=5,
            dialog_act='acknowledge_waiting',
            phase='notified_waiting',
            latest_utterance='ここで待っていればいいですか。',
            visitor_info=VisitorInfo(name='島中', affiliation='研究室', purpose='学長に会いに来ました'),
        )
    )

    assert text == '承知しました。担当者への連絡は継続しておりますので、少々お待ちください。'


def test_dialog_adapter_accepts_valid_waiting_guidance():
    responses = iter(
        [
            '{"spoken_response":"はい、そのままそこでお待ちください。担当者へ連絡しております。"}',
            '{"accept":true,"spoken_response":"はい、そのままそこでお待ちください。担当者へ連絡しております。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=5,
            dialog_act='acknowledge_waiting',
            phase='notified_waiting',
            latest_utterance='ここで待っていればいいですか。',
            visitor_info=VisitorInfo(name='島中', affiliation='研究室', purpose='学長に会いに来ました'),
        )
    )

    assert text == 'はい、そのままそこでお待ちください。担当者へ連絡しております。'


def test_dialog_adapter_rejects_notify_waiting_that_repeats_purpose():
    responses = iter(
        [
            '{"spoken_response":"はい、学長に会いに来ました。担当者へ連絡しました。そのままお待ちください。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=6,
            dialog_act='notify_waiting',
            phase='notified_waiting',
            latest_utterance='はい',
            visitor_info=VisitorInfo(name='島中', affiliation='菅谷研究室', purpose='学長に会いに来ました'),
        )
    )

    assert text == '担当者へ連絡しました。少々お待ちください。'


def test_dialog_adapter_rejects_statement_only_purpose_echo_for_ask_name():
    responses = iter(
        [
            '{"spoken_response":"学長に会いに来ました。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=4,
            dialog_act='ask_name',
            phase='collecting',
            latest_utterance='学長に会いに来ました。',
            visitor_info=VisitorInfo(),
        )
    )

    assert text == '恐れ入ります。お名前を伺ってもよろしいでしょうか。'


def test_dialog_adapter_rejects_multi_slot_question_for_ask_name():
    responses = iter(
        [
            '{"spoken_response":"恐れ入りますが、お名前とご所属を伺ってもよろしいでしょうか。"}',
            '{"accept":false,"spoken_response":"恐れ入ります。お名前を伺ってもよろしいでしょうか。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=2,
            dialog_act='ask_name',
            phase='collecting',
            latest_utterance='こんにちは',
            visitor_info=VisitorInfo(),
        )
    )

    assert text == '恐れ入ります。お名前を伺ってもよろしいでしょうか。'


def test_dialog_adapter_rejects_polite_statement_that_does_not_request_information():
    responses = iter(
        [
            '{"spoken_response":"学長に会いに来られること、ご存じでしたら、お早めにお立ち寄りいただけますようお願い申し上げます。"}',
        ]
    )

    def _invoke(session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema):
        del session_id, user_message, system_prompt, temperature, max_tokens, stateless, response_json_schema
        return next(responses)

    adapter = DialogAdapter(_invoke, temperature=0.0, max_tokens=80)
    text = adapter.render(
        DialogRenderRequest(
            session_id='session-1',
            turn_id=3,
            dialog_act='ask_affiliation',
            phase='collecting',
            latest_utterance='菅谷研究室に所属しており、学長に会いに来ました。',
            visitor_info=VisitorInfo(name='島中'),
        )
    )

    assert text == '島中さん、ご所属を教えていただけますか。'
