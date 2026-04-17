from __future__ import annotations

from types import SimpleNamespace

from ros2_reception_orchestrator.response_planner_server import _fallback_dialog_text
from ros2_reception_orchestrator.response_planner_server import _is_valid_rendered_response
from ros2_reception_orchestrator.response_planner_server import _sanitize_response_language


def test_fallback_dialog_text_uses_english_templates() -> None:
    assert _fallback_dialog_text('ask_name', '', '', '', 'en') == 'May I have your name, please?'
    assert _fallback_dialog_text('clarify_affiliation', '', '', '', 'en').startswith('I may have misheard your affiliation.')
    assert _fallback_dialog_text('notify_waiting', '', '', '', 'en') == (
        'I have notified the person in charge. Please wait for a moment.'
    )


def test_sanitize_response_language_defaults_to_japanese() -> None:
    assert _sanitize_response_language('en') == 'en'
    assert _sanitize_response_language('ja') == 'ja'
    assert _sanitize_response_language('unknown') == 'ja'


def test_is_valid_rendered_response_rejects_echo_for_ask_affiliation() -> None:
    req = SimpleNamespace(
        dialog_act='ask_affiliation',
        response_language='ja',
        latest_user_text='えっと名前はえっと島中雄大と言います。',
        working_info=SimpleNamespace(name='島中雄大', affiliation='', purpose=''),
    )

    assert not _is_valid_rendered_response(req, 'えっと名前はえっと島中雄大と言います。')


def test_is_valid_rendered_response_accepts_slot_question_for_ask_affiliation() -> None:
    req = SimpleNamespace(
        dialog_act='ask_affiliation',
        response_language='ja',
        latest_user_text='島中雄大と言います。',
        working_info=SimpleNamespace(name='島中雄大', affiliation='', purpose=''),
    )

    assert _is_valid_rendered_response(req, '島中雄大様、ご所属を教えてください。')


def test_is_valid_rendered_response_rejects_brief_ack_for_clarify_affiliation() -> None:
    req = SimpleNamespace(
        dialog_act='clarify_affiliation',
        response_language='ja',
        latest_user_text='はい。',
        working_info=SimpleNamespace(name='島中雄大', affiliation='', purpose=''),
    )

    assert not _is_valid_rendered_response(req, 'はい。')
