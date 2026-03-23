from __future__ import annotations

from ros2_reception_orchestrator.response_planner_server import _fallback_dialog_text
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
