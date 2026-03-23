from __future__ import annotations

from types import SimpleNamespace

from ros2_reception_orchestrator.semantic_extractor_server import SemanticExtractorServer


def test_usable_payload_requires_operations_and_detected_language() -> None:
    assert SemanticExtractorServer._usable_payload(
        {
            'speech_act': 'inform',
            'target_slot': 'name',
            'ambiguity': 'low',
            'requires_confirmation': False,
            'confidence': 0.8,
            'evidence': 'test',
            'grounded_segments': [],
        }
    ) is False


def test_heuristic_payload_marks_language_unknown_and_requests_clarification() -> None:
    payload = SemanticExtractorServer._heuristic_payload('こんにちは')

    assert payload['detected_language'] == 'unknown'
    assert payload['operations'][0]['op'] == 'request_clarification'


def test_build_prompt_includes_operation_few_shots() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='collecting',
        working_info=SimpleNamespace(name='', affiliation='', purpose=''),
        committed_info=SimpleNamespace(name='', affiliation='', purpose=''),
        focus_slot='affiliation',
        last_system_act='clarify_affiliation',
        pending_clarification_slot='affiliation',
        current_response_language='ja',
        turn=SimpleNamespace(text='あ、それ違いますね。それじゃなくて、えっと、菅屋研究室です。'),
    )

    prompt = server._build_prompt(req)

    assert 'confirm_working_state' in prompt
    assert 'replace_slot' in prompt
    assert 'Greeting-only utterances' not in prompt


def test_needs_semantic_rescue_for_missing_slot_operations() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='collecting',
        pending_clarification_slot='',
        focus_slot='name',
        working_info=SimpleNamespace(name='', affiliation='', purpose=''),
    )

    needed = server._needs_semantic_rescue(
        req,
        {
            'speech_act': 'inform',
            'target_slot': 'name',
            'operations': [],
        },
    )

    assert needed is True


def test_preferred_slot_uses_missing_field_when_target_absent() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        pending_clarification_slot='',
        focus_slot='none',
        working_info=SimpleNamespace(name='島中', affiliation='', purpose=''),
    )

    assert server._preferred_slot(req, {'target_slot': 'none'}) == 'affiliation'
