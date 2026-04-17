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


def test_build_structured_prompt_includes_function_schema_hint() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='collecting',
        working_info=SimpleNamespace(name='', affiliation='', purpose=''),
        committed_info=SimpleNamespace(name='', affiliation='', purpose=''),
        focus_slot='purpose',
        last_system_act='ask_purpose',
        pending_clarification_slot='',
        current_response_language='ja',
        turn=SimpleNamespace(text='打ち合わせです'),
    )

    prompt = server._build_structured_prompt(req)

    assert 'Function schema:' in prompt
    assert 'Return exactly one JSON object' in prompt


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


def test_sanitize_provider_defaults_unknown_to_chat_llm() -> None:
    assert SemanticExtractorServer._sanitize_provider('structured_extractor') == 'structured_extractor'
    assert SemanticExtractorServer._sanitize_provider('mystery_backend') == 'chat_llm'


def test_provider_diff_summary_flags_changed_slot() -> None:
    diff = SemanticExtractorServer._provider_diff_summary(
        {'speech_act': 'inform', 'target_slot': 'name', 'operations': [{'op': 'set_slot'}]},
        {'speech_act': 'inform', 'target_slot': 'purpose', 'operations': [{'op': 'set_slot'}]},
    )

    assert diff['speech_act_changed'] is False
    assert diff['target_slot_changed'] is True


def test_contextual_override_prefers_purpose_for_visit_intent() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='collecting',
        pending_clarification_slot='purpose',
        focus_slot='purpose',
        last_system_act='clarify_purpose',
        working_info=SimpleNamespace(name='アダチ', affiliation='科学研究室', purpose=''),
        turn=SimpleNamespace(text='中山さんに会いに来ました。'),
    )
    payload = {
        'speech_act': 'inform',
        'target_slot': 'name',
        'ambiguity': 'low',
        'requires_confirmation': False,
        'confidence': 0.94,
        'evidence': 'states a person name',
        'grounded_segments': ['中山'],
        'operations': [
            {
                'op': 'set_slot',
                'slot': 'name',
                'value': '中山',
                'grounded_text': '中山さんに会いに来ました。',
                'confidence': 0.94,
            }
        ],
    }

    adjusted = server._apply_contextual_overrides(req, payload)

    assert adjusted is not None
    assert adjusted['target_slot'] == 'purpose'
    assert adjusted['operations'][0]['slot'] == 'purpose'
    assert adjusted['operations'][0]['value'] == '中山さんに会いに来ました'


def test_contextual_override_retargets_name_misclassification_to_purpose() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='collecting',
        pending_clarification_slot='',
        focus_slot='name',
        last_system_act='ask_name',
        working_info=SimpleNamespace(name='', affiliation='', purpose=''),
        turn=SimpleNamespace(text='中山さんに会いに来ました。'),
    )
    payload = {
        'speech_act': 'inform',
        'target_slot': 'name',
        'ambiguity': 'low',
        'requires_confirmation': False,
        'confidence': 0.94,
        'evidence': 'states a person name',
        'grounded_segments': ['中山'],
        'operations': [
            {
                'op': 'set_slot',
                'slot': 'name',
                'value': '中山',
                'grounded_text': '中山さんに会いに来ました。',
                'confidence': 0.94,
            }
        ],
    }

    adjusted = server._apply_contextual_overrides(req, payload)

    assert adjusted is not None
    assert adjusted['speech_act'] == 'inform'
    assert adjusted['target_slot'] == 'purpose'
    assert adjusted['operations'][0]['slot'] == 'purpose'
    assert adjusted['operations'][0]['value'] == '中山さんに会いに来ました'


def test_contextual_override_rejects_incomplete_affiliation_fragment() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='collecting',
        pending_clarification_slot='affiliation',
        focus_slot='affiliation',
        last_system_act='ask_affiliation',
        working_info=SimpleNamespace(name='アダチ', affiliation='', purpose=''),
        turn=SimpleNamespace(text='書籍は。'),
    )
    payload = {
        'speech_act': 'inform',
        'target_slot': 'affiliation',
        'ambiguity': 'low',
        'requires_confirmation': False,
        'confidence': 0.95,
        'evidence': 'states an affiliation',
        'grounded_segments': ['書籍'],
        'operations': [
            {
                'op': 'set_slot',
                'slot': 'affiliation',
                'value': '書籍',
                'grounded_text': '書籍',
                'confidence': 0.95,
            }
        ],
    }

    adjusted = server._apply_contextual_overrides(req, payload)

    assert adjusted is not None
    assert adjusted['target_slot'] == 'affiliation'
    assert adjusted['operations'][0]['op'] == 'request_clarification'
    assert adjusted['operations'][0]['slot'] == 'affiliation'


def test_contextual_override_updates_purpose_during_confirming() -> None:
    server = SemanticExtractorServer.__new__(SemanticExtractorServer)
    req = SimpleNamespace(
        phase='confirming',
        pending_clarification_slot='',
        focus_slot='none',
        last_system_act='confirm_snapshot',
        working_info=SimpleNamespace(name='アダチ', affiliation='科学研究室', purpose='科学研究室'),
        turn=SimpleNamespace(text='中山さんに会いに来ました。'),
    )
    payload = {
        'speech_act': 'correction',
        'target_slot': 'purpose',
        'ambiguity': 'low',
        'requires_confirmation': True,
        'confidence': 0.94,
        'evidence': 'rejects the current confirmation',
        'grounded_segments': ['中山'],
        'operations': [
            {
                'op': 'reject_confirmation',
                'slot': 'purpose',
                'value': '',
                'grounded_text': '中山さんに会いに来ました。',
                'confidence': 0.94,
            }
        ],
    }

    adjusted = server._apply_contextual_overrides(req, payload)

    assert adjusted is not None
    assert adjusted['speech_act'] == 'correction'
    assert adjusted['target_slot'] == 'purpose'
    assert adjusted['operations'][0]['op'] == 'replace_slot'
    assert adjusted['operations'][0]['slot'] == 'purpose'
    assert adjusted['operations'][0]['value'] == '中山さんに会いに来ました'
