from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

from ros2_reception_orchestrator.node_v2 import ReceptionOrchestratorNodeV2
from ros2_reception_orchestrator.node_v2 import _QueueVisitorEvent
from ros2_reception_orchestrator.llm_stage_utils import extract_json_object
from ros2_reception_orchestrator.v2_types import BeliefOperationData
from ros2_reception_orchestrator.v2_types import SemanticDecisionData
from ros2_reception_orchestrator.v2_types import VisitorInfoData
from ros2_reception_orchestrator.v2_types import SessionStateData
from ros2_reception_orchestrator.v2_types import TurnEnvelopeData


def test_normalized_decision_preserves_detected_language_and_operations() -> None:
    decision = SemanticDecisionData(
        turn_seq=1,
        speech_act='greeting',
        detected_language='en',
        target_slot='none',
        ambiguity='medium',
        confidence=0.8,
        evidence='test',
        operations=[BeliefOperationData(op='ignore', slot='none', confidence=0.1)],
    )

    normalized = ReceptionOrchestratorNodeV2._normalize_semantic_decision(decision)

    assert normalized.speech_act == 'greeting'
    assert normalized.detected_language == 'en'
    assert normalized.operations[0].op == 'ignore'


def test_normalized_decision_infers_english_from_latin_utterance() -> None:
    decision = SemanticDecisionData(
        turn_seq=1,
        speech_act='greeting',
        detected_language='unknown',
        target_slot='none',
        ambiguity='medium',
        confidence=0.8,
        evidence='test',
        operations=[BeliefOperationData(op='ignore', slot='none', confidence=0.1)],
    )

    normalized = ReceptionOrchestratorNodeV2._normalize_semantic_decision(
        decision,
        utterance_text='Hello there.',
    )

    assert normalized.detected_language == 'en'


def test_normalized_decision_overrides_wrong_japanese_label_for_english_utterance() -> None:
    decision = SemanticDecisionData(
        turn_seq=2,
        speech_act='inform',
        detected_language='ja',
        target_slot='name',
        ambiguity='low',
        confidence=0.95,
        evidence='test',
        operations=[BeliefOperationData(op='set_slot', slot='name', value='Yuda', confidence=0.95)],
    )

    normalized = ReceptionOrchestratorNodeV2._normalize_semantic_decision(
        decision,
        utterance_text='My name is Yuda.',
    )

    assert normalized.detected_language == 'en'


def test_stage1_language_hint_uses_unknown_before_first_turn() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SimpleNamespace(latest_applied_turn=0, response_language='ja')
    )

    assert node._stage1_language_hint() == 'unknown'


def test_stage1_language_hint_preserves_detected_language_after_progress() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SimpleNamespace(latest_applied_turn=1, response_language='en')
    )

    assert node._stage1_language_hint() == 'en'


def test_session_has_user_activity_ignores_fresh_blank_session() -> None:
    state = SessionStateData(session_id='s')

    assert ReceptionOrchestratorNodeV2._session_has_user_activity(state) is False


def test_handle_session_inactivity_resets_active_session_after_timeout() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    expired_state = SessionStateData(
        session_id='expired-session',
        latest_applied_turn=2,
        last_activity_at=datetime.now(tz=UTC) - timedelta(seconds=31),
    )
    node._reducer = SimpleNamespace(
        state=expired_state,
        reset_called=False,
    )

    def _reset() -> None:
        node._reducer.reset_called = True
        node._reducer.state = SessionStateData(session_id='fresh-session')

    node._reducer.reset = _reset
    node._session_inactivity_reset_sec = 30
    node._pending_turn_events = ['pending']
    node._ingestor = SimpleNamespace(reset_called=False)
    node._ingestor.reset = lambda: setattr(node._ingestor, 'reset_called', True)
    node._effect_executor = SimpleNamespace(calls=[])
    node._effect_executor.cancel_pending_tts = lambda **kwargs: node._effect_executor.calls.append(kwargs)
    node._conversation_log_writer = SimpleNamespace(flush_all_called=False)
    node._conversation_log_writer.flush_all = lambda: setattr(
        node._conversation_log_writer, 'flush_all_called', True
    )
    node._publish_conversation_trace_calls = []
    node._publish_conversation_trace = lambda **kwargs: node._publish_conversation_trace_calls.append(kwargs)
    node._reception_active = True
    node._pending_visitor_trigger = object()
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)

    node._handle_session_inactivity()

    assert node._reducer.reset_called is True
    assert node._ingestor.reset_called is True
    assert node._conversation_log_writer.flush_all_called is True
    assert node._pending_turn_events == []
    assert node._effect_executor.calls == [{'detail': 'session_timeout_pending_tts'}]
    assert node._reception_active is False
    assert node._pending_visitor_trigger is None
    assert len(node._publish_conversation_trace_calls) == 1
    assert node._publish_conversation_trace_calls[0]['session_id'] == 'expired-session'
    assert node._publish_conversation_trace_calls[0]['event_type'] == 'SESSION_TIMEOUT'


def test_process_visitor_event_activates_reception_when_ready() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reception_active = False
    node._pending_visitor_trigger = None
    node._dependencies_ready = lambda: True
    node._publish_trace_event_calls = []
    node._publish_trace_event = lambda trace, turn_seq: node._publish_trace_event_calls.append((trace, turn_seq))
    node._submit_visitor_greeting_called = False
    node._submit_visitor_greeting = lambda: setattr(node, '_submit_visitor_greeting_called', True)
    node._publish_state_called = False
    node._publish_state = lambda: setattr(node, '_publish_state_called', True)
    node._reducer = SimpleNamespace(state=SessionStateData(session_id='session-1'))
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)

    node._process_visitor_event(_QueueVisitorEvent(event_type='VISITOR_TRIGGERED', confidence=0.9, detail=''))

    assert node._reception_active is True
    assert node._submit_visitor_greeting_called is True
    assert node._publish_state_called is True
    assert node._publish_trace_event_calls[0][0].event_type == 'SESSION_STARTED'


def test_process_visitor_event_defers_until_ready() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reception_active = False
    node._pending_visitor_trigger = None
    node._dependencies_ready = lambda: False
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)

    event = _QueueVisitorEvent(event_type='VISITOR_TRIGGERED', confidence=0.75, detail='warmup')
    node._process_visitor_event(event)

    assert node._pending_visitor_trigger is event


def test_extract_json_object_accepts_fenced_json() -> None:
    payload = extract_json_object(
        '```json\n{"speech_act":"affirm","operations":[]}\n```'
    )

    assert payload == {'speech_act': 'affirm', 'operations': []}


def test_decision_needs_stage_rescue_for_inform_without_operations() -> None:
    decision = SemanticDecisionData(
        turn_seq=1,
        speech_act='inform',
        target_slot='name',
        ambiguity='low',
        confidence=1.0,
        operations=[],
    )

    assert ReceptionOrchestratorNodeV2._decision_needs_stage_rescue(decision) is True


def test_decision_does_not_need_stage_rescue_when_slot_operation_exists() -> None:
    decision = SemanticDecisionData(
        turn_seq=1,
        speech_act='inform',
        target_slot='name',
        ambiguity='low',
        confidence=1.0,
        operations=[BeliefOperationData(op='set_slot', slot='name', value='島中', confidence=1.0)],
    )

    assert ReceptionOrchestratorNodeV2._decision_needs_stage_rescue(decision) is False


def test_decision_needs_stage_rescue_for_inform_without_substantive_op_even_with_none_slot() -> None:
    decision = SemanticDecisionData(
        turn_seq=1,
        speech_act='inform',
        target_slot='none',
        ambiguity='low',
        confidence=1.0,
        operations=[BeliefOperationData(op='replace_slot', slot='none', value='yes', confidence=1.0)],
    )

    assert ReceptionOrchestratorNodeV2._decision_needs_stage_rescue(decision) is True


def test_refine_long_slot_decision_adds_missing_slot_from_same_utterance() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(
            session_id='session-1',
            phase='collecting',
            focus_slot='name',
            working_info=VisitorInfoData(),
        )
    )
    def _fake_invoke_chat_action(**kwargs: str) -> str:
        session_id = kwargs['session_id']
        if session_id.endswith(':extract-long-slot-refine:2'):
            return json.dumps(
                {
                    'name': '島中',
                    'affiliation': '柴原工業大学',
                    'purpose': None,
                },
                ensure_ascii=False,
            )
        if session_id.endswith(':slot-commit:2'):
            return json.dumps(
                {
                    'name': '島中',
                    'affiliation': '柴原工業大学',
                    'purpose': None,
                },
                ensure_ascii=False,
            )
        if session_id.endswith(':field-commit:2:name'):
            return '{"value":"島中"}'
        if session_id.endswith(':field-commit:2:affiliation'):
            return '{"value":"柴原工業大学"}'
        if session_id.endswith(':field-commit:2:purpose'):
            return '{"value":null}'
        raise AssertionError(f'unexpected session id: {session_id}')

    node._invoke_chat_action = _fake_invoke_chat_action

    decision = SemanticDecisionData(
        turn_seq=2,
        speech_act='inform',
        detected_language='ja',
        target_slot='affiliation',
        ambiguity='low',
        confidence=0.94,
        evidence='test',
        operations=[
            BeliefOperationData(
                op='set_slot',
                slot='affiliation',
                value='柴原工業大学',
                grounded_text='柴原工業大学',
                confidence=0.94,
            )
        ],
        grounded_segments=['柴原工業大学'],
    )
    turn = TurnEnvelopeData(
        session_id='session-1',
        turn_seq=2,
        utterance_id='utt-1',
        text='島中です。いや、だから芝原工業大学です。',
        captured_during_tts=False,
        asr_confidence=0.98,
    )

    refined = node._refine_long_slot_decision(turn, decision)

    slots = {(operation.slot, operation.value) for operation in refined.operations}
    assert ('name', '島中') in slots
    assert ('affiliation', '柴原工業大学') in slots


def test_call_render_stage_uses_render_action_result(monkeypatch) -> None:
    class _FakeResult:
        def __init__(self) -> None:
            self.text = '承知しました。ご所属をもう一度お願いいたします。'
            self.used_fallback = False

    class _FakeWrapped:
        def __init__(self) -> None:
            self.result = _FakeResult()

    class _FakeGoalHandle:
        def __init__(self) -> None:
            self.accepted = True

        def get_result_async(self):
            return _FakeWrapped()

    class _FakeRenderClient:
        def __init__(self) -> None:
            self.goal = None

        def wait_for_server(self, timeout_sec):
            del timeout_sec
            return True

        def send_goal_async(self, goal):
            self.goal = goal
            return _FakeGoalHandle()

    monkeypatch.setattr('ros2_reception_orchestrator.node_v2.wait_future', lambda future, timeout_sec: future)

    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._render_action_name = '/reception/render_dialog'
    node._render_client = _FakeRenderClient()
    node._session_transcript = []
    node._publish_execution_event = lambda *args, **kwargs: None
    node._short = lambda text: text
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None)
    node._reducer = SimpleNamespace(
        state=SessionStateData(
            session_id='session-2',
            phase='collecting',
            response_language='ja',
            focus_slot='affiliation',
            pending_clarification_slot='affiliation',
            working_info=VisitorInfoData(name='島中'),
            committed_info=VisitorInfoData(),
        )
    )

    text = node._call_render_stage(
        turn_seq=5,
        dialog_act='clarify_affiliation',
        latest_user_text='いや、だから芝原工業大学です。',
        secretary_reply_text='',
    )

    assert text == '承知しました。ご所属をもう一度お願いいたします。'
    assert node._render_client.goal.dialog_act == 'clarify_affiliation'


def test_refine_long_slot_decision_can_retarget_single_slot_extraction() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(session_id='s', working_info=VisitorInfoData(name='島中', affiliation='須江野県', purpose=''))
    )
    def _fake_invoke_chat_action(**kwargs: str) -> str:
        session_id = kwargs['session_id']
        if session_id.endswith(':extract-long-slot-refine:4'):
            return '{"name":null,"affiliation":"菅屋研究室","purpose":null}'
        if session_id.endswith(':slot-commit:4'):
            return '{"name":null,"affiliation":"菅屋研究室","purpose":null}'
        if session_id.endswith(':field-commit:4:affiliation'):
            return '{"value":"菅屋研究室"}'
        raise AssertionError(f'unexpected session id: {session_id}')

    node._invoke_chat_action = _fake_invoke_chat_action

    decision = SemanticDecisionData(
        turn_seq=4,
        speech_act='inform',
        target_slot='purpose',
        confidence=1.0,
        operations=[BeliefOperationData(op='replace_slot', slot='purpose', value='菅屋研究室です', confidence=1.0)],
    )
    turn = TurnEnvelopeData(session_id='s', turn_seq=4, utterance_id='u', text='あ、それ違いますね。それじゃなくて、えっと、菅屋研究室です', captured_during_tts=False, asr_confidence=1.0)

    refined = node._refine_long_slot_decision(turn, decision)

    assert refined.target_slot == 'affiliation'
    assert refined.operations[0].slot == 'affiliation'
    assert refined.operations[0].value == '菅屋研究室'


def test_refine_long_slot_decision_falls_back_to_per_slot_recovery() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(session_id='s', working_info=VisitorInfoData(name='島中', affiliation='須江野県', purpose=''))
    )

    def fake_invoke_chat_action(**kwargs: str) -> str:
        session_id = kwargs['session_id']
        if session_id.endswith(':extract-long-slot-refine:4'):
            return '{"name":"島中","affiliation":"菅屋研究室","purpose":"研究室訪問"}'
        if session_id.endswith(':slot-commit:4'):
            return '{"name":null,"affiliation":"菅屋研究室","purpose":null}'
        if session_id.endswith(':field-commit:4:name'):
            return '{"value":null}'
        if session_id.endswith(':field-commit:4:affiliation'):
            return '{"value":"菅屋研究室"}'
        if session_id.endswith(':field-commit:4:purpose'):
            return '{"value":null}'
        raise AssertionError(f'unexpected session id: {session_id}')

    node._invoke_chat_action = fake_invoke_chat_action

    decision = SemanticDecisionData(
        turn_seq=4,
        speech_act='inform',
        target_slot='purpose',
        confidence=1.0,
        operations=[BeliefOperationData(op='replace_slot', slot='purpose', value='菅屋研究室です', confidence=1.0)],
    )
    turn = TurnEnvelopeData(session_id='s', turn_seq=4, utterance_id='u', text='あ、それ違いますね。それじゃなくて、えっと、菅屋研究室です', captured_during_tts=False, asr_confidence=1.0)

    refined = node._refine_long_slot_decision(turn, decision)

    assert refined.target_slot == 'affiliation'


def test_commit_extracted_slot_candidates_uses_transcript_aware_slot_commit() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(
            session_id='session-3',
            phase='collecting',
            focus_slot='name',
            last_system_act='ask_name',
            working_info=VisitorInfoData(),
        )
    )

    def fake_invoke_chat_action(**kwargs: str) -> str:
        session_id = kwargs['session_id']
        if session_id.endswith(':slot-commit:2'):
            return '{"name":"島中","affiliation":"芝原工業大学","purpose":null}'
        if session_id.endswith(':field-commit:2:name'):
            return '{"value":"島中"}'
        if session_id.endswith(':field-commit:2:affiliation'):
            return '{"value":"芝原工業大学"}'
        raise AssertionError(f'unexpected session id: {session_id}')

    node._invoke_chat_action = fake_invoke_chat_action
    turn = TurnEnvelopeData(
        session_id='session-3',
        turn_seq=2,
        utterance_id='utt-2',
        text='島中です。いや、だから芝原工業大学です。',
        captured_during_tts=False,
        asr_confidence=0.98,
    )

    committed = node._commit_extracted_slot_candidates(
        turn,
        {'name': '島中', 'affiliation': '芝原工業大学'},
    )

    assert committed == {'name': '島中', 'affiliation': '芝原工業大学'}


def test_normalize_slot_operation_values_removes_fillers_from_affiliation_and_purpose() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(session_id='s', working_info=VisitorInfoData(name='しまなか', affiliation='', purpose=''))
    )
    node._invoke_chat_action = lambda **kwargs: '{"name":null,"affiliation":"芝浦工業大学","purpose":"鈴木さんに会いに来ました。"}'

    decision = SemanticDecisionData(
        turn_seq=4,
        speech_act='inform',
        target_slot='purpose',
        confidence=1.0,
        operations=[
            BeliefOperationData(op='set_slot', slot='affiliation', value='えっと芝浦工業大学', grounded_text='所属はえっと芝浦工業大学です。', confidence=1.0),
            BeliefOperationData(op='replace_slot', slot='purpose', value='えっと鈴木さんに会いに来ました。', grounded_text='要件はえっと鈴木さんに会いに来ました。', confidence=1.0),
        ],
    )
    turn = TurnEnvelopeData(session_id='s', turn_seq=4, utterance_id='u', text='所属はえっと芝浦工業大学です。要件はえっと鈴木さんに会いに来ました。', captured_during_tts=False, asr_confidence=1.0)

    normalized = node._normalize_slot_operation_values(turn, decision)

    assert normalized.operations[0].value == '芝浦工業大学'
    assert normalized.operations[1].value == '鈴木さんに会いに来ました'


def test_normalize_slot_operation_values_removes_fillers_from_name() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(session_id='s', working_info=VisitorInfoData(name='', affiliation='', purpose=''))
    )
    node._invoke_chat_action = lambda **kwargs: '{"name":"アダチ","affiliation":null,"purpose":null}'

    decision = SemanticDecisionData(
        turn_seq=2,
        speech_act='inform',
        target_slot='name',
        confidence=1.0,
        operations=[
            BeliefOperationData(
                op='replace_slot',
                slot='name',
                value='えっとアダチです',
                grounded_text='えっと名前はえっとアダチです。',
                confidence=1.0,
            ),
        ],
    )
    turn = TurnEnvelopeData(
        session_id='s',
        turn_seq=2,
        utterance_id='u',
        text='えっと名前はえっとアダチです。',
        captured_during_tts=False,
        asr_confidence=1.0,
    )

    normalized = node._normalize_slot_operation_values(turn, decision)

    assert normalized.operations[0].value == 'アダチ'


def test_apply_confirmation_rescue_overrides_broken_confirming_slot_update() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._reducer = SimpleNamespace(
        state=SessionStateData(
            session_id='s',
            phase='confirming',
            working_info=VisitorInfoData(name='アダチ', affiliation='スガヤ研究室', purpose='学長に会いに来ました'),
        )
    )
    node._rescue_direct_semantic_decision = lambda turn, decision: SemanticDecisionData(
        turn_seq=turn.turn_seq,
        speech_act='affirm',
        target_slot='none',
        ambiguity='low',
        confidence=0.95,
        evidence='test_confirmation_rescue',
        operations=[BeliefOperationData(op='confirm_working_state', slot='none', grounded_text=turn.text, confidence=0.95)],
    )

    decision = SemanticDecisionData(
        turn_seq=5,
        speech_act='inform',
        target_slot='name',
        ambiguity='low',
        confidence=0.95,
        operations=[BeliefOperationData(op='replace_slot', slot='name', value='ya', grounded_text='iya', confidence=0.95)],
    )
    turn = TurnEnvelopeData(
        session_id='s',
        turn_seq=5,
        utterance_id='u',
        text='うん お k です。',
        captured_during_tts=False,
        asr_confidence=1.0,
    )

    rescued = node._apply_confirmation_rescue_if_needed(turn, decision)

    assert rescued.speech_act == 'affirm'
    assert rescued.operations[0].op == 'confirm_working_state'


def test_call_extract_stage_prefers_stage_action_result() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._publish_execution_event = lambda *args, **kwargs: None
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None)
    node._short = lambda value: value
    node._apply_confirmation_rescue_if_needed = lambda turn, decision: decision
    node._decision_needs_stage_rescue = lambda decision: False
    node._refine_long_slot_decision = lambda turn, decision: decision
    node._normalize_slot_operation_values = lambda turn, decision: decision
    node._normalize_semantic_decision = lambda decision, **kwargs: decision
    node._call_extract_stage_action = lambda turn: SemanticDecisionData(
        turn_seq=turn.turn_seq,
        speech_act='inform',
        target_slot='name',
        ambiguity='low',
        confidence=0.9,
        evidence='stage',
        operations=[BeliefOperationData(op='set_slot', slot='name', value='佐藤', confidence=0.9)],
    )
    node._call_extract_direct_llm = lambda turn: (_ for _ in ()).throw(
        AssertionError('direct path should not be used')
    )

    turn = TurnEnvelopeData(
        session_id='s',
        turn_seq=1,
        utterance_id='u',
        text='私の名前は佐藤です。',
        captured_during_tts=False,
        asr_confidence=1.0,
    )

    decision = node._call_extract_stage(turn)

    assert decision.evidence == 'stage'
    assert decision.operations[0].value == '佐藤'


def test_call_extract_stage_falls_back_to_direct_llm_when_stage_action_fails() -> None:
    node = ReceptionOrchestratorNodeV2.__new__(ReceptionOrchestratorNodeV2)
    node._publish_execution_event = lambda *args, **kwargs: None
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None)
    node._short = lambda value: value
    node._apply_confirmation_rescue_if_needed = lambda turn, decision: decision
    node._decision_needs_stage_rescue = lambda decision: False
    node._refine_long_slot_decision = lambda turn, decision: decision
    node._normalize_slot_operation_values = lambda turn, decision: decision
    node._normalize_semantic_decision = lambda decision, **kwargs: decision
    node._call_extract_stage_action = lambda turn: (_ for _ in ()).throw(RuntimeError('stage unavailable'))
    node._call_extract_direct_llm = lambda turn: SemanticDecisionData(
        turn_seq=turn.turn_seq,
        speech_act='inform',
        target_slot='affiliation',
        ambiguity='low',
        confidence=0.8,
        evidence='direct',
        operations=[BeliefOperationData(op='set_slot', slot='affiliation', value='東京大学', confidence=0.8)],
    )

    turn = TurnEnvelopeData(
        session_id='s',
        turn_seq=2,
        utterance_id='u',
        text='所属は東京大学です。',
        captured_during_tts=False,
        asr_confidence=1.0,
    )

    decision = node._call_extract_stage(turn)

    assert decision.evidence == 'direct'
    assert decision.operations[0].slot == 'affiliation'
