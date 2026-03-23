from __future__ import annotations

import json
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from reception_interfaces.action import ExtractTurn
from reception_interfaces.msg import BeliefOperation
from reception_interfaces.msg import SemanticDecision
from ros2_vllm_interfaces.action import Chat

from .llm_stage_utils import extract_json_object
from .llm_stage_utils import invoke_chat_action
from .prompt_templates import RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA
from .prompt_templates import RECEPTION_REPAIR_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_SLOT_EXTRACT_JSON_SCHEMA
from .prompt_templates import build_reception_confirmation_rescue_prompt
from .prompt_templates import build_reception_slot_extract_prompt
from .state_models import SessionSnapshot
from .state_models import VisitorInfo


_OPERATION_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'op': {
            'type': 'string',
            'enum': [
                'set_slot',
                'replace_slot',
                'clear_slot',
                'confirm_working_state',
                'reject_confirmation',
                'request_clarification',
                'ignore',
            ],
        },
        'slot': {
            'type': 'string',
            'enum': ['name', 'affiliation', 'purpose', 'none'],
        },
        'value': {'type': ['string', 'null']},
        'grounded_text': {'type': ['string', 'null']},
        'confidence': {'type': 'number'},
    },
    'required': ['op', 'slot', 'value', 'grounded_text', 'confidence'],
    'additionalProperties': False,
}

_STAGE1_SYSTEM_PROMPT = (
    'You are a strict multilingual receptionist semantic extractor and belief-operation planner. '
    'Return JSON only. Do not output prose. '
    'Your job is to decide how the latest utterance should edit the receptionist belief state. '
    'Never emit free-form slot dumps when a narrower edit is sufficient. '
    'Use operations as the primary output. '
    'Prefer the smallest safe edit that is grounded in the utterance and state context. '
    'Do not overwrite unrelated slots just because one candidate phrase could fit multiple slots. '
    'If the user is correcting or revising previous information, prefer replace_slot on the most plausible target slot. '
    'If the utterance is ambiguous, incomplete, or low-grounding, emit request_clarification rather than guessing. '
    'detected_language means the language the receptionist should use to reply: ja, en, or unknown. '
    'current_response_language is only a weak hint. Always prioritize the latest utterance. '
    'Use ja for mainly Japanese utterances, en for mainly English utterances, and unknown when it is too ambiguous to judge. '
    'Do not infer language from acronyms or proper nouns alone. '
    'target_slot should identify the slot most relevant to the utterance when possible. '
    'ambiguity must be low, medium, or high. '
    'requires_confirmation should be true when the safest next step is a slot-specific clarification or explicit confirmation. '
    'Never fabricate names, affiliations, or purposes if the utterance does not support them. '
    'Greeting-only utterances should usually produce ignore, not slot writes. '
    'In confirming phase, clear acceptance should usually produce confirm_working_state, not a fake slot edit. '
    'When the state focus is a specific slot and the user gives a natural correction, update only that slot unless the utterance clearly supplies other slots too. '
    'If the utterance clearly provides exactly one slot value, operations must include one grounded set_slot or replace_slot for that slot instead of an empty list.'
)

_STAGE1_JSON_SCHEMA = json.dumps(
    {
        'type': 'object',
        'properties': {
            'speech_act': {
                'type': 'string',
                'enum': ['inform', 'affirm', 'deny', 'correction', 'question', 'complaint', 'greeting', 'unknown'],
            },
            'detected_language': {
                'type': 'string',
                'enum': ['ja', 'en', 'unknown'],
            },
            'target_slot': {
                'type': 'string',
                'enum': ['name', 'affiliation', 'purpose', 'none'],
            },
            'ambiguity': {
                'type': 'string',
                'enum': ['low', 'medium', 'high'],
            },
            'requires_confirmation': {'type': 'boolean'},
            'confidence': {'type': 'number'},
            'evidence': {'type': 'string'},
            'grounded_segments': {
                'type': 'array',
                'items': {'type': 'string'},
            },
            'operations': {
                'type': 'array',
                'items': _OPERATION_SCHEMA,
            },
        },
        'required': [
            'speech_act',
            'detected_language',
            'target_slot',
            'ambiguity',
            'requires_confirmation',
            'confidence',
            'evidence',
            'grounded_segments',
            'operations',
        ],
        'additionalProperties': False,
    },
    ensure_ascii=False,
)

_SUPPORTED_EXTRACT_PROVIDERS = {'chat_llm', 'structured_extractor'}

_STRUCTURED_PROVIDER_SYSTEM_PROMPT = (
    _STAGE1_SYSTEM_PROMPT
    + ' Treat the result as a structured extraction function call. '
    + 'Prefer a single grounded operation over broad interpretation. '
    + 'Use grounded_segments to justify every slot operation.'
)


class SemanticExtractorServer(Node):
    def __init__(self) -> None:
        super().__init__('semantic_extractor_server')
        self._server_cb_group = ReentrantCallbackGroup()
        self._client_cb_group = ReentrantCallbackGroup()

        self.declare_parameter('llm.chat_action_name', '/llm/chat')
        self.declare_parameter('llm.temperature', 0.0)
        self.declare_parameter('llm.max_tokens', 220)
        self.declare_parameter('extract.action_name', '/reception/extract_turn')
        self.declare_parameter('extract.provider', 'chat_llm')
        self.declare_parameter('extract.shadow_enabled', False)
        self.declare_parameter('extract.shadow_provider', '')

        self._chat_action_name = str(self.get_parameter('llm.chat_action_name').value)
        self._temperature = float(self.get_parameter('llm.temperature').value)
        self._max_tokens = int(self.get_parameter('llm.max_tokens').value)
        self._extract_action_name = str(self.get_parameter('extract.action_name').value)
        self._extract_provider = self._sanitize_provider(
            self.get_parameter('extract.provider').value
        )
        self._shadow_enabled = bool(self.get_parameter('extract.shadow_enabled').value)
        self._shadow_provider = self._sanitize_provider(
            self.get_parameter('extract.shadow_provider').value
        )

        self._chat_client = ActionClient(
            self,
            Chat,
            self._chat_action_name,
            callback_group=self._client_cb_group,
        )
        self._server = ActionServer(
            self,
            ExtractTurn,
            self._extract_action_name,
            self._execute,
            callback_group=self._server_cb_group,
        )

    def _execute(self, goal_handle: Any) -> ExtractTurn.Result:
        req = goal_handle.request
        result = ExtractTurn.Result()

        payload, raw, repaired = self._run_provider(req, self._extract_provider)

        if not self._usable_payload(payload):
            payload = self._heuristic_payload(req.turn.text)
            self.get_logger().warn(
                f'stage1 heuristic fallback used turn={req.turn.turn_seq} text={req.turn.text}'
            )
        elif self._needs_semantic_rescue(req, payload):
            rescued_payload = self._semantic_rescue_payload(req, payload)
            if self._usable_payload(rescued_payload):
                payload = rescued_payload
        payload = self._apply_contextual_overrides(req, payload)

        self._maybe_run_shadow_provider(req, payload)

        decision = self._to_decision(req.turn.turn_seq, payload)
        result.decision = decision
        result.repaired = repaired
        result.raw_response = raw
        goal_handle.succeed()
        return result

    def _build_prompt(self, req: ExtractTurn.Goal) -> str:
        return (
            'Task: semantic extraction for reception flow.\n'
            f'phase={req.phase}\n'
            f'working_name={req.working_info.name}\n'
            f'working_affiliation={req.working_info.affiliation}\n'
            f'working_purpose={req.working_info.purpose}\n'
            f'committed_name={req.committed_info.name}\n'
            f'committed_affiliation={req.committed_info.affiliation}\n'
            f'committed_purpose={req.committed_info.purpose}\n'
            f'focus_slot={req.focus_slot}\n'
            f'last_system_act={req.last_system_act}\n'
            f'pending_clarification_slot={req.pending_clarification_slot}\n'
            f'current_response_language={req.current_response_language}\n'
            f'latest_utterance={req.turn.text}\n'
            'Examples:\n'
            '- context: phase=collecting focus_slot=name latest_utterance=こんにちは。\n'
            '  output: {"speech_act":"greeting","detected_language":"ja","target_slot":"none","ambiguity":"low","requires_confirmation":false,"confidence":0.92,"evidence":"greeting only","grounded_segments":["こんにちは。"],"operations":[{"op":"ignore","slot":"none","value":null,"grounded_text":"こんにちは。","confidence":0.92}]}\n'
            '- context: phase=collecting focus_slot=name latest_utterance=私の名前は島中です。\n'
            '  output: {"speech_act":"inform","detected_language":"ja","target_slot":"name","ambiguity":"low","requires_confirmation":false,"confidence":0.93,"evidence":"states the visitor name","grounded_segments":["島中"],"operations":[{"op":"set_slot","slot":"name","value":"島中","grounded_text":"島中","confidence":0.93}]}\n'
            '- context: phase=collecting focus_slot=affiliation latest_utterance=所属は菅屋研究室です。\n'
            '  output: {"speech_act":"inform","detected_language":"ja","target_slot":"affiliation","ambiguity":"low","requires_confirmation":false,"confidence":0.92,"evidence":"states the affiliation","grounded_segments":["菅屋研究室"],"operations":[{"op":"set_slot","slot":"affiliation","value":"菅屋研究室","grounded_text":"菅屋研究室","confidence":0.92}]}\n'
            '- context: phase=collecting focus_slot=purpose latest_utterance=用件は打ち合わせです。\n'
            '  output: {"speech_act":"inform","detected_language":"ja","target_slot":"purpose","ambiguity":"low","requires_confirmation":false,"confidence":0.91,"evidence":"states the visit purpose","grounded_segments":["打ち合わせ"],"operations":[{"op":"set_slot","slot":"purpose","value":"打ち合わせ","grounded_text":"打ち合わせ","confidence":0.91}]}\n'
            '- context: phase=confirming latest_utterance=はい。\n'
            '  output: {"speech_act":"affirm","detected_language":"ja","target_slot":"none","ambiguity":"low","requires_confirmation":false,"confidence":0.95,"evidence":"accepts current snapshot","grounded_segments":["はい。"],"operations":[{"op":"confirm_working_state","slot":"none","value":null,"grounded_text":"はい。","confidence":0.95}]}\n'
            '- context: phase=collecting focus_slot=affiliation last_system_act=clarify_affiliation latest_utterance=あ、それ違いますね。それじゃなくて、えっと、菅屋研究室です。\n'
            '  output: {"speech_act":"correction","detected_language":"ja","target_slot":"affiliation","ambiguity":"medium","requires_confirmation":false,"confidence":0.82,"evidence":"revises the affiliation answer","grounded_segments":["菅屋研究室"],"operations":[{"op":"replace_slot","slot":"affiliation","value":"菅屋研究室","grounded_text":"菅屋研究室","confidence":0.82}]}\n'
            '- context: phase=collecting focus_slot=purpose latest_utterance=I have a meeting.\n'
            '  output: {"speech_act":"inform","detected_language":"en","target_slot":"purpose","ambiguity":"low","requires_confirmation":false,"confidence":0.85,"evidence":"states visit purpose","grounded_segments":["I have a meeting"],"operations":[{"op":"set_slot","slot":"purpose","value":"meeting","grounded_text":"I have a meeting","confidence":0.85}]}\n'
        )

    def _build_structured_prompt(self, req: ExtractTurn.Goal) -> str:
        return (
            self._build_prompt(req)
            + '\nFunction schema:\n'
            + '{"speech_act":"...","detected_language":"ja|en|unknown","target_slot":"name|affiliation|purpose|none","ambiguity":"low|medium|high","requires_confirmation":true|false,"confidence":0.0,"evidence":"...","grounded_segments":["..."],"operations":[{"op":"set_slot|replace_slot|clear_slot|confirm_working_state|reject_confirmation|request_clarification|ignore","slot":"name|affiliation|purpose|none","value":"...","grounded_text":"...","confidence":0.0}]}\n'
            + 'Return exactly one JSON object matching the function schema.'
        )

    def _run_provider(
        self,
        req: ExtractTurn.Goal,
        provider_name: str,
    ) -> tuple[dict[str, Any] | None, str, bool]:
        provider = self._sanitize_provider(provider_name)
        if provider == 'structured_extractor':
            return self._extract_with_chat_provider(
                req,
                provider='structured_extractor',
                prompt=self._build_structured_prompt(req),
                system_prompt=_STRUCTURED_PROVIDER_SYSTEM_PROMPT,
            )
        return self._extract_with_chat_provider(
            req,
            provider='chat_llm',
            prompt=self._build_prompt(req),
            system_prompt=_STAGE1_SYSTEM_PROMPT,
        )

    def _extract_with_chat_provider(
        self,
        req: ExtractTurn.Goal,
        *,
        provider: str,
        prompt: str,
        system_prompt: str,
    ) -> tuple[dict[str, Any] | None, str, bool]:
        raw = ''
        repaired = False
        payload: dict[str, Any] | None = None
        try:
            started = time.monotonic()
            raw = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.turn.session_id}:extract:{provider}:{req.turn.turn_seq}',
                user_message=prompt,
                system_prompt=system_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stateless=True,
                response_json_schema=_STAGE1_JSON_SCHEMA,
                total_timeout_sec=20.0,
            )
            self.get_logger().info(
                f'stage1 provider={provider} completed turn={req.turn.turn_seq} '
                f'latency_ms={(time.monotonic() - started) * 1000.0:.1f} raw_len={len(raw)}'
            )
            payload = extract_json_object(raw)
            self.get_logger().info(
                f'stage1 provider={provider} parsed turn={req.turn.turn_seq} '
                f'payload_ok={self._usable_payload(payload)}'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'stage1 provider={provider} failed: {exc}')

        if self._usable_payload(payload):
            return payload, raw, repaired

        repaired = True
        repair_prompt = (
            'Fix the following output into valid JSON matching the schema.\n'
            f'raw={raw}\n'
            f'utterance={req.turn.text}\n'
        )
        try:
            started = time.monotonic()
            raw = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.turn.session_id}:extract-repair:{provider}:{req.turn.turn_seq}',
                user_message=repair_prompt,
                system_prompt='Return fixed JSON only.',
                temperature=0.0,
                max_tokens=min(self._max_tokens, 180),
                stateless=True,
                response_json_schema=_STAGE1_JSON_SCHEMA,
                total_timeout_sec=12.0,
            )
            self.get_logger().info(
                f'stage1 repair provider={provider} completed turn={req.turn.turn_seq} '
                f'latency_ms={(time.monotonic() - started) * 1000.0:.1f} raw_len={len(raw)}'
            )
            payload = extract_json_object(raw)
            self.get_logger().info(
                f'stage1 repair provider={provider} parsed turn={req.turn.turn_seq} '
                f'payload_ok={self._usable_payload(payload)}'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'stage1 repair provider={provider} failed: {exc}')
        return payload, raw, repaired

    def _maybe_run_shadow_provider(
        self,
        req: ExtractTurn.Goal,
        primary_payload: dict[str, Any] | None,
    ) -> None:
        if not self._shadow_enabled:
            return
        shadow_provider = self._sanitize_provider(self._shadow_provider)
        if shadow_provider == self._extract_provider:
            return
        try:
            shadow_payload, _, _ = self._run_provider(req, shadow_provider)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'shadow extractor provider={shadow_provider} failed turn={req.turn.turn_seq}: {exc}'
            )
            return
        diff = self._provider_diff_summary(primary_payload, shadow_payload)
        self.get_logger().info(
            f'shadow extractor turn={req.turn.turn_seq} primary={self._extract_provider} '
            f'shadow={shadow_provider} diff={json.dumps(diff, ensure_ascii=False)}'
        )

    @staticmethod
    def _usable_payload(payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        required = {
            'speech_act',
            'detected_language',
            'target_slot',
            'ambiguity',
            'requires_confirmation',
            'confidence',
            'evidence',
            'grounded_segments',
            'operations',
        }
        return required.issubset(payload.keys())

    @staticmethod
    def _sanitize_provider(value: object) -> str:
        candidate = str(value or '').strip().lower()
        if candidate in _SUPPORTED_EXTRACT_PROVIDERS:
            return candidate
        return 'chat_llm'

    @staticmethod
    def _provider_diff_summary(
        primary_payload: dict[str, Any] | None,
        shadow_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        primary = primary_payload or {}
        shadow = shadow_payload or {}
        primary_ops = primary.get('operations', []) if isinstance(primary, dict) else []
        shadow_ops = shadow.get('operations', []) if isinstance(shadow, dict) else []
        return {
            'speech_act_changed': primary.get('speech_act') != shadow.get('speech_act'),
            'target_slot_changed': primary.get('target_slot') != shadow.get('target_slot'),
            'operation_count_changed': len(primary_ops) != len(shadow_ops),
            'primary_ops': primary_ops,
            'shadow_ops': shadow_ops,
        }

    def _needs_semantic_rescue(self, req: ExtractTurn.Goal, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        operations = payload.get('operations', [])
        has_substantive_op = False
        if isinstance(operations, list):
            for raw_operation in operations:
                if not isinstance(raw_operation, dict):
                    continue
                op = str(raw_operation.get('op', '')).strip()
                slot = self._sanitize_slot(raw_operation.get('slot', 'none'))
                value = str(raw_operation.get('value') or '').strip()
                if op in {'set_slot', 'replace_slot'} and slot in {'name', 'affiliation', 'purpose'} and value:
                    has_substantive_op = True
                    break
                if op in {'confirm_working_state', 'reject_confirmation'}:
                    has_substantive_op = True
                    break
        if has_substantive_op:
            return False
        speech_act = str(payload.get('speech_act', 'unknown')).strip()
        target_slot = self._sanitize_slot(payload.get('target_slot', 'none'))
        if req.phase == 'confirming':
            return True
        return speech_act in {'inform', 'correction'} and target_slot in {'name', 'affiliation', 'purpose'}

    def _semantic_rescue_payload(self, req: ExtractTurn.Goal, payload: dict[str, Any]) -> dict[str, Any]:
        if req.phase == 'confirming':
            rescued = self._rescue_confirmation_payload(req, payload)
            if rescued is not None:
                return rescued
        rescued = self._rescue_slot_operation_payload(req, payload)
        return rescued or payload

    def _rescue_confirmation_payload(
        self,
        req: ExtractTurn.Goal,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        snapshot = self._legacy_snapshot(req)
        try:
            raw = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.turn.session_id}:extract-confirm-rescue:{req.turn.turn_seq}',
                user_message=build_reception_confirmation_rescue_prompt(snapshot, req.turn.text),
                system_prompt=RECEPTION_REPAIR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=min(self._max_tokens, 96),
                stateless=True,
                response_json_schema=RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA,
                total_timeout_sec=12.0,
            )
            rescue_payload = extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'stage1 confirmation rescue failed: {exc}')
            return None
        if not isinstance(rescue_payload, dict):
            return None
        speech_act = str(rescue_payload.get('speech_act', payload.get('speech_act', 'unknown'))).strip()
        correction = rescue_payload.get('correction', {})
        target = self._sanitize_slot(correction.get('target', 'none')) if isinstance(correction, dict) else 'none'
        if speech_act == 'affirm':
            payload['speech_act'] = 'affirm'
            payload['target_slot'] = 'none'
            payload['ambiguity'] = 'low'
            payload['requires_confirmation'] = False
            payload['evidence'] = str(payload.get('evidence') or 'confirmation rescue')
            payload['operations'] = [
                {
                    'op': 'confirm_working_state',
                    'slot': 'none',
                    'value': None,
                    'grounded_text': req.turn.text,
                    'confidence': float(payload.get('confidence', 0.0) or 0.0),
                }
            ]
            return payload
        if speech_act in {'deny', 'correction'}:
            payload['speech_act'] = speech_act
            payload['target_slot'] = target
            payload['requires_confirmation'] = True
            payload['operations'] = [
                {
                    'op': 'reject_confirmation',
                    'slot': target if target in {'name', 'affiliation', 'purpose'} else self._preferred_slot(req, payload),
                    'value': None,
                    'grounded_text': req.turn.text,
                    'confidence': float(payload.get('confidence', 0.0) or 0.0),
                }
            ]
            return payload
        return None

    def _rescue_slot_operation_payload(
        self,
        req: ExtractTurn.Goal,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        target_slot = self._preferred_slot(req, payload)
        if target_slot not in {'name', 'affiliation', 'purpose'}:
            return None
        snapshot = self._legacy_snapshot(req)
        try:
            raw = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.turn.session_id}:extract-slot-rescue:{req.turn.turn_seq}',
                user_message=build_reception_slot_extract_prompt(snapshot, req.turn.text, target_fields=[target_slot]),
                system_prompt=RECEPTION_REPAIR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=min(self._max_tokens, 96),
                stateless=True,
                response_json_schema=RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
                total_timeout_sec=12.0,
            )
            rescue_payload = extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'stage1 slot rescue failed: {exc}')
            return None
        if not isinstance(rescue_payload, dict):
            return None
        rescued_value = str(rescue_payload.get(target_slot) or '').strip()
        if not rescued_value:
            return None
        current_value = getattr(req.working_info, target_slot)
        operation_name = 'replace_slot' if current_value else 'set_slot'
        payload['target_slot'] = target_slot
        payload['ambiguity'] = 'low' if str(payload.get('ambiguity', 'medium')) == 'high' else payload.get('ambiguity', 'medium')
        payload['requires_confirmation'] = False
        payload['grounded_segments'] = [rescued_value]
        payload['operations'] = [
            {
                'op': operation_name,
                'slot': target_slot,
                'value': rescued_value,
                'grounded_text': rescued_value,
                'confidence': float(payload.get('confidence', 0.0) or 0.0),
            }
        ]
        return payload

    def _legacy_snapshot(self, req: ExtractTurn.Goal) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=req.turn.session_id,
            phase=req.phase,
            visitor_info=VisitorInfo(
                name=req.working_info.name or None,
                affiliation=req.working_info.affiliation or None,
                purpose=req.working_info.purpose or None,
            ),
            last_user_utterance=req.turn.text,
            last_dialog_act=req.last_system_act or None,
            last_spoken_text='',
            pending_confirmation=(
                VisitorInfo(
                    name=req.committed_info.name or None,
                    affiliation=req.committed_info.affiliation or None,
                    purpose=req.committed_info.purpose or None,
                )
                if req.phase == 'confirming'
                else None
            ),
            latest_turn_id=int(req.turn.turn_seq),
        )

    def _preferred_slot(self, req: ExtractTurn.Goal, payload: dict[str, Any]) -> str:
        pending = self._sanitize_slot(req.pending_clarification_slot)
        if pending in {'name', 'affiliation', 'purpose'}:
            return pending
        target = self._sanitize_slot(payload.get('target_slot', 'none'))
        if target in {'name', 'affiliation', 'purpose'}:
            return target
        focus = self._sanitize_slot(req.focus_slot)
        if focus in {'name', 'affiliation', 'purpose'}:
            return focus
        if not req.working_info.name:
            return 'name'
        if not req.working_info.affiliation:
            return 'affiliation'
        if not req.working_info.purpose:
            return 'purpose'
        return 'none'

    def _apply_contextual_overrides(
        self,
        req: ExtractTurn.Goal,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return payload

        utterance = str(req.turn.text or '').strip()
        expected_slot = self._preferred_slot(req, payload)

        if _should_override_to_purpose(req=req, expected_slot=expected_slot, utterance=utterance):
            payload = dict(payload)
            payload['speech_act'] = 'correction' if req.phase == 'confirming' else 'inform'
            payload['target_slot'] = 'purpose'
            payload['ambiguity'] = 'low'
            payload['requires_confirmation'] = req.phase == 'confirming'
            purpose_value = _extract_purpose_value(utterance)
            operation_name = 'replace_slot' if str(req.working_info.purpose or '').strip() else 'set_slot'
            payload['grounded_segments'] = [purpose_value]
            payload['evidence'] = (
                str(payload.get('evidence') or '').strip()
                + ' | contextual_override:purpose_intent'
            ).strip(' |')
            payload['operations'] = [
                {
                    'op': operation_name,
                    'slot': 'purpose',
                    'value': purpose_value,
                    'grounded_text': utterance,
                    'confidence': float(payload.get('confidence', 0.0) or 0.0),
                }
            ]
            return payload

        if expected_slot == 'affiliation':
            if _looks_like_incomplete_affiliation_fragment(utterance):
                payload = dict(payload)
                payload['speech_act'] = 'unknown'
                payload['target_slot'] = 'affiliation'
                payload['ambiguity'] = 'high'
                payload['requires_confirmation'] = True
                payload['grounded_segments'] = []
                payload['evidence'] = (
                    str(payload.get('evidence') or '').strip()
                    + ' | contextual_override:incomplete_affiliation_fragment'
                ).strip(' |')
                payload['operations'] = [
                    {
                        'op': 'request_clarification',
                        'slot': 'affiliation',
                        'value': None,
                        'grounded_text': utterance,
                        'confidence': float(payload.get('confidence', 0.0) or 0.0),
                    }
                ]
                return payload

            if (
                _looks_like_affiliation_utterance(utterance)
                and self._sanitize_slot(payload.get('target_slot', 'none')) != 'affiliation'
            ):
                affiliation_value = _extract_affiliation_value(utterance)
                if affiliation_value:
                    payload = dict(payload)
                    payload['speech_act'] = 'inform'
                    payload['target_slot'] = 'affiliation'
                    payload['ambiguity'] = 'low'
                    payload['requires_confirmation'] = False
                    payload['grounded_segments'] = [affiliation_value]
                    payload['evidence'] = (
                        str(payload.get('evidence') or '').strip()
                        + ' | contextual_override:affiliation_context'
                    ).strip(' |')
                    payload['operations'] = [
                        {
                            'op': 'replace_slot' if str(req.working_info.affiliation or '').strip() else 'set_slot',
                            'slot': 'affiliation',
                            'value': affiliation_value,
                            'grounded_text': utterance,
                            'confidence': float(payload.get('confidence', 0.0) or 0.0),
                        }
                    ]
                    return payload

        return payload

    @staticmethod
    def _to_decision(turn_seq: int, payload: dict[str, Any] | None) -> SemanticDecision:
        msg = SemanticDecision()
        msg.turn_seq = int(turn_seq)
        msg.speech_act = 'unknown'
        msg.detected_language = 'unknown'
        msg.target_slot = 'none'
        msg.ambiguity = 'high'
        msg.requires_confirmation = False
        msg.confidence = 0.0
        msg.evidence = ''
        msg.operations = []
        msg.grounded_segments = []

        if isinstance(payload, dict):
            msg.speech_act = str(payload.get('speech_act', 'unknown'))
            msg.detected_language = SemanticExtractorServer._sanitize_detected_language(
                payload.get('detected_language', 'unknown')
            )
            msg.target_slot = SemanticExtractorServer._sanitize_slot(payload.get('target_slot', 'none'))
            msg.ambiguity = SemanticExtractorServer._sanitize_ambiguity(payload.get('ambiguity', 'high'))
            msg.requires_confirmation = bool(payload.get('requires_confirmation', False))
            try:
                msg.confidence = float(payload.get('confidence', 0.0))
            except (TypeError, ValueError):
                msg.confidence = 0.0
            msg.evidence = str(payload.get('evidence', ''))
            msg.grounded_segments = [str(item).strip() for item in payload.get('grounded_segments', []) if str(item).strip()]
            raw_operations = payload.get('operations', [])
            if isinstance(raw_operations, list):
                for raw_operation in raw_operations:
                    if not isinstance(raw_operation, dict):
                        continue
                    op = BeliefOperation()
                    op.op = str(raw_operation.get('op', 'ignore')).strip()
                    op.slot = SemanticExtractorServer._sanitize_slot(raw_operation.get('slot', 'none'))
                    op.value = str(raw_operation.get('value') or '').strip()
                    op.grounded_text = str(raw_operation.get('grounded_text') or '').strip()
                    try:
                        op.confidence = float(raw_operation.get('confidence', 0.0))
                    except (TypeError, ValueError):
                        op.confidence = 0.0
                    msg.operations.append(op)

        return msg

    @staticmethod
    def _sanitize_detected_language(value: object) -> str:
        candidate = str(value or 'unknown').strip().lower()
        if candidate in {'ja', 'en'}:
            return candidate
        return 'unknown'

    @staticmethod
    def _sanitize_slot(value: object) -> str:
        candidate = str(value or 'none').strip().lower()
        if candidate in {'name', 'affiliation', 'purpose'}:
            return candidate
        return 'none'

    @staticmethod
    def _sanitize_ambiguity(value: object) -> str:
        candidate = str(value or 'high').strip().lower()
        if candidate in {'low', 'medium', 'high'}:
            return candidate
        return 'high'

    @staticmethod
    def _heuristic_payload(text: str) -> dict[str, Any]:
        utterance = str(text or '').strip()
        if not utterance:
            return {
                'speech_act': 'unknown',
                'detected_language': 'unknown',
                'target_slot': 'none',
                'ambiguity': 'high',
                'requires_confirmation': False,
                'confidence': 0.0,
                'evidence': 'semantic_extractor_heuristic_empty',
                'grounded_segments': [],
                'operations': [
                    {
                        'op': 'ignore',
                        'slot': 'none',
                        'value': None,
                        'grounded_text': None,
                        'confidence': 0.0,
                    }
                ],
            }
        return {
            'speech_act': 'unknown',
            'detected_language': 'unknown',
            'target_slot': 'none',
            'ambiguity': 'high',
            'requires_confirmation': True,
            'confidence': 0.0,
            'evidence': 'semantic_extractor_heuristic_request_clarification',
            'grounded_segments': [],
            'operations': [
                {
                    'op': 'request_clarification',
                    'slot': 'none',
                    'value': None,
                    'grounded_text': None,
                    'confidence': 0.0,
                }
            ],
        }


_VISIT_PURPOSE_MARKERS = (
    '会いに来',
    '会いにき',
    'お会いし',
    '面会',
    '打ち合わせ',
    'ミーティング',
    '相談',
    'お願い',
    '訪問',
    '伺い',
    '来ました',
    '来た',
    '要件',
    '用件',
)

_IDENTITY_MARKERS = (
    '名前は',
    '申します',
    'といいます',
    '所属は',
)

_AFFILIATION_MARKERS = (
    '研究室',
    '大学',
    '会社',
    '学部',
    '学科',
    '学院',
    'センター',
    '株式会社',
    '有限会社',
    '高校',
    '病院',
)


def _looks_like_visit_purpose_utterance(text: str) -> bool:
    utterance = str(text or '').strip()
    return bool(utterance) and any(marker in utterance for marker in _VISIT_PURPOSE_MARKERS)


def _looks_like_explicit_identity_utterance(text: str) -> bool:
    utterance = str(text or '').strip()
    return bool(utterance) and any(marker in utterance for marker in _IDENTITY_MARKERS)


def _extract_purpose_value(text: str) -> str:
    candidate = str(text or '').strip()
    for prefix in ('本日の要件は', '本日の用件は', '要件は', '用件は'):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
            break
    while candidate and candidate[-1] in '。.!?！？':
        candidate = candidate[:-1].rstrip()
    return candidate or str(text or '').strip()


def _looks_like_incomplete_affiliation_fragment(text: str) -> bool:
    utterance = str(text or '').strip()
    if not utterance:
        return False
    trimmed = utterance.rstrip('。.!?！？')
    if any(marker in trimmed for marker in _AFFILIATION_MARKERS):
        return False
    return len(trimmed) <= 4 and trimmed.endswith(('は', 'が', 'を', 'の'))


def _looks_like_affiliation_utterance(text: str) -> bool:
    utterance = str(text or '').strip()
    return bool(utterance) and any(marker in utterance for marker in _AFFILIATION_MARKERS)


def _extract_affiliation_value(text: str) -> str:
    candidate = str(text or '').strip()
    for prefix in ('所属は', '所属です', '所属', 'ご所属は'):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
            break
    for suffix in ('です', 'になります'):
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)].rstrip()
    while candidate and candidate[-1] in '。.!?！？':
        candidate = candidate[:-1].rstrip()
    return candidate


def _should_override_to_purpose(
    *,
    req: ExtractTurn.Goal,
    expected_slot: str,
    utterance: str,
) -> bool:
    if not utterance:
        return False
    if _looks_like_explicit_identity_utterance(utterance):
        return False
    if not _looks_like_visit_purpose_utterance(utterance):
        return False
    if req.phase == 'confirming':
        return True
    return expected_slot == 'purpose'


def main() -> None:
    rclpy.init()
    node = SemanticExtractorServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
