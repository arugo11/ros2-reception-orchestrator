from __future__ import annotations

import json
import re
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from reception_interfaces.action import ExtractTurn
from reception_interfaces.msg import SemanticDecision
from reception_interfaces.msg import VisitorInfo
from ros2_vllm_interfaces.action import Chat

from .llm_stage_utils import extract_json_object
from .llm_stage_utils import invoke_chat_action


_STAGE1_SYSTEM_PROMPT = (
    'You are a strict receptionist semantic extractor. '
    'Return JSON only. Do not output prose. '
    'Infer speech_act and slot_updates from the latest utterance. '
    'Never fabricate names/affiliations/purposes if not present.'
)

_STAGE1_JSON_SCHEMA = json.dumps(
    {
        'type': 'object',
        'properties': {
            'speech_act': {
                'type': 'string',
                'enum': ['inform', 'affirm', 'deny', 'correction', 'question', 'complaint', 'greeting', 'unknown'],
            },
            'slot_updates': {
                'type': 'object',
                'properties': {
                    'name': {'type': ['string', 'null']},
                    'affiliation': {'type': ['string', 'null']},
                    'purpose': {'type': ['string', 'null']},
                },
                'required': ['name', 'affiliation', 'purpose'],
                'additionalProperties': False,
            },
            'correction_target': {
                'type': 'string',
                'enum': ['none', 'name', 'affiliation', 'purpose', 'all'],
            },
            'ignore_input': {'type': 'boolean'},
            'confidence': {'type': 'number'},
            'evidence': {'type': 'string'},
        },
        'required': ['speech_act', 'slot_updates', 'correction_target', 'ignore_input', 'confidence', 'evidence'],
        'additionalProperties': False,
    },
    ensure_ascii=False,
)


class SemanticExtractorServer(Node):
    def __init__(self) -> None:
        super().__init__('semantic_extractor_server')
        self._server_cb_group = ReentrantCallbackGroup()
        self._client_cb_group = ReentrantCallbackGroup()

        self.declare_parameter('llm.chat_action_name', '/llm/chat')
        self.declare_parameter('llm.temperature', 0.0)
        self.declare_parameter('llm.max_tokens', 180)
        self.declare_parameter('extract.action_name', '/reception/extract_turn')

        self._chat_action_name = str(self.get_parameter('llm.chat_action_name').value)
        self._temperature = float(self.get_parameter('llm.temperature').value)
        self._max_tokens = int(self.get_parameter('llm.max_tokens').value)
        self._extract_action_name = str(self.get_parameter('extract.action_name').value)

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

        prompt = self._build_prompt(req)
        raw = ''
        repaired = False
        payload: dict[str, Any] | None = None
        try:
            started = time.monotonic()
            raw = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.turn.session_id}:extract:{req.turn.turn_seq}',
                user_message=prompt,
                system_prompt=_STAGE1_SYSTEM_PROMPT,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stateless=True,
                response_json_schema=_STAGE1_JSON_SCHEMA,
                total_timeout_sec=20.0,
            )
            self.get_logger().info(
                f'stage1 primary completed turn={req.turn.turn_seq} '
                f'latency_ms={(time.monotonic() - started) * 1000.0:.1f} raw_len={len(raw)}'
            )
            payload = extract_json_object(raw)
            self.get_logger().info(
                f'stage1 primary parsed turn={req.turn.turn_seq} payload_ok={self._usable_payload(payload)}'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'stage1 primary failed: {exc}')

        if not self._usable_payload(payload):
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
                    session_id=f'{req.turn.session_id}:extract-repair:{req.turn.turn_seq}',
                    user_message=repair_prompt,
                    system_prompt='Return fixed JSON only.',
                    temperature=0.0,
                    max_tokens=min(self._max_tokens, 140),
                    stateless=True,
                    response_json_schema=_STAGE1_JSON_SCHEMA,
                    total_timeout_sec=12.0,
                )
                self.get_logger().info(
                    f'stage1 repair completed turn={req.turn.turn_seq} '
                    f'latency_ms={(time.monotonic() - started) * 1000.0:.1f} raw_len={len(raw)}'
                )
                payload = extract_json_object(raw)
                self.get_logger().info(
                    f'stage1 repair parsed turn={req.turn.turn_seq} payload_ok={self._usable_payload(payload)}'
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'stage1 repair failed: {exc}')

        if not self._usable_payload(payload):
            payload = self._heuristic_payload(req.turn.text)
            self.get_logger().warn(
                f'stage1 heuristic fallback used turn={req.turn.turn_seq} text={req.turn.text}'
            )

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
            f'current_name={req.visitor_info.name}\n'
            f'current_affiliation={req.visitor_info.affiliation}\n'
            f'current_purpose={req.visitor_info.purpose}\n'
            f'pending_name={req.pending_confirmation.name}\n'
            f'pending_affiliation={req.pending_confirmation.affiliation}\n'
            f'pending_purpose={req.pending_confirmation.purpose}\n'
            f'latest_utterance={req.turn.text}\n'
        )

    @staticmethod
    def _usable_payload(payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        required = {'speech_act', 'slot_updates', 'correction_target', 'ignore_input', 'confidence', 'evidence'}
        return required.issubset(payload.keys())

    @staticmethod
    def _to_decision(turn_seq: int, payload: dict[str, Any] | None) -> SemanticDecision:
        msg = SemanticDecision()
        msg.turn_seq = int(turn_seq)
        msg.speech_act = 'unknown'
        msg.correction_target = 'none'
        msg.ignore_input = False
        msg.confidence = 0.0
        msg.evidence = ''
        slot = VisitorInfo()
        slot.name = ''
        slot.affiliation = ''
        slot.purpose = ''

        if isinstance(payload, dict):
            msg.speech_act = str(payload.get('speech_act', 'unknown'))
            msg.correction_target = str(payload.get('correction_target', 'none'))
            msg.ignore_input = bool(payload.get('ignore_input', False))
            try:
                msg.confidence = float(payload.get('confidence', 0.0))
            except (TypeError, ValueError):
                msg.confidence = 0.0
            msg.evidence = str(payload.get('evidence', ''))
            updates = payload.get('slot_updates', {})
            if isinstance(updates, dict):
                slot.name = str(updates.get('name') or '').strip()
                slot.affiliation = str(updates.get('affiliation') or '').strip()
                slot.purpose = str(updates.get('purpose') or '').strip()

        msg.slot_patch = slot
        return msg

    @staticmethod
    def _heuristic_payload(text: str) -> dict[str, Any]:
        utterance = (text or '').strip()
        normalized = utterance.replace('　', ' ')
        lower = normalized.lower()
        segments = [s.strip() for s in re.split(r'[、。,.!?！？\n]+', normalized) if s.strip()]

        speech_act = 'inform'
        if not utterance:
            speech_act = 'unknown'
        elif any(token in lower for token in ('訂正', '違います', 'ではなく', 'じゃなく', '修正')):
            speech_act = 'correction'
        elif any(token in lower for token in ('こんにちは', 'こんばんは', 'おはよう', 'はじめまして')):
            speech_act = 'greeting'
        elif '?' in utterance or '？' in utterance:
            speech_act = 'question'

        updates: dict[str, str | None] = {'name': None, 'affiliation': None, 'purpose': None}
        affiliation_markers = (
            '株式会社',
            '有限会社',
            '合同会社',
            '大学',
            '研究所',
            '病院',
            '銀行',
            '商事',
            'コーポレーション',
        )
        purpose_markers = (
            '打ち合わせ',
            '面談',
            '訪問',
            '会議',
            '相談',
            '商談',
            '納品',
            '手続き',
            '説明',
            '挨拶',
        )

        name_patterns = (
            r'(?:私(?:の)?名前は|名前は|わたしは|私は)\s*([^\s、。,.]{1,20})\s*(?:です|と申します|といいます)?',
            r'(?:申します|といいます)\s*([^\s、。,.]{1,20})',
        )
        for pattern in name_patterns:
            match = re.search(pattern, normalized)
            if match:
                candidate = match.group(1).strip()
                if candidate and not any(m in candidate for m in affiliation_markers + purpose_markers):
                    updates['name'] = candidate
                    break

        aff_patterns = (
            r'(?:所属(?:は)?|会社(?:名)?(?:は)?|勤務先(?:は)?|学校(?:名)?(?:は)?)\s*([^\n、。]{1,40})',
        )
        for pattern in aff_patterns:
            match = re.search(pattern, normalized)
            if match:
                updates['affiliation'] = match.group(1).strip()
                break
        if not updates['affiliation']:
            for seg in segments:
                if any(marker in seg for marker in affiliation_markers):
                    updates['affiliation'] = seg
                    break

        purpose_patterns = (
            r'(?:用件(?:は)?|目的(?:は)?|本日(?:の)?(?:用件|目的)(?:は)?)\s*([^\n、。]{1,40})',
        )
        for pattern in purpose_patterns:
            match = re.search(pattern, normalized)
            if match:
                updates['purpose'] = match.group(1).strip()
                break
        if not updates['purpose']:
            for seg in segments:
                if any(marker in seg for marker in purpose_markers):
                    updates['purpose'] = seg
                    break

        correction_target = 'none'
        if speech_act == 'correction':
            has_name = any(token in lower for token in ('名前', '氏名'))
            has_aff = any(token in lower for token in ('所属', '会社', '勤務先', '学校'))
            has_purpose = any(token in lower for token in ('用件', '目的', '要件'))
            count = int(has_name) + int(has_aff) + int(has_purpose)
            if count >= 2:
                correction_target = 'all'
            elif has_name:
                correction_target = 'name'
            elif has_aff:
                correction_target = 'affiliation'
            elif has_purpose:
                correction_target = 'purpose'
            else:
                correction_target = 'all'

        extracted_count = sum(1 for value in updates.values() if value)
        if extracted_count > 0:
            confidence = 0.86
        elif speech_act == 'greeting':
            confidence = 0.80
        elif speech_act == 'correction':
            confidence = 0.72
        elif speech_act == 'question':
            confidence = 0.55
        else:
            confidence = 0.40

        return {
            'speech_act': speech_act,
            'slot_updates': {
                'name': updates['name'],
                'affiliation': updates['affiliation'],
                'purpose': updates['purpose'],
            },
            'correction_target': correction_target,
            'ignore_input': False,
            'confidence': confidence,
            'evidence': 'heuristic_fallback',
        }


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
