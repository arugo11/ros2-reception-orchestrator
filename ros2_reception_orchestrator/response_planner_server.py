from __future__ import annotations

import re
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from reception_interfaces.action import RenderDialog
from ros2_vllm_interfaces.action import Chat

from .conversation_context import clone_chat_messages
from .llm_stage_utils import invoke_chat_action


def _sanitize_response_language(value: object) -> str:
    candidate = str(value or 'ja').strip().lower()
    if candidate == 'en':
        return 'en'
    return 'ja'


def _stage2_system_prompt(response_language: str) -> str:
    if _sanitize_response_language(response_language) == 'en':
        return (
            'You are a polite university receptionist assistant. '
            'Generate one concise spoken response in English. '
            'No JSON. No markdown. Keep to 1-2 sentences.'
        )
    return (
        'You are a polite Japanese university receptionist assistant. '
        'Generate one concise spoken response in Japanese. '
        'No JSON. No markdown. Keep to 1-2 sentences.'
    )


def _normalize_text(value: object) -> str:
    return ' '.join(str(value or '').split()).strip().lower()


def _looks_like_question(text: str) -> bool:
    compact = ' '.join(text.split()).strip()
    if not compact:
        return False
    if '?' in compact or '？' in compact:
        return True
    question_markers = (
        '教えて',
        '伺',
        'いただけます',
        'よろしいでしょうか',
        'please',
        'could you',
        'may i',
        'what brings you',
    )
    lowered = compact.lower()
    return any(marker in lowered for marker in question_markers)


def _contains_expected_slot_marker(dialog_act: str, text: str, response_language: str) -> bool:
    lowered = _normalize_text(text)
    language = _sanitize_response_language(response_language)
    markers: tuple[str, ...]
    if language == 'en':
        marker_map = {
            'ask_name': ('name',),
            'clarify_name': ('name',),
            'ask_affiliation': ('affiliation', 'organization', 'company', 'school', 'department', 'lab'),
            'clarify_affiliation': ('affiliation', 'organization', 'company', 'school', 'department', 'lab'),
            'ask_purpose': ('purpose', 'reason', 'what brings you here'),
            'clarify_purpose': ('purpose', 'reason', 'what brings you here'),
        }
        markers = marker_map.get(dialog_act, ())
    else:
        marker_map = {
            'ask_name': ('名前', 'お名前'),
            'clarify_name': ('名前', 'お名前'),
            'ask_affiliation': ('所属', 'ご所属'),
            'clarify_affiliation': ('所属', 'ご所属'),
            'ask_purpose': ('用件', 'ご用件', '目的'),
            'clarify_purpose': ('用件', 'ご用件', '目的'),
        }
        markers = marker_map.get(dialog_act, ())
    return any(marker.lower() in lowered for marker in markers)


def _is_valid_rendered_response(req: Any, text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if len(normalized) > 200:
        return False
    if normalized == _normalize_text(getattr(req, 'latest_user_text', '')):
        return False

    dialog_act = str(getattr(req, 'dialog_act', '') or '')
    response_language = str(getattr(req, 'response_language', 'ja') or 'ja')

    if dialog_act in {
        'ask_name',
        'ask_affiliation',
        'ask_purpose',
        'clarify_name',
        'clarify_affiliation',
        'clarify_purpose',
    }:
        return _looks_like_question(text) and _contains_expected_slot_marker(dialog_act, text, response_language)

    if dialog_act == 'confirm_snapshot':
        working = getattr(req, 'working_info', None)
        values = [
            _normalize_text(getattr(working, 'name', '')),
            _normalize_text(getattr(working, 'affiliation', '')),
            _normalize_text(getattr(working, 'purpose', '')),
        ]
        return _looks_like_question(text) and any(value and value in normalized for value in values)

    if dialog_act in {'notify_waiting', 'acknowledge_waiting'}:
        return not _looks_like_question(text)

    if dialog_act == 'retry':
        return _looks_like_question(text)

    return True


class ResponsePlannerServer(Node):
    def __init__(self) -> None:
        super().__init__('response_planner_server')
        self._server_cb_group = ReentrantCallbackGroup()
        self._client_cb_group = ReentrantCallbackGroup()

        self.declare_parameter('llm.chat_action_name', '/llm/chat')
        self.declare_parameter('llm.temperature', 0.2)
        self.declare_parameter('llm.max_tokens', 96)
        self.declare_parameter('render.action_name', '/reception/render_dialog')

        self._chat_action_name = str(self.get_parameter('llm.chat_action_name').value)
        self._temperature = float(self.get_parameter('llm.temperature').value)
        self._max_tokens = int(self.get_parameter('llm.max_tokens').value)
        self._render_action_name = str(self.get_parameter('render.action_name').value)

        self._chat_client = ActionClient(
            self,
            Chat,
            self._chat_action_name,
            callback_group=self._client_cb_group,
        )
        self._server = ActionServer(
            self,
            RenderDialog,
            self._render_action_name,
            self._execute,
            callback_group=self._server_cb_group,
        )

    def _execute(self, goal_handle: Any) -> RenderDialog.Result:
        req = goal_handle.request
        result = RenderDialog.Result()
        response_language = _sanitize_response_language(req.response_language)
        fallback = _fallback_dialog_text(
            req.dialog_act,
            req.working_info.name,
            req.working_info.affiliation,
            req.working_info.purpose,
            response_language,
        )

        if req.dialog_act == 'relay_secretary':
            result.text = req.secretary_reply_text.strip() or fallback
            result.used_fallback = not bool(req.secretary_reply_text.strip())
            goal_handle.succeed()
            return result

        prompt = (
            'Render spoken response for receptionist flow.\n'
            'The visible conversation transcript is provided separately as chat history.\n'
            'Use that history to stay consistent with the ongoing reception session.\n'
            f'response_language={response_language}\n'
            f'dialog_act={req.dialog_act}\n'
            f'phase={req.phase}\n'
            f'focus_slot={req.focus_slot}\n'
            f'pending_clarification_slot={req.pending_clarification_slot}\n'
            f'working_name={req.working_info.name}\n'
            f'working_affiliation={req.working_info.affiliation}\n'
            f'working_purpose={req.working_info.purpose}\n'
            f'committed_name={req.committed_info.name}\n'
            f'committed_affiliation={req.committed_info.affiliation}\n'
            f'committed_purpose={req.committed_info.purpose}\n'
            f'latest_user_text={req.latest_user_text}\n'
            f'fallback={fallback}\n'
        )

        try:
            text = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.session_id}:render:{req.turn_seq}',
                user_message=prompt,
                system_prompt=_stage2_system_prompt(response_language),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stateless=True,
                response_json_schema='',
                messages=clone_chat_messages(list(getattr(req, 'transcript_messages', []))),
            )
            cleaned = text.strip()
            if cleaned and _is_valid_rendered_response(req, cleaned):
                result.text = cleaned
                result.used_fallback = False
            else:
                if cleaned:
                    self.get_logger().warn(
                        f'render produced invalid response for dialog_act={req.dialog_act}; fallback used: {cleaned}'
                    )
                result.text = fallback
                result.used_fallback = True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'render failed, fallback used: {exc}')
            result.text = fallback
            result.used_fallback = True

        goal_handle.succeed()
        return result


def _fallback_dialog_text(dialog_act: str, name: str, affiliation: str, purpose: str, response_language: str) -> str:
    if _sanitize_response_language(response_language) == 'en':
        if dialog_act == 'ask_name':
            return 'May I have your name, please?'
        if dialog_act == 'ask_affiliation':
            return 'May I ask your affiliation, please?'
        if dialog_act == 'ask_purpose':
            return 'What brings you here today?'
        if dialog_act == 'clarify_name':
            return 'I may have misheard your name. Could you please tell me your name once more?'
        if dialog_act == 'clarify_affiliation':
            return 'I may have misheard your affiliation. Could you please tell me your affiliation once more?'
        if dialog_act == 'clarify_purpose':
            return 'I may have misheard your purpose. Could you please tell me what brings you here today once more?'
        if dialog_act == 'confirm_snapshot':
            n = name or 'unknown'
            a = affiliation or 'unknown'
            p = purpose or 'unknown'
            return f'Let me confirm: your name is {n}, your affiliation is {a}, and your purpose is {p}. Is that correct?'
        if dialog_act == 'notify_waiting':
            return 'I have notified the person in charge. Please wait for a moment.'
        if dialog_act == 'acknowledge_waiting':
            return 'The person in charge has already been notified. Please wait for a moment.'
        if dialog_act == 'retry':
            return 'Could you please say that again?'
        if dialog_act == 'relay_secretary':
            return 'A reply has arrived from the person in charge.'
        return 'Please wait for a moment.'
    if dialog_act == 'ask_name':
        return 'お名前を教えてください。'
    if dialog_act == 'ask_affiliation':
        honor = f'{name}様、' if name else ''
        return f'{honor}ご所属を教えてください。'
    if dialog_act == 'ask_purpose':
        return '本日のご用件を教えてください。'
    if dialog_act == 'clarify_name':
        return 'お名前を正しく伺えなかったため、もう一度お名前をお願いいたします。'
    if dialog_act == 'clarify_affiliation':
        return 'ご所属を正しく伺えなかったため、もう一度ご所属をお願いいたします。'
    if dialog_act == 'clarify_purpose':
        return 'ご用件を正しく伺えなかったため、もう一度ご用件をお願いいたします。'
    if dialog_act == 'confirm_snapshot':
        n = name or '未取得'
        a = affiliation or '未取得'
        p = purpose or '未取得'
        return f'お名前は{n}様、ご所属は{a}、ご用件は{p}でお間違いないでしょうか。'
    if dialog_act == 'notify_waiting':
        return '担当者へ連絡しました。少々お待ちください。'
    if dialog_act == 'acknowledge_waiting':
        return '担当者へ連絡済みです。少々お待ちください。'
    if dialog_act == 'retry':
        return 'もう一度お願いいたします。'
    if dialog_act == 'relay_secretary':
        return '担当者から連絡が入りました。'
    return '少々お待ちください。'


def main() -> None:
    rclpy.init()
    node = ResponsePlannerServer()
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
