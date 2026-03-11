from __future__ import annotations

from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from reception_interfaces.action import RenderDialog
from ros2_vllm_interfaces.action import Chat

from .llm_stage_utils import invoke_chat_action


_STAGE2_SYSTEM_PROMPT = (
    'You are a polite Japanese university receptionist assistant. '
    'Generate one concise spoken response in Japanese. '
    'No JSON. No markdown. Keep to 1-2 sentences.'
)


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
        fallback = _fallback_dialog_text(req.dialog_act, req.visitor_info.name, req.visitor_info.affiliation, req.visitor_info.purpose)

        if req.dialog_act == 'relay_secretary':
            result.text = req.secretary_reply_text.strip() or fallback
            result.used_fallback = not bool(req.secretary_reply_text.strip())
            goal_handle.succeed()
            return result

        prompt = (
            'Render spoken response for receptionist flow.\n'
            f'dialog_act={req.dialog_act}\n'
            f'phase={req.phase}\n'
            f'name={req.visitor_info.name}\n'
            f'affiliation={req.visitor_info.affiliation}\n'
            f'purpose={req.visitor_info.purpose}\n'
            f'latest_user_text={req.latest_user_text}\n'
            f'fallback={fallback}\n'
        )

        try:
            text = invoke_chat_action(
                client=self._chat_client,
                action_name=self._chat_action_name,
                session_id=f'{req.session_id}:render:{req.turn_seq}',
                user_message=prompt,
                system_prompt=_STAGE2_SYSTEM_PROMPT,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stateless=True,
                response_json_schema='',
            )
            cleaned = text.strip()
            result.text = cleaned or fallback
            result.used_fallback = not bool(cleaned)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'render failed, fallback used: {exc}')
            result.text = fallback
            result.used_fallback = True

        goal_handle.succeed()
        return result


def _fallback_dialog_text(dialog_act: str, name: str, affiliation: str, purpose: str) -> str:
    if dialog_act == 'ask_name':
        return 'お名前を教えてください。'
    if dialog_act == 'ask_affiliation':
        honor = f'{name}様、' if name else ''
        return f'{honor}ご所属を教えてください。'
    if dialog_act == 'ask_purpose':
        return '本日のご用件を教えてください。'
    if dialog_act == 'confirm':
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
