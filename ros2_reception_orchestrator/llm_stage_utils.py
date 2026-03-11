from __future__ import annotations

import json
import time
from typing import Any

from rclpy.action import ActionClient

from ros2_vllm_interfaces.action import Chat


def extract_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or '').strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    first = text.find('{')
    last = text.rfind('}')
    if first < 0 or last <= first:
        return None

    candidate = text[first : last + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def wait_future(future: Any, timeout_sec: float) -> Any | None:
    deadline = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not future.done():
        return None
    return future.result()


def invoke_chat_action(
    *,
    client: ActionClient,
    action_name: str,
    session_id: str,
    user_message: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    stateless: bool,
    response_json_schema: str = '',
    server_wait_timeout_sec: float = 10.0,
    total_timeout_sec: float = 90.0,
) -> str:
    if not client.wait_for_server(timeout_sec=server_wait_timeout_sec):
        raise RuntimeError(f'{action_name} action server is unavailable')

    goal = Chat.Goal()
    goal.session_id = session_id
    goal.user_message = user_message
    goal.system_prompt = system_prompt
    goal.temperature = float(temperature)
    goal.max_tokens = int(max_tokens)
    goal.stateless = bool(stateless)
    if hasattr(goal, 'response_json_schema'):
        goal.response_json_schema = response_json_schema or ''

    goal_handle = wait_future(client.send_goal_async(goal), timeout_sec=10.0)
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError(f'{action_name} goal rejected')

    wrapped = wait_future(goal_handle.get_result_async(), timeout_sec=total_timeout_sec)
    if wrapped is None:
        raise RuntimeError(f'{action_name} timeout waiting result')

    result = wrapped.result
    if not result.success:
        raise RuntimeError(result.error_message or f'{action_name} failed')
    return str(result.assistant_message or '').strip()
