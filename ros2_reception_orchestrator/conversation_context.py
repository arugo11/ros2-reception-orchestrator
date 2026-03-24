from __future__ import annotations

from ros2_vllm_interfaces.msg import ChatMessage


_VALID_CHAT_ROLES = {'system', 'user', 'assistant'}


def sanitize_chat_role(value: object) -> str:
    role = str(value or 'user').strip().lower()
    if role in _VALID_CHAT_ROLES:
        return role
    return 'user'


def make_chat_message(role: object, content: object) -> ChatMessage:
    message = ChatMessage()
    message.role = sanitize_chat_role(role)
    message.content = str(content or '').strip()
    return message


def clone_chat_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    cloned: list[ChatMessage] = []
    for message in messages:
        copied = ChatMessage()
        copied.role = sanitize_chat_role(getattr(message, 'role', 'user'))
        copied.content = str(getattr(message, 'content', '') or '').strip()
        if copied.content:
            cloned.append(copied)
    return cloned
