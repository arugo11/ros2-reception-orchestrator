from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class DiscordCreateThreadResponse:
    success: bool
    thread_id: str = ''
    channel_id: str = ''
    message_id: str = ''
    error_message: str = ''


CreateThreadCallable = Callable[[str, str, str], DiscordCreateThreadResponse]
SendMessageCallable = Callable[[str, str], bool]


class DiscordAdapter:
    def __init__(
        self,
        create_thread: CreateThreadCallable,
        send_thread_message: SendMessageCallable,
    ) -> None:
        self._create_thread = create_thread
        self._send_thread_message = send_thread_message

    def create_thread(self, thread_title: str, initial_text: str, session_label: str) -> DiscordCreateThreadResponse:
        return self._create_thread(thread_title, initial_text, session_label)

    def send_thread_message(self, thread_id: str, text: str) -> bool:
        return self._send_thread_message(thread_id, text)
