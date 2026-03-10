from __future__ import annotations

from typing import Callable

from .formatters import fallback_dialog_text
from .state_models import DialogRenderRequest


ChatInvoker = Callable[[str, str, str, float, int, bool], str]


class DialogAdapter:
    """Compatibility wrapper kept for older tests/imports.

    The reception runtime now uses a single-pass LLM path and does not call this
    adapter in the main orchestrator flow.
    """

    def __init__(
        self,
        invoke_chat: ChatInvoker,
        *,
        temperature: float,
        max_tokens: int,
        system_prompt: str = '',
    ) -> None:
        del invoke_chat, temperature, max_tokens, system_prompt

    def render(self, request: DialogRenderRequest) -> str:
        return fallback_dialog_text(request.dialog_act, request.visitor_info)
