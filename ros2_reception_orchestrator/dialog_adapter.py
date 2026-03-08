from __future__ import annotations

from collections import Counter
from typing import Callable

from .formatters import fallback_dialog_text
from .prompt_templates import DIALOG_SYSTEM_PROMPT
from .prompt_templates import build_dialog_repair_prompt
from .prompt_templates import build_dialog_user_prompt
from .state_models import DialogRenderRequest


ChatInvoker = Callable[[str, str, str, float, int, bool], str]


class DialogAdapter:
    def __init__(
        self,
        invoke_chat: ChatInvoker,
        *,
        temperature: float,
        max_tokens: int,
        system_prompt: str = DIALOG_SYSTEM_PROMPT,
    ) -> None:
        self._invoke_chat = invoke_chat
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    def render(self, request: DialogRenderRequest) -> str:
        fallback = fallback_dialog_text(request.dialog_act, request.visitor_info)
        try:
            rendered = self._invoke_chat(
                _request_session_id(request, 'dialog'),
                build_dialog_user_prompt(request),
                self._system_prompt,
                self._temperature,
                self._max_tokens,
                True,
            ).strip()
            if self._is_valid(request.dialog_act, rendered):
                return rendered

            repaired = self._invoke_chat(
                _request_session_id(request, 'dialog-repair'),
                build_dialog_repair_prompt(request, rendered),
                self._system_prompt,
                self._temperature,
                min(self._max_tokens, 48),
                True,
            ).strip()
            if self._is_valid(request.dialog_act, repaired):
                return repaired
        except Exception:
            return fallback
        return fallback

    def _is_valid(self, dialog_act: str, text: str) -> bool:
        if not text:
            return False
        if '\n' in text:
            return False
        if any(token in text for token in ('{', '}', '[', ']', 'dialog_act', 'phase=')):
            return False
        if any(token in text for token in ('来訪者', '出力ルール', '確認し', '情報を提供', '案内します', '受付として')):
            return False
        if text.startswith('「') and text.endswith('」'):
            return False
        if len(text) > 80:
            return False
        normalized = text.replace('？', '?').replace('。', '.')
        if normalized.count('?') + normalized.count('？') > 1:
            return False
        if self._has_repetition(text):
            return False
        if not self._matches_dialog_act(dialog_act, text):
            return False
        return True

    @staticmethod
    def _matches_dialog_act(dialog_act: str, text: str) -> bool:
        if dialog_act == 'ask_name':
            return '所属' not in text and '用件' not in text and '目的' not in text
        if dialog_act == 'ask_affiliation':
            return (
                '名前' not in text
                and '用件' not in text
                and '目的' not in text
                and 'ですね' not in text
                and 'ご所属' in text
            )
        if dialog_act == 'ask_purpose':
            return (
                '名前' not in text
                and '所属' not in text
                and 'でしょうか' not in text
                and ('用件' in text or '目的' in text or 'お越し' in text or '会いに' in text)
            )
        if dialog_act == 'notify_waiting':
            return '待' in text or '連絡' in text
        return True

    @staticmethod
    def _has_repetition(text: str) -> bool:
        sentences = [part.strip() for part in text.replace('？', '。').split('。') if part.strip()]
        if len(sentences) >= 2 and len(set(sentences)) < len(sentences):
            return True
        tokens = [token for token in text.replace('、', ' ').replace('。', ' ').split() if token]
        if not tokens:
            return False
        counts = Counter(tokens)
        return any(count >= 4 for count in counts.values())


def _request_session_id(request: DialogRenderRequest, purpose: str) -> str:
    return f'{request.session_id}:{purpose}:{request.turn_id}'
