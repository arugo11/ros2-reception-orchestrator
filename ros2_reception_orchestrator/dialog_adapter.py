from __future__ import annotations

import json
import re
from typing import Callable

from .formatters import fallback_dialog_text
from .prompt_templates import RECEPTION_DIALOG_REVIEW_JSON_SCHEMA
from .prompt_templates import RECEPTION_DIALOG_REVIEW_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_DIALOG_RESPONSE_JSON_SCHEMA
from .prompt_templates import RECEPTION_DIALOG_SYSTEM_PROMPT
from .prompt_templates import build_reception_dialog_prompt
from .prompt_templates import build_reception_dialog_review_prompt
from .state_models import DialogRenderRequest


ChatInvoker = Callable[[str, str, str, float, int, bool, str | None], str]


class DialogAdapter:
    def __init__(
        self,
        invoke_chat: ChatInvoker,
        *,
        temperature: float,
        max_tokens: int,
        system_prompt: str = RECEPTION_DIALOG_SYSTEM_PROMPT,
    ) -> None:
        self._invoke_chat = invoke_chat
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    def render(self, request: DialogRenderRequest) -> str:
        fallback = fallback_dialog_text(request.dialog_act, request.visitor_info)
        if request.dialog_act == 'relay_secretary' and request.secretary_reply_text.strip():
            return request.secretary_reply_text.strip()
        try:
            raw = self._invoke_chat(
                f'{request.session_id}:turn:{request.turn_id}:dialog',
                build_reception_dialog_prompt(request),
                self._system_prompt,
                self._temperature,
                self._max_tokens,
                True,
                RECEPTION_DIALOG_RESPONSE_JSON_SCHEMA,
            )
        except Exception:
            return fallback

        spoken_response = _extract_spoken_response(raw)
        if not spoken_response:
            return fallback
        reviewed = self._review_response(request, spoken_response)
        if reviewed:
            return reviewed
        return spoken_response

    def _review_response(self, request: DialogRenderRequest, candidate_response: str) -> str | None:
        try:
            raw = self._invoke_chat(
                f'{request.session_id}:turn:{request.turn_id}:dialog-review',
                build_reception_dialog_review_prompt(request, candidate_response),
                RECEPTION_DIALOG_REVIEW_SYSTEM_PROMPT,
                0.0,
                min(self._max_tokens, 80),
                True,
                RECEPTION_DIALOG_REVIEW_JSON_SCHEMA,
            )
        except Exception:
            return None

        review = _extract_review_result(raw)
        if review is None:
            return None
        accept, spoken_response = review
        if accept:
            return candidate_response
        return spoken_response or None


def _extract_spoken_response(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get('spoken_response')
    if not isinstance(value, str):
        return None
    cleaned = ' '.join(part for part in value.splitlines() if part.strip()).strip()
    if not cleaned:
        return None
    return cleaned[:160]


def _extract_review_result(raw: str) -> tuple[bool, str | None] | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    accept = bool(parsed.get('accept', False))
    spoken_response = parsed.get('spoken_response')
    if not isinstance(spoken_response, str):
        return None
    cleaned = ' '.join(part for part in spoken_response.splitlines() if part.strip()).strip()
    if not cleaned:
        return None
    return accept, cleaned[:160]
