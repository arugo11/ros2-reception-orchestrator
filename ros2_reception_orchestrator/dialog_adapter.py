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
TraceCallable = Callable[[str], None]


class DialogAdapter:
    def __init__(
        self,
        invoke_chat: ChatInvoker,
        *,
        temperature: float,
        max_tokens: int,
        system_prompt: str = RECEPTION_DIALOG_SYSTEM_PROMPT,
        trace: TraceCallable | None = None,
    ) -> None:
        self._invoke_chat = invoke_chat
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._trace = trace or (lambda _message: None)

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
        if not self._is_valid_response(request, spoken_response):
            self._trace(
                f'dialog_response_rejected stage=candidate act={request.dialog_act} '
                f'text={spoken_response}'
            )
            return fallback
        reviewed = self._review_response(request, spoken_response)
        if reviewed and self._is_valid_response(request, reviewed):
            return reviewed
        if reviewed:
            self._trace(
                f'dialog_response_rejected stage=review act={request.dialog_act} '
                f'text={reviewed}'
            )
        self._trace(f'dialog_response_fallback act={request.dialog_act}')
        return fallback

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

    def _is_valid_response(self, request: DialogRenderRequest, response: str) -> bool:
        normalized = _normalize_text(response)
        if not normalized:
            return False
        if len(normalized) > 160:
            return False
        if _normalize_text(request.latest_utterance) == normalized:
            return False

        if request.dialog_act in {'ask_name', 'ask_affiliation', 'ask_purpose', 'clarify'}:
            return _looks_like_information_request(normalized)

        if request.dialog_act == 'confirm':
            info = request.pending_confirmation or request.visitor_info
            expected_values = [
                _normalize_text(info.name or ''),
                _normalize_text(info.affiliation or ''),
                _normalize_text(info.purpose or ''),
            ]
            if not _looks_like_question(normalized):
                return False
            return any(value and value in normalized for value in expected_values)

        if request.dialog_act in {'notify_waiting', 'acknowledge_waiting'}:
            info = request.pending_confirmation or request.visitor_info
            repeated_values = [
                _normalize_text(info.name or ''),
                _normalize_text(info.affiliation or ''),
                _normalize_text(info.purpose or ''),
            ]
            if _looks_like_question(normalized):
                return False
            return not any(value and value in normalized for value in repeated_values)

        if request.dialog_act in {'ack_correction', 'close'}:
            return not _looks_like_question(normalized)

        return True


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


def _normalize_text(text: str) -> str:
    compact = ' '.join(text.split()).strip()
    return compact.lower()


def _looks_like_information_request(text: str) -> bool:
    compact = ' '.join(text.split()).strip()
    if not compact:
        return False
    if _looks_like_question(compact):
        return True
    solicitation_markers = (
        '教えて',
        '伺',
        'お聞かせ',
        'いただけます',
        'お願いできます',
        'お願いしても',
        '差し支えなければ',
        'もう一度',
    )
    return any(marker in compact for marker in solicitation_markers)


def _looks_like_question(text: str) -> bool:
    compact = ' '.join(text.split()).strip()
    if not compact:
        return False
    if '？' in compact or '?' in compact:
        return True
    question_markers = (
        '教えて',
        '伺',
        'いただけます',
        'よろしいでしょうか',
    )
    return any(marker in compact for marker in question_markers)
