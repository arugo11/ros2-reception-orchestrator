from __future__ import annotations

import json
import re
from typing import Callable

from .prompt_templates import SUPERVISOR_REPAIR_SYSTEM_PROMPT
from .prompt_templates import SLOT_EXTRACTION_SYSTEM_PROMPT
from .prompt_templates import SUPERVISOR_SYSTEM_PROMPT
from .prompt_templates import build_slot_extraction_prompt
from .prompt_templates import build_supervisor_repair_prompt
from .prompt_templates import build_supervisor_user_prompt
from .state_models import DialogAct
from .state_models import FieldName
from .state_models import SessionSnapshot
from .state_models import SupervisorDecision


ChatInvoker = Callable[[str, str, str, float, int, bool], str]

_VALID_ACTS: set[str] = {
    'ask_name',
    'ask_affiliation',
    'ask_purpose',
    'confirm',
    'notify_waiting',
    'acknowledge_waiting',
    'clarify',
    'retry',
    'relay_secretary',
}
_VALID_SPEECH_ACTS: set[str] = {
    'inform',
    'affirm',
    'deny',
    'correction',
    'question',
    'complaint',
    'greeting',
    'unknown',
}
_VALID_CORRECTIONS: set[str] = {'none', 'name', 'affiliation', 'purpose', 'all'}
_VALID_DISCORD_KINDS: set[str] = {'initial', 'update', 'confirmed', 'none'}


class SupervisorAdapter:
    def __init__(
        self,
        invoke_chat: ChatInvoker,
        *,
        temperature: float,
        max_tokens: int,
        system_prompt: str = SUPERVISOR_SYSTEM_PROMPT,
    ) -> None:
        self._invoke_chat = invoke_chat
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    def analyze(self, snapshot: SessionSnapshot, latest_utterance: str) -> SupervisorDecision:
        fallback = SupervisorDecision(missing_fields=snapshot.visitor_info.missing_fields())
        request_session_id = _request_session_id(snapshot)
        payload: dict[str, object] | None = None
        try:
            raw = self._invoke_chat(
                request_session_id,
                build_supervisor_user_prompt(snapshot, latest_utterance),
                self._system_prompt,
                self._temperature,
                self._max_tokens,
                True,
            )
            payload = _extract_json_object(raw)
        except Exception:
            return fallback

        if not _has_usable_supervisor_payload(payload):
            try:
                repaired = self._invoke_chat(
                    f'{request_session_id}:repair',
                    build_supervisor_repair_prompt(snapshot, latest_utterance, _truncate(raw)),
                    SUPERVISOR_REPAIR_SYSTEM_PROMPT,
                    0.0,
                    min(self._max_tokens, 64),
                    True,
                )
                repaired_payload = _extract_json_object(repaired)
                if isinstance(payload, dict) and isinstance(repaired_payload, dict):
                    payload = _merge_payloads(payload, repaired_payload)
                elif isinstance(repaired_payload, dict):
                    payload = repaired_payload
            except Exception:
                pass
        if not _has_usable_supervisor_payload(payload):
            return fallback

        assert isinstance(payload, dict)
        speech_act = str(payload.get('speech_act', 'unknown')).strip()
        next_dialog_act = str(payload.get('next_dialog_act', '')).strip()
        correction_target = str(payload.get('correction_target', 'none')).strip()
        discord_update_kind = str(payload.get('discord_update_kind', 'none')).strip()
        decision = SupervisorDecision(
            **_normalize_decision_kwargs(
                snapshot=snapshot,
                speech_act=(speech_act if speech_act in _VALID_SPEECH_ACTS else 'unknown'),
                extracted_name=_optional_string(payload.get('extracted_name')),
                extracted_affiliation=_optional_string(payload.get('extracted_affiliation')),
                extracted_purpose=_optional_string(payload.get('extracted_purpose')),
                slot_confidence=_optional_float(payload.get('slot_confidence')),
                next_dialog_act=(next_dialog_act if next_dialog_act in _VALID_ACTS else None),
                should_confirm=bool(payload.get('should_confirm', False)),
                correction_target=(
                    correction_target if correction_target in _VALID_CORRECTIONS else 'none'
                ),
                discord_update_kind=(
                    discord_update_kind if discord_update_kind in _VALID_DISCORD_KINDS else 'none'
                ),
                ignore_input=bool(payload.get('ignore_input', False)),
            )
        )
        if decision.ignore_input or decision.speech_act not in {'inform', 'correction'}:
            if decision.speech_act not in {'deny', 'correction'}:
                return decision
        return self._enrich_missing_slots(snapshot, latest_utterance, decision, request_session_id)

    def _enrich_missing_slots(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        decision: SupervisorDecision,
        request_session_id: str,
    ) -> SupervisorDecision:
        if not decision.missing_fields:
            return decision

        prioritized_fields = self._target_fields(snapshot, decision, latest_utterance)
        if not prioritized_fields:
            return decision

        try:
            raw = self._invoke_chat(
                f'{request_session_id}:slot-extract',
                build_slot_extraction_prompt(snapshot, latest_utterance, prioritized_fields),
                SLOT_EXTRACTION_SYSTEM_PROMPT,
                0.0,
                min(self._max_tokens, 48),
                True,
            )
        except Exception:
            return decision

        payload = _extract_json_object(raw)
        if not isinstance(payload, dict):
            return decision

        extracted_name = _optional_string(payload.get('name'))
        extracted_affiliation = _optional_string(payload.get('affiliation'))
        extracted_purpose = _optional_string(payload.get('purpose'))
        enriched_name = decision.extracted_name or extracted_name
        enriched_affiliation = decision.extracted_affiliation or extracted_affiliation
        enriched_purpose = decision.extracted_purpose or extracted_purpose

        normalized_speech_act = decision.speech_act
        normalized_correction_target = decision.correction_target
        if decision.speech_act == 'deny':
            corrected_fields = _infer_corrected_fields(
                snapshot,
                extracted_name=enriched_name,
                extracted_affiliation=enriched_affiliation,
                extracted_purpose=enriched_purpose,
            )
            if corrected_fields:
                normalized_speech_act = 'correction'
                normalized_correction_target = _correction_target_for_fields(corrected_fields)

        return SupervisorDecision(
            **_normalize_decision_kwargs(
                snapshot=snapshot,
                speech_act=normalized_speech_act,
                extracted_name=enriched_name,
                extracted_affiliation=enriched_affiliation,
                extracted_purpose=enriched_purpose,
                slot_confidence=decision.slot_confidence,
                next_dialog_act=decision.next_dialog_act,
                should_confirm=decision.should_confirm,
                correction_target=normalized_correction_target,
                discord_update_kind=decision.discord_update_kind,
                ignore_input=decision.ignore_input,
            )
        )

    @staticmethod
    def _target_fields(
        snapshot: SessionSnapshot,
        decision: SupervisorDecision,
        latest_utterance: str,
    ) -> list[str]:
        last_act = snapshot.last_dialog_act or ''
        lowered = latest_utterance.strip()
        if decision.speech_act in {'deny', 'correction'} or any(
            token in lowered for token in ('違います', '訂正', 'ではなく')
        ):
            correction_fields = _explicit_correction_targets(lowered)
            if correction_fields:
                return correction_fields
            return ['name', 'affiliation', 'purpose']
        if last_act == 'ask_name' and 'name' in decision.missing_fields:
            return ['name']
        if last_act == 'ask_affiliation' and 'affiliation' in decision.missing_fields:
            return ['affiliation']
        if last_act == 'ask_purpose' and 'purpose' in decision.missing_fields:
            return ['purpose']
        if len(decision.missing_fields) == 1:
            return list(decision.missing_fields)
        return []


def _normalize_decision_kwargs(
    *,
    snapshot: SessionSnapshot,
    speech_act: str,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
    slot_confidence: float,
    next_dialog_act: str | None,
    should_confirm: bool,
    correction_target: str,
    discord_update_kind: str,
    ignore_input: bool,
) -> dict[str, object]:
    current_name = extracted_name or snapshot.visitor_info.name
    current_affiliation = extracted_affiliation or snapshot.visitor_info.affiliation
    current_purpose = extracted_purpose or snapshot.visitor_info.purpose

    missing_fields: list[FieldName] = []
    if not current_name:
        missing_fields.append('name')
    if not current_affiliation:
        missing_fields.append('affiliation')
    if not current_purpose:
        missing_fields.append('purpose')

    normalized_should_confirm = bool(should_confirm and not missing_fields)
    normalized_correction_target = correction_target if speech_act == 'correction' else 'none'
    normalized_next_dialog_act = next_dialog_act

    changed_fields: list[FieldName] = []
    if extracted_name and snapshot.visitor_info.name and extracted_name != snapshot.visitor_info.name:
        changed_fields.append('name')
    if (
        extracted_affiliation
        and snapshot.visitor_info.affiliation
        and extracted_affiliation != snapshot.visitor_info.affiliation
    ):
        changed_fields.append('affiliation')
    if (
        extracted_purpose
        and snapshot.visitor_info.purpose
        and extracted_purpose != snapshot.visitor_info.purpose
    ):
        changed_fields.append('purpose')

    if changed_fields:
        speech_act = 'correction'
        normalized_correction_target = _correction_target_for_fields(changed_fields)

    if speech_act == 'deny':
        corrected_fields = _infer_corrected_fields(
            snapshot,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        if corrected_fields:
            speech_act = 'correction'
            normalized_correction_target = _correction_target_for_fields(corrected_fields)

    if ignore_input:
        normalized_should_confirm = False
        normalized_next_dialog_act = 'retry'
        normalized_correction_target = 'none'
    elif snapshot.phase == 'confirming' and speech_act == 'affirm':
        normalized_next_dialog_act = 'notify_waiting'
        normalized_should_confirm = True
    elif missing_fields:
        normalized_should_confirm = False
        normalized_next_dialog_act = _dialog_act_for_missing(missing_fields[0])
    else:
        normalized_next_dialog_act = 'confirm'
        normalized_should_confirm = True

    return {
        'speech_act': speech_act,
        'extracted_name': extracted_name,
        'extracted_affiliation': extracted_affiliation,
        'extracted_purpose': extracted_purpose,
        'slot_confidence': slot_confidence,
        'missing_fields': missing_fields,
        'next_dialog_act': normalized_next_dialog_act,
        'should_confirm': normalized_should_confirm,
        'correction_target': normalized_correction_target,
        'discord_update_kind': discord_update_kind,
        'ignore_input': ignore_input,
    }


def _dialog_act_for_missing(field_name: FieldName) -> DialogAct:
    if field_name == 'name':
        return 'ask_name'
    if field_name == 'affiliation':
        return 'ask_affiliation'
    return 'ask_purpose'


def _extract_json_object(raw: str) -> dict[str, object] | None:
    if not raw.strip():
        return None
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    lowered = text.lower()
    if lowered in {'', 'unknown', 'none', 'null', 'n/a', 'na', '-'}:
        return None
    if text in {'不明', '未取得', 'なし', '未定', '該当なし'}:
        return None
    return text or None


def _optional_float(value: object) -> float:
    if isinstance(value, dict):
        numeric_values: list[float] = []
        for nested in value.values():
            try:
                numeric_values.append(float(nested))
            except (TypeError, ValueError):
                continue
        if numeric_values:
            return max(0.0, min(1.0, max(numeric_values)))
        return 0.0
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _merge_payloads(
    original: dict[str, object],
    repaired: dict[str, object],
) -> dict[str, object]:
    merged = dict(original)
    for key, value in repaired.items():
        if key in {'extracted_name', 'extracted_affiliation', 'extracted_purpose'}:
            if _optional_string(merged.get(key)) and not _optional_string(value):
                continue
        if key == 'missing_fields':
            if merged.get(key) and not value:
                continue
        merged[key] = value
    return merged


def _explicit_correction_targets(latest_utterance: str) -> list[str]:
    lowered = latest_utterance.lower()
    targets: list[str] = []
    if '名前' in latest_utterance or '氏名' in latest_utterance:
        targets.append('name')
    if '所属' in latest_utterance or '研究室' in latest_utterance:
        targets.append('affiliation')
    if '用件' in latest_utterance or '目的' in latest_utterance or '会いに' in latest_utterance:
        targets.append('purpose')
    if '名前' not in latest_utterance and '所属' not in latest_utterance and '用件' not in latest_utterance and '目的' not in latest_utterance:
        if any(token in lowered for token in ('違います', '訂正', 'ではなく')):
            targets.append('name')
    seen: list[str] = []
    for target in targets:
        if target not in seen:
            seen.append(target)
    return seen


def _infer_corrected_fields(
    snapshot: SessionSnapshot,
    *,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> list[FieldName]:
    corrected: list[FieldName] = []
    if extracted_name and extracted_name != snapshot.visitor_info.name:
        corrected.append('name')
    if extracted_affiliation and extracted_affiliation != snapshot.visitor_info.affiliation:
        corrected.append('affiliation')
    if extracted_purpose and extracted_purpose != snapshot.visitor_info.purpose:
        corrected.append('purpose')
    return corrected


def _correction_target_for_fields(fields: list[FieldName]) -> str:
    if not fields:
        return 'none'
    if len(fields) == 1:
        return fields[0]
    return 'all'


def _request_session_id(snapshot: SessionSnapshot) -> str:
    return f'{snapshot.session_id}:supervisor:{snapshot.latest_turn_id}'


def _truncate(text: str, limit: int = 200) -> str:
    cleaned = ' '.join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + '...'


def _has_usable_supervisor_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        key in payload
        for key in (
            'speech_act',
            'extracted_name',
            'extracted_affiliation',
            'extracted_purpose',
            'next_dialog_act',
            'ignore_input',
        )
    )
