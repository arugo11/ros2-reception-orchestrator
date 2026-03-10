from __future__ import annotations

import json
import re
from typing import Callable

from .prompt_templates import RECEPTION_REPAIR_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA
from .prompt_templates import RECEPTION_RESPONSE_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_EXTRACT_JSON_SCHEMA
from .prompt_templates import RECEPTION_SYSTEM_PROMPT
from .prompt_templates import build_reception_confirmation_rescue_prompt
from .prompt_templates import build_reception_correction_rescue_prompt
from .prompt_templates import build_reception_repair_prompt
from .prompt_templates import build_reception_slot_extract_prompt
from .prompt_templates import build_reception_user_prompt
from .state_models import FieldName
from .state_models import SessionSnapshot
from .state_models import SupervisorDecision


ChatInvoker = Callable[[str, str, str, float, int, bool, str | None], str]

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
_AFFILIATION_TOKENS: tuple[str, ...] = (
    '研究室',
    '大学',
    '株式会社',
    '会社',
    '学部',
    '研究所',
    'センター',
    '部署',
    '部',
    '課',
    '学校',
    '法人',
)
_PURPOSE_CUES: tuple[str, ...] = (
    '面会',
    '打ち合わせ',
    '訪問',
    '用件',
    '用が',
    '要が',
    'ようが',
    '相談',
    '面談',
    'アポイント',
    '会いに',
    '来ました',
    'きました',
    '参りました',
    'まいりました',
    'できました',
    'やってきました',
    '伺いました',
    '書類',
    'の件',
)
_GENERIC_AFFILIATIONS: frozenset[str] = frozenset(_AFFILIATION_TOKENS)


class SupervisorAdapter:
    def __init__(
        self,
        invoke_chat: ChatInvoker,
        *,
        temperature: float,
        max_tokens: int,
        system_prompt: str = RECEPTION_SYSTEM_PROMPT,
        trace: Callable[[str], None] | None = None,
    ) -> None:
        self._invoke_chat = invoke_chat
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._trace = trace or (lambda _message: None)

    def analyze(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        currently_speaking: bool,
        captured_during_tts: bool,
    ) -> SupervisorDecision:
        fallback = _fallback_decision(snapshot)
        request_session_id = _request_session_id(snapshot)
        raw = ''
        try:
            raw = self._invoke_chat(
                request_session_id,
                build_reception_user_prompt(
                    snapshot,
                    latest_utterance,
                    currently_speaking=currently_speaking,
                    captured_during_tts=captured_during_tts,
                ),
                self._system_prompt,
                self._temperature,
                self._max_tokens,
                True,
                RECEPTION_RESPONSE_JSON_SCHEMA,
            )
            payload = _extract_json_object(raw)
        except Exception as exc:
            self._trace(f'supervisor_primary_failed error={exc}')
            return fallback

        if not _has_usable_payload(payload):
            self._trace(f'supervisor_primary_unusable raw={_truncate(raw, 120)}')
            try:
                repaired = self._invoke_chat(
                    f'{request_session_id}:repair',
                    build_reception_repair_prompt(
                        snapshot,
                        latest_utterance,
                        _truncate(raw),
                        currently_speaking=currently_speaking,
                        captured_during_tts=captured_during_tts,
                    ),
                    RECEPTION_REPAIR_SYSTEM_PROMPT,
                    0.0,
                    min(self._max_tokens, 64),
                    True,
                    RECEPTION_RESPONSE_JSON_SCHEMA,
                )
                payload = _extract_json_object(repaired)
            except Exception as exc:
                self._trace(f'supervisor_repair_failed error={exc}')
                return fallback

        if not _has_usable_payload(payload):
            self._trace('supervisor_repair_unusable')
            return fallback

        assert isinstance(payload, dict)
        updates = payload.get('slot_updates')
        if not isinstance(updates, dict):
            updates = {}
        correction = payload.get('correction')
        if not isinstance(correction, dict):
            correction = {}
        confirmation = payload.get('confirmation')
        if not isinstance(confirmation, dict):
            confirmation = {}

        speech_act = str(payload.get('speech_act', 'unknown')).strip()
        extracted_name = _optional_string(updates.get('name'))
        extracted_affiliation = _optional_string(updates.get('affiliation'))
        extracted_purpose = _optional_string(updates.get('purpose'))
        correction_target = str(correction.get('target', 'none')).strip()
        overwrite = bool(correction.get('overwrite', False))
        spoken_response = _normalize_spoken_response(payload.get('spoken_response'))
        ignore_input = bool(payload.get('ignore_input', False))
        slot_confidence = _optional_float(payload.get('confidence'))
        if speech_act not in _VALID_SPEECH_ACTS:
            speech_act = 'unknown'
        if correction_target not in _VALID_CORRECTIONS:
            correction_target = 'none'
        if speech_act not in {'deny', 'correction'}:
            correction_target = 'none'
            overwrite = False

        extracted_name, extracted_affiliation, extracted_purpose = _sanitize_slot_updates(
            snapshot,
            speech_act=speech_act,
            correction_target=correction_target,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )

        if _needs_correction_rescue(
            snapshot,
            speech_act=speech_act,
            correction_target=correction_target,
            overwrite=overwrite,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        ):
            self._trace(f'supervisor_correction_rescue target={correction_target} overwrite={overwrite}')
            try:
                rescue_raw = self._invoke_chat(
                    f'{request_session_id}:correction',
                    build_reception_correction_rescue_prompt(
                        snapshot,
                        latest_utterance,
                        target_field=correction_target if correction_target in {'name', 'affiliation', 'purpose'} else 'name',
                    ),
                    RECEPTION_REPAIR_SYSTEM_PROMPT,
                    0.0,
                    min(self._max_tokens, 64),
                    True,
                    RECEPTION_RESPONSE_JSON_SCHEMA,
                )
                rescue_payload = _extract_json_object(rescue_raw)
                if isinstance(rescue_payload, dict):
                    rescue_updates = rescue_payload.get('slot_updates')
                    if isinstance(rescue_updates, dict):
                        extracted_name = _optional_string(rescue_updates.get('name')) or extracted_name
                        extracted_affiliation = _optional_string(rescue_updates.get('affiliation')) or extracted_affiliation
                        extracted_purpose = _optional_string(rescue_updates.get('purpose')) or extracted_purpose
                    rescue_correction = rescue_payload.get('correction')
                    if isinstance(rescue_correction, dict):
                        correction_target = str(rescue_correction.get('target', correction_target)).strip() or correction_target
                        overwrite = bool(rescue_correction.get('overwrite', overwrite))
            except Exception:
                pass

        rescue_fields = _target_fields_for_slot_rescue(
            snapshot,
            latest_utterance=latest_utterance,
            speech_act=speech_act,
            slot_confidence=slot_confidence,
            correction_target=correction_target,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        if rescue_fields:
            self._trace(f'supervisor_slot_rescue fields={",".join(rescue_fields)}')
            try:
                rescue_raw = self._invoke_chat(
                    f'{request_session_id}:slot-extract',
                    build_reception_slot_extract_prompt(
                        snapshot,
                        latest_utterance,
                        target_fields=rescue_fields,
                    ),
                    RECEPTION_REPAIR_SYSTEM_PROMPT,
                    0.0,
                    min(self._max_tokens, 48),
                    True,
                    RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
                )
                slot_payload = _extract_json_object(rescue_raw)
                if isinstance(slot_payload, dict):
                    if 'name' in rescue_fields:
                        extracted_name = _optional_string(slot_payload.get('name'))
                    if 'affiliation' in rescue_fields:
                        extracted_affiliation = _optional_string(slot_payload.get('affiliation'))
                    if 'purpose' in rescue_fields:
                        extracted_purpose = _optional_string(slot_payload.get('purpose'))
            except Exception:
                pass

        extracted_name, extracted_affiliation, extracted_purpose = _sanitize_slot_updates(
            snapshot,
            speech_act=speech_act,
            correction_target=correction_target,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        (
            extracted_name,
            extracted_affiliation,
            extracted_purpose,
            rejected_fields,
            recovered_fields,
        ) = _ground_slot_updates(
            snapshot,
            latest_utterance,
            speech_act=speech_act,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        for field_name in rejected_fields:
            self._trace(f'slot_rejected field={field_name} utterance={_truncate(latest_utterance, 80)}')
        for field_name in recovered_fields:
            self._trace(f'slot_recovered field={field_name} utterance={_truncate(latest_utterance, 80)}')
        if rejected_fields or recovered_fields:
            spoken_response = None

        changed_fields = _infer_corrected_fields(
            snapshot,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        if (
            snapshot.phase == 'confirming'
            and not changed_fields
            and speech_act not in {'affirm', 'deny', 'correction'}
        ):
            self._trace('supervisor_confirmation_rescue')
            try:
                rescue_raw = self._invoke_chat(
                    f'{request_session_id}:confirm',
                    build_reception_confirmation_rescue_prompt(snapshot, latest_utterance),
                    RECEPTION_REPAIR_SYSTEM_PROMPT,
                    0.0,
                    min(self._max_tokens, 48),
                    True,
                    RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA,
                )
                rescue_payload = _extract_json_object(rescue_raw)
                if isinstance(rescue_payload, dict):
                    rescue_speech_act = str(rescue_payload.get('speech_act', '')).strip()
                    if rescue_speech_act in {'affirm', 'deny', 'correction', 'unknown'}:
                        speech_act = rescue_speech_act
                    rescue_correction = rescue_payload.get('correction')
                    if isinstance(rescue_correction, dict):
                        correction_target = str(rescue_correction.get('target', correction_target)).strip() or correction_target
                        overwrite = bool(rescue_correction.get('overwrite', overwrite))
                    rescue_confirmation = rescue_payload.get('confirmation')
                    if isinstance(rescue_confirmation, dict):
                        confirmation = {
                            'ready': bool(rescue_confirmation.get('ready', confirmation.get('ready', False))),
                            'accepted': bool(rescue_confirmation.get('accepted', confirmation.get('accepted', False))),
                        }
            except Exception:
                pass
        if speech_act == 'deny' and changed_fields:
            speech_act = 'correction'
            correction_target = _correction_target_for_fields(changed_fields)
        if overwrite and changed_fields:
            speech_act = 'correction'
            correction_target = _correction_target_for_fields(changed_fields)

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

        should_confirm = bool(confirmation.get('ready', False) and not missing_fields)
        accepted = bool(confirmation.get('accepted', False) and not missing_fields)
        if accepted:
            speech_act = 'affirm'
        if ignore_input:
            should_confirm = False
            correction_target = 'none'

        return SupervisorDecision(
            speech_act=speech_act,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
            slot_confidence=slot_confidence,
            missing_fields=missing_fields,
            next_dialog_act=None,
            should_confirm=should_confirm,
            correction_target=correction_target if speech_act == 'correction' else 'none',
            discord_update_kind='none',
            ignore_input=ignore_input,
            spoken_response=spoken_response,
        )


def _fallback_decision(snapshot: SessionSnapshot) -> SupervisorDecision:
    return SupervisorDecision(missing_fields=snapshot.visitor_info.missing_fields())


def _extract_json_object(raw: str) -> dict[str, object] | None:
    stripped = raw.strip()
    if not stripped:
        return None
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
    if _is_noninformative_slot_value(text):
        return None
    return text or None


def _optional_float(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_spoken_response(value: object) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    if '\n' in text:
        text = ' '.join(part for part in text.splitlines() if part.strip())
    if any(token in text for token in ('{', '}', '[', ']', 'slot_updates', 'speech_act')):
        return None
    return text[:160]


def _request_session_id(snapshot: SessionSnapshot) -> str:
    return f'{snapshot.session_id}:turn:{snapshot.latest_turn_id}'


def _truncate(text: str, limit: int = 200) -> str:
    cleaned = ' '.join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + '...'


def _has_usable_payload(payload: object) -> bool:
    return isinstance(payload, dict) and any(
        key in payload for key in ('speech_act', 'slot_updates', 'confirmation', 'spoken_response')
    )


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


def _sanitize_slot_updates(
    snapshot: SessionSnapshot,
    *,
    speech_act: str,
    correction_target: str,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> tuple[str | None, str | None, str | None]:
    current = snapshot.visitor_info
    correction_mode = speech_act in {'deny', 'correction'} and correction_target != 'none'

    if current.name and extracted_name and extracted_name != current.name and not correction_mode:
        extracted_name = None
    if current.affiliation and extracted_affiliation and extracted_affiliation != current.affiliation and not correction_mode:
        extracted_affiliation = None
    if current.purpose and extracted_purpose and extracted_purpose != current.purpose and not correction_mode:
        extracted_purpose = None

    return extracted_name, extracted_affiliation, extracted_purpose


def _needs_correction_rescue(
    snapshot: SessionSnapshot,
    *,
    speech_act: str,
    correction_target: str,
    overwrite: bool,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> bool:
    if speech_act not in {'deny', 'correction'} and correction_target == 'none' and not overwrite:
        return False
    if correction_target == 'name' and extracted_name and extracted_name != snapshot.visitor_info.name:
        return False
    if correction_target == 'affiliation' and extracted_affiliation and extracted_affiliation != snapshot.visitor_info.affiliation:
        return False
    if correction_target == 'purpose' and extracted_purpose and extracted_purpose != snapshot.visitor_info.purpose:
        return False
    if correction_target == 'all' and any(
        (
            extracted_name and extracted_name != snapshot.visitor_info.name,
            extracted_affiliation and extracted_affiliation != snapshot.visitor_info.affiliation,
            extracted_purpose and extracted_purpose != snapshot.visitor_info.purpose,
        )
    ):
        return False
    if correction_target == 'none' and any(
        value is not None for value in (extracted_name, extracted_affiliation, extracted_purpose)
    ):
        return False
    return any(
        value is not None
        for value in (
            snapshot.visitor_info.name,
            snapshot.visitor_info.affiliation,
            snapshot.visitor_info.purpose,
        )
    )


def _target_fields_for_slot_rescue(
    snapshot: SessionSnapshot,
    *,
    latest_utterance: str,
    speech_act: str,
    slot_confidence: float,
    correction_target: str,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> list[FieldName]:
    targets: list[FieldName] = []
    current = snapshot.visitor_info
    if speech_act in {'deny', 'correction'} and correction_target in {'name', 'affiliation', 'purpose'}:
        target = correction_target
        current_value = getattr(current, target)
        extracted_value = {
            'name': extracted_name,
            'affiliation': extracted_affiliation,
            'purpose': extracted_purpose,
        }[target]
        if not extracted_value or extracted_value == current_value:
            targets.append(target)  # explicit correction but no new usable value yet
        return targets
    if speech_act in {'deny', 'correction'} and correction_target == 'all':
        if current.name and not extracted_name:
            targets.append('name')
        if current.affiliation and not extracted_affiliation:
            targets.append('affiliation')
        if current.purpose and not extracted_purpose:
            targets.append('purpose')
        return targets

    # Explicit deny/correction turns can revise any field; re-extract all slots from the latest utterance.
    if speech_act in {'deny', 'correction'}:
        return ['name', 'affiliation', 'purpose']

    # For normal collection turns, always re-extract currently missing fields from the latest utterance.
    # This keeps rescue aligned with canonical state rather than trusting the primary free-form output.
    if speech_act in {'inform', 'question', 'unknown', 'greeting'}:
        low_confidence = slot_confidence < 0.95
        if not current.name and (not extracted_name or low_confidence):
            targets.append('name')
        if not current.affiliation and (not extracted_affiliation or low_confidence):
            targets.append('affiliation')
        if (
            not current.purpose
            and _utterance_has_purpose_cue(latest_utterance)
            and (not extracted_purpose or low_confidence)
        ):
            targets.append('purpose')
    return targets


def _correction_target_for_fields(fields: list[FieldName]) -> str:
    if not fields:
        return 'none'
    if len(fields) == 1:
        return fields[0]
    return 'all'


def _is_noninformative_slot_value(text: str) -> bool:
    normalized = ''.join(text.split()).replace('。', '').replace('、', '').lower()
    return normalized in {
        'こんにちは',
        'こんばんは',
        'おはようございます',
        'もしもし',
        'すみません',
        'お願いします',
        'はい',
        'いいえ',
    }


def _ground_slot_updates(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    speech_act: str,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> tuple[str | None, str | None, str | None, list[str], list[str]]:
    rejected_fields: list[str] = []
    recovered_fields: list[str] = []

    if (
        extracted_name
        and extracted_name != snapshot.visitor_info.name
        and not _is_grounded_name(extracted_name, latest_utterance)
    ):
        extracted_name = None
        rejected_fields.append('name')
    if (
        extracted_affiliation
        and extracted_affiliation != snapshot.visitor_info.affiliation
        and not _is_grounded_affiliation(extracted_affiliation, latest_utterance)
    ):
        extracted_affiliation = None
        rejected_fields.append('affiliation')
    if (
        extracted_purpose
        and extracted_purpose != snapshot.visitor_info.purpose
        and not _is_grounded_purpose(extracted_purpose, latest_utterance)
    ):
        extracted_purpose = None
        rejected_fields.append('purpose')

    recovered_name = None
    if (
        speech_act in {'inform', 'question', 'unknown', 'greeting', 'correction', 'deny'}
        and not snapshot.visitor_info.name
        and not extracted_name
    ):
        recovered_name = _extract_self_introduction_name(latest_utterance)
    if recovered_name:
        extracted_name = recovered_name
        recovered_fields.append('name')

    recovered_affiliation = None
    if not snapshot.visitor_info.affiliation and (
        not extracted_affiliation or _is_generic_affiliation(extracted_affiliation)
    ):
        recovered_affiliation = _extract_grounded_affiliation(latest_utterance)
    if recovered_affiliation:
        extracted_affiliation = recovered_affiliation
        recovered_fields.append('affiliation')

    recovered_purpose = None
    if not snapshot.visitor_info.purpose and not extracted_purpose:
        recovered_purpose = _extract_grounded_purpose(latest_utterance)
    if recovered_purpose:
        extracted_purpose = recovered_purpose
        recovered_fields.append('purpose')

    return extracted_name, extracted_affiliation, extracted_purpose, rejected_fields, recovered_fields


def _extract_self_introduction_name(latest_utterance: str) -> str | None:
    candidates = [
        re.search(r'(?:私の名前は|名前は)([^。、「」\s]{1,12})です', latest_utterance),
        re.search(r'([^。、「」\s]{1,12})と申します', latest_utterance),
        re.search(r'([^。、「」\s]{1,12})といいます', latest_utterance),
        re.search(r'(?:^|[。 、])([^。、「」\s]{1,12})です(?:$|[。 、])', latest_utterance),
    ]
    for match in candidates:
        if match is None:
            continue
        name = _optional_string(match.group(1))
        if name and _looks_like_person_name(name):
            return name
    return None


def _looks_like_person_name(text: str) -> bool:
    normalized = _normalize_grounding_text(text)
    if not normalized or len(normalized) > 8:
        return False
    if any(token in text for token in _AFFILIATION_TOKENS):
        return False
    if any(token in text for token in _PURPOSE_CUES):
        return False
    if any(char.isdigit() for char in normalized):
        return False
    return True


def _is_grounded_name(candidate: str, latest_utterance: str) -> bool:
    recovered_name = _extract_self_introduction_name(latest_utterance)
    if recovered_name is None:
        return False
    return _normalize_grounding_text(recovered_name) == _normalize_grounding_text(candidate)


def _is_grounded_affiliation(candidate: str, latest_utterance: str) -> bool:
    if _is_generic_affiliation(candidate):
        return False
    if not _candidate_in_utterance(candidate, latest_utterance):
        return False
    if any(token in candidate for token in _AFFILIATION_TOKENS):
        return True
    normalized_utterance = _normalize_grounding_text(latest_utterance)
    normalized_candidate = _normalize_grounding_text(candidate)
    return f'{normalized_candidate}の' in normalized_utterance or '所属' in latest_utterance


def _is_grounded_purpose(candidate: str, latest_utterance: str) -> bool:
    grounded = _extract_grounded_purpose(latest_utterance)
    if not grounded:
        return False
    return _normalize_grounding_text(candidate) == _normalize_grounding_text(grounded)


def _extract_grounded_purpose(latest_utterance: str) -> str | None:
    cleaned = latest_utterance.strip().strip('。！？!?、, ')
    if not cleaned:
        return None
    if not _utterance_has_purpose_cue(cleaned):
        return None
    if '所属' in cleaned and not any(token in cleaned for token in ('用件', '用が', '要が', 'ようが', '面会', '打ち合わせ', '相談', '面談', '訪問', 'の件')):
        return None
    fragments = [fragment.strip('。！？!?、, ') for fragment in re.split(r'[。！？!?]', cleaned)]
    fragments = [fragment for fragment in fragments if fragment]
    for fragment in reversed(fragments):
        purpose = _match_purpose_fragment(fragment)
        if purpose:
            return purpose

    clause_candidates = [fragment.strip('、, ') for fragment in re.split(r'[、,]', cleaned)]
    for fragment in reversed(clause_candidates):
        purpose = _match_purpose_fragment(fragment)
        if purpose:
            return purpose

    purpose = _match_purpose_fragment(cleaned)
    if purpose:
        return purpose
    return None


def _match_purpose_fragment(fragment: str) -> str | None:
    cleaned = fragment.strip().strip('。！？!?、, ')
    if not cleaned or not _utterance_has_purpose_cue(cleaned):
        return None
    if re.search(r'^(?:私の名前は|名前は)?[^。、「」\s]{1,12}です$', cleaned):
        return None
    if re.search(r'.+(?:用件|用が|要が|ようが).*(?:来ました|きました|参りました|まいりました|やってきました)', cleaned):
        return cleaned
    if re.search(r'.+(?:面会|打ち合わせ|相談|面談|訪問|アポイント|書類の件|の件).*(?:来ました|きました|参りました|まいりました|やってきました)?', cleaned):
        return cleaned
    if re.search(r'.+会いに.*(?:来ました|きました|参りました|まいりました|やってきました)', cleaned):
        return cleaned
    return None


def _utterance_has_purpose_cue(latest_utterance: str) -> bool:
    return any(token in latest_utterance for token in _PURPOSE_CUES)


def _candidate_in_utterance(candidate: str, latest_utterance: str) -> bool:
    normalized_candidate = _normalize_grounding_text(candidate)
    normalized_utterance = _normalize_grounding_text(latest_utterance)
    return bool(normalized_candidate and normalized_candidate in normalized_utterance)


def _normalize_grounding_text(text: str) -> str:
    normalized = ''.join(text.split())
    normalized = re.sub(r'[、。,.!！?？「」『』（）()・]', '', normalized)
    normalized = normalized.lower()
    for source, target in (
        ('で来ました', ''),
        ('で参りました', ''),
        ('でまいりました', ''),
        ('できました', ''),
    ):
        normalized = normalized.replace(source, target)
    return normalized


def _is_generic_affiliation(candidate: str) -> bool:
    return _normalize_grounding_text(candidate) in {
        _normalize_grounding_text(token) for token in _GENERIC_AFFILIATIONS
    }


def _extract_grounded_affiliation(latest_utterance: str) -> str | None:
    matches: list[str] = []
    for token in _AFFILIATION_TOKENS:
        pattern = rf'([^。、「」\s]{{1,20}}{re.escape(token)})'
        matches.extend(match.group(1) for match in re.finditer(pattern, latest_utterance))
    for candidate in sorted(matches, key=len, reverse=True):
        affiliation = _optional_string(candidate)
        if affiliation and not _is_generic_affiliation(affiliation):
            return affiliation
    return None
