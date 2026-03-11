from __future__ import annotations

import json
import os
import re
from typing import Callable

from .prompt_templates import RECEPTION_REPAIR_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA
from .prompt_templates import RECEPTION_RESPONSE_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_EXTRACT_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_COMMIT_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_COMMIT_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_FIELD_COMMIT_JSON_SCHEMA
from .prompt_templates import RECEPTION_FIELD_COMMIT_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_SLOT_NORMALIZE_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_NORMALIZE_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_SYSTEM_PROMPT
from .prompt_templates import build_reception_confirmation_rescue_prompt
from .prompt_templates import build_reception_correction_rescue_prompt
from .prompt_templates import build_reception_field_commit_prompt
from .prompt_templates import build_reception_repair_prompt
from .prompt_templates import build_reception_slot_commit_prompt
from .prompt_templates import build_reception_slot_extract_prompt
from .prompt_templates import build_reception_slot_normalize_prompt
from .prompt_templates import build_reception_user_prompt
from .state_models import FieldName
from .state_models import SessionSnapshot
from .state_models import SupervisorDecision
from .state_models import VisitorInfo


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
_IGNORE_SPEECH_ACTS: set[str] = {'greeting', 'unknown'}
_NONINFORMATIVE_NORMALIZED_VALUES: set[str] = {
    'こんにちは',
    'こんばんは',
    'おはようございます',
    'もしもし',
    'すみません',
    'よろしくお願いします',
    'よろしくおねがいします',
    'お願いします',
    'はい',
    'いいえ',
    '未知',
    '未知の先生',
    '未知の研究所',
    '来訪理由不明',
    'selfintroduction',
    'null',
}
_FILLER_PREFIXES: tuple[str, ...] = (
    'えっと',
    'えーと',
    'ええと',
    'あの',
    'その',
    'まあ',
)
_MEETING_TARGET_SUFFIXES: tuple[str, ...] = (
    'に会い',
    'へ会い',
    'と会い',
    'にお会い',
    'とお会い',
    'への面会',
    'との面会',
    'に面会',
    'と面会',
    '宛',
)
_PURPOSE_MARKERS: tuple[str, ...] = (
    '会い',
    '面会',
    '相談',
    '打ち合わせ',
    '提出',
    '届け',
    '来ました',
    '参りました',
    '訪問',
    '用件',
)
_AFFILIATION_MARKERS: tuple[str, ...] = (
    '研究室',
    '学科',
    '会社',
    '大学',
    '学校',
    '所属',
    '部',
    '課',
    '室',
)
_PURPOSE_TRAILING_PHRASES: tuple[str, ...] = (
    'でございます',
    'になります',
    'となります',
    'していました',
    'しています',
    'してます',
    'いたします',
    'ませんでした',
    'でしたら',
    'でしたか',
    'ましたか',
    'しました',
    'でした',
    'ました',
    'ません',
    'です',
    'ます',
    'だ',
    'た',
)


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
                    min(self._max_tokens, 96),
                    True,
                    RECEPTION_RESPONSE_JSON_SCHEMA,
                )
                payload = _extract_json_object(repaired)
            except Exception as exc:
                self._trace(f'supervisor_repair_failed error={exc}')
                return fallback

        if not _has_usable_payload(payload):
            self._trace('supervisor_repair_unusable')
            confirmation_rescue = self._recover_confirmation_from_unusable_turn(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
            )
            if confirmation_rescue is not None:
                return confirmation_rescue
            rescued = self._recover_slots_from_unusable_turn(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
            )
            if rescued is not None:
                return rescued
            return fallback

        assert isinstance(payload, dict)
        slot_candidates_payload = payload.get('slot_candidates')
        if not isinstance(slot_candidates_payload, dict):
            slot_candidates_payload = {}
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
        candidate_name = _optional_string(slot_candidates_payload.get('name'))
        candidate_affiliation = _optional_string(slot_candidates_payload.get('affiliation'))
        candidate_purpose = _optional_string(slot_candidates_payload.get('purpose'))
        extracted_name = _optional_string(updates.get('name')) or candidate_name
        extracted_affiliation = _optional_string(updates.get('affiliation')) or candidate_affiliation
        extracted_purpose = _optional_string(updates.get('purpose')) or candidate_purpose
        correction_target = str(correction.get('target', 'none')).strip()
        correction_scope = str(payload.get('correction_scope', correction_target)).strip()
        overwrite = bool(correction.get('overwrite', False))
        spoken_response = _normalize_spoken_response(payload.get('spoken_response'))
        ignore_input = bool(payload.get('ignore_input', False))
        slot_confidence = _optional_float(payload.get('confidence'))
        self._trace(
            'supervisor_primary_result '
            f'speech_act={speech_act} '
            f'candidate_name={candidate_name or "-"} '
            f'candidate_affiliation={candidate_affiliation or "-"} '
            f'candidate_purpose={candidate_purpose or "-"} '
            f'extracted_name={extracted_name or "-"} '
            f'extracted_affiliation={extracted_affiliation or "-"} '
            f'extracted_purpose={extracted_purpose or "-"}'
        )
        if speech_act not in _VALID_SPEECH_ACTS:
            speech_act = 'unknown'
        if correction_target not in _VALID_CORRECTIONS:
            correction_target = 'none'
        if correction_scope not in _VALID_CORRECTIONS:
            correction_scope = correction_target
        if snapshot.phase == 'confirming' and speech_act != 'affirm':
            confirmation_rescue = self._recover_confirmation_from_unusable_turn(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
            )
            if confirmation_rescue is not None and confirmation_rescue.speech_act == 'affirm':
                return confirmation_rescue
        if speech_act not in {'deny', 'correction'}:
            correction_target = 'none'
            overwrite = False
        if snapshot.phase == 'confirming' and speech_act != 'affirm':
            changed_fields = _infer_corrected_fields(
                snapshot,
                extracted_name=extracted_name,
                extracted_affiliation=extracted_affiliation,
                extracted_purpose=extracted_purpose,
            )
            if changed_fields:
                correction_target = _correction_target_for_fields(changed_fields)
                correction_scope = correction_target
                overwrite = True
        if (
            snapshot.phase == 'collecting'
            and speech_act in _IGNORE_SPEECH_ACTS
            and not any((extracted_name, extracted_affiliation, extracted_purpose))
        ):
            ignore_input = True
        if (
            snapshot.phase == 'collecting'
            and _is_noninformative_utterance(latest_utterance)
            and not any((extracted_name, extracted_affiliation, extracted_purpose))
        ):
            ignore_input = True

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
                    min(self._max_tokens, 96),
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
            ignore_input=ignore_input,
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
                    min(self._max_tokens, 96),
                    True,
                    RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
                )
                slot_payload = _extract_json_object(rescue_raw)
                if isinstance(slot_payload, dict):
                    if 'name' in rescue_fields:
                        extracted_name = _optional_string(slot_payload.get('name')) or extracted_name
                    if 'affiliation' in rescue_fields:
                        extracted_affiliation = (
                            _optional_string(slot_payload.get('affiliation')) or extracted_affiliation
                        )
                    if 'purpose' in rescue_fields:
                        extracted_purpose = _optional_string(slot_payload.get('purpose')) or extracted_purpose
            except Exception:
                pass

        extracted_name, extracted_affiliation, extracted_purpose = self._semantic_normalize_slots(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )

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
        self._trace(
            'supervisor_post_grounding '
            f'name={extracted_name or "-"} '
            f'affiliation={extracted_affiliation or "-"} '
            f'purpose={extracted_purpose or "-"} '
            f'rejected={",".join(rejected_fields) or "none"}'
        )
        if rejected_fields:
            extracted_name, extracted_affiliation, extracted_purpose = self._recover_rejected_slots(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                rejected_fields=rejected_fields,
                primary_field=_primary_field_for_snapshot(snapshot),
                extracted_name=extracted_name,
                extracted_affiliation=extracted_affiliation,
                extracted_purpose=extracted_purpose,
            )
            spoken_response = None

        extracted_name, extracted_affiliation, extracted_purpose = self._apply_turn_commit_policy(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            speech_act=speech_act,
            correction_target=correction_target,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        extracted_name, extracted_affiliation, extracted_purpose = self._refine_committed_slots(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        self._trace(
            'supervisor_post_commit '
            f'name={extracted_name or "-"} '
            f'affiliation={extracted_affiliation or "-"} '
            f'purpose={extracted_purpose or "-"}'
        )

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
                    min(self._max_tokens, 96),
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
        if accepted and not (
            snapshot.phase == 'confirming'
            and (
                changed_fields
                or correction_target != 'none'
                or any((extracted_name, extracted_affiliation, extracted_purpose))
            )
        ):
            speech_act = 'affirm'
        if ignore_input:
            should_confirm = False
            correction_target = 'none'

        return SupervisorDecision(
            speech_act=speech_act,
            slot_candidates=VisitorInfo(
                name=candidate_name,
                affiliation=candidate_affiliation,
                purpose=candidate_purpose,
            ),
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
            slot_confidence=slot_confidence,
            missing_fields=missing_fields,
            next_dialog_act=None,
            should_confirm=should_confirm,
            correction_scope=correction_scope,
            correction_target=correction_target if speech_act == 'correction' else 'none',
            discord_update_kind='none',
            ignore_input=ignore_input,
            spoken_response=spoken_response,
        )

    def _recover_slots_from_unusable_turn(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
    ) -> SupervisorDecision | None:
        if snapshot.phase == 'confirming':
            return None
        else:
            primary_field = _primary_field_for_snapshot(snapshot)
            if primary_field == 'name':
                target_fields = ['name', 'affiliation', 'purpose']
            elif primary_field == 'affiliation':
                target_fields = ['affiliation']
                if _utterance_likely_contains_purpose(latest_utterance):
                    target_fields.append('purpose')
            elif primary_field == 'purpose':
                target_fields = ['purpose']
                if _utterance_likely_contains_affiliation(latest_utterance):
                    target_fields.append('affiliation')
            else:
                missing_fields = snapshot.visitor_info.missing_fields()
                target_fields = missing_fields[:1]
        if not target_fields:
            return None
        self._trace(f'supervisor_unusable_slot_recovery fields={",".join(target_fields)}')
        try:
            rescue_raw = self._invoke_chat(
                f'{request_session_id}:unusable-slot-extract',
                build_reception_slot_extract_prompt(
                    snapshot,
                    latest_utterance,
                    target_fields=target_fields,
                ),
                RECEPTION_REPAIR_SYSTEM_PROMPT,
                0.0,
                max(64, min(self._max_tokens, 96)),
                True,
                RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
            )
            slot_payload = _extract_json_object(rescue_raw)
        except Exception:
            self._trace('supervisor_unusable_slot_recovery_failed')
            return None
        if not isinstance(slot_payload, dict):
            self._trace('supervisor_unusable_slot_recovery_empty')
            slot_payload = {}

        extracted_name = _optional_string(slot_payload.get('name'))
        extracted_affiliation = _optional_string(slot_payload.get('affiliation'))
        extracted_purpose = _optional_string(slot_payload.get('purpose'))

        missing_after_combined: list[FieldName] = []
        if 'name' in target_fields and not extracted_name:
            missing_after_combined.append('name')
        if 'affiliation' in target_fields and not extracted_affiliation:
            missing_after_combined.append('affiliation')
        if 'purpose' in target_fields and not extracted_purpose:
            missing_after_combined.append('purpose')

        for field_name in missing_after_combined:
            self._trace(f'supervisor_unusable_slot_recovery_retry field={field_name}')
            recovered = self._recover_single_slot(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_field=field_name,
            )
            if not recovered:
                continue
            if field_name == 'name':
                extracted_name = recovered
            elif field_name == 'affiliation':
                extracted_affiliation = recovered
            else:
                extracted_purpose = recovered

        if len(target_fields) > 1:
            extracted_name, extracted_affiliation, extracted_purpose = self._refine_combined_slot_recovery(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_fields=target_fields,
                extracted_name=extracted_name,
                extracted_affiliation=extracted_affiliation,
                extracted_purpose=extracted_purpose,
            )

        extracted_name, extracted_affiliation, extracted_purpose = self._semantic_normalize_slots(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )

        extracted_name, extracted_affiliation, extracted_purpose = _sanitize_slot_updates(
            snapshot,
            speech_act='inform',
            correction_target='none',
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        (
            extracted_name,
            extracted_affiliation,
            extracted_purpose,
            rejected_fields,
        ) = _ground_slot_updates(
            snapshot,
            latest_utterance,
            speech_act='inform',
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        if rejected_fields:
            extracted_name, extracted_affiliation, extracted_purpose = self._recover_rejected_slots(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                rejected_fields=rejected_fields,
                primary_field=_primary_field_for_snapshot(snapshot),
                extracted_name=extracted_name,
                extracted_affiliation=extracted_affiliation,
                extracted_purpose=extracted_purpose,
            )
        extracted_name, extracted_affiliation, extracted_purpose = self._apply_turn_commit_policy(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            speech_act='inform',
            correction_target='none',
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
        )
        self._trace(
            'supervisor_unusable_slot_recovery_result '
            f'name={extracted_name or "-"} '
            f'affiliation={extracted_affiliation or "-"} '
            f'purpose={extracted_purpose or "-"} '
            f'rejected={",".join(rejected_fields) or "none"}'
        )

        if not any((extracted_name, extracted_affiliation, extracted_purpose)):
            return None

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

        return SupervisorDecision(
            speech_act='inform',
            slot_candidates=VisitorInfo(
                name=extracted_name,
                affiliation=extracted_affiliation,
                purpose=extracted_purpose,
            ),
            extracted_name=extracted_name,
            extracted_affiliation=extracted_affiliation,
            extracted_purpose=extracted_purpose,
            slot_confidence=0.0,
            missing_fields=missing_fields,
            next_dialog_act=None,
            should_confirm=not missing_fields,
            correction_scope='none',
            correction_target='none',
            discord_update_kind='none',
            ignore_input=False,
            spoken_response=None,
        )

    def _recover_confirmation_from_unusable_turn(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
    ) -> SupervisorDecision | None:
        if snapshot.phase != 'confirming':
            return None
        self._trace('supervisor_unusable_confirmation_recovery')
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
        except Exception:
            return None
        if not isinstance(rescue_payload, dict):
            return None

        speech_act = str(rescue_payload.get('speech_act', 'unknown')).strip()
        if speech_act not in {'affirm', 'deny', 'correction', 'unknown'}:
            speech_act = 'unknown'
        correction = rescue_payload.get('correction')
        if not isinstance(correction, dict):
            correction = {}
        confirmation = rescue_payload.get('confirmation')
        if not isinstance(confirmation, dict):
            confirmation = {}

        current = snapshot.visitor_info
        missing_fields: list[FieldName] = []
        if not current.name:
            missing_fields.append('name')
        if not current.affiliation:
            missing_fields.append('affiliation')
        if not current.purpose:
            missing_fields.append('purpose')

        should_confirm = bool(confirmation.get('ready', False) and not missing_fields)
        accepted = bool(confirmation.get('accepted', False) and not missing_fields)
        if accepted:
            speech_act = 'affirm'

        correction_target = str(correction.get('target', 'none')).strip()
        if correction_target not in _VALID_CORRECTIONS:
            correction_target = 'none'
        if speech_act not in {'deny', 'correction'}:
            correction_target = 'none'

        if speech_act == 'unknown' and correction_target == 'none' and not accepted:
            return None

        return SupervisorDecision(
            speech_act=speech_act,
            slot_candidates=VisitorInfo(),
            extracted_name=None,
            extracted_affiliation=None,
            extracted_purpose=None,
            slot_confidence=0.0,
            missing_fields=missing_fields,
            next_dialog_act=None,
            should_confirm=should_confirm,
            correction_scope=correction_target,
            correction_target=correction_target if speech_act == 'correction' else 'none',
            discord_update_kind='none',
            ignore_input=False,
            spoken_response=None,
        )

    def _refine_combined_slot_recovery(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        target_fields: list[FieldName],
        extracted_name: str | None,
        extracted_affiliation: str | None,
        extracted_purpose: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        values = {
            'name': extracted_name,
            'affiliation': extracted_affiliation,
            'purpose': extracted_purpose,
        }
        for field_name in target_fields:
            current_value = values[field_name]
            if not current_value:
                continue
            recovered = self._recover_single_slot(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_field=field_name,
            )
            if _prefer_more_specific_slot_value(current_value, recovered):
                self._trace(f'supervisor_unusable_slot_refine field={field_name}')
                values[field_name] = recovered
        return values['name'], values['affiliation'], values['purpose']

    def _recover_single_slot(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        target_field: FieldName,
    ) -> str | None:
        try:
            rescue_raw = self._invoke_chat(
                f'{request_session_id}:unusable-slot-extract:{target_field}',
                build_reception_slot_extract_prompt(
                    snapshot,
                    latest_utterance,
                    target_fields=[target_field],
                ),
                RECEPTION_REPAIR_SYSTEM_PROMPT,
                0.0,
                min(self._max_tokens, 64),
                True,
                RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
            )
            slot_payload = _extract_json_object(rescue_raw)
        except Exception:
            return None
        if not isinstance(slot_payload, dict):
            return None
        return _optional_string(slot_payload.get(target_field))

    def _recover_rejected_slots(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        rejected_fields: list[str],
        primary_field: FieldName | None,
        extracted_name: str | None,
        extracted_affiliation: str | None,
        extracted_purpose: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        recovered = {
            'name': extracted_name,
            'affiliation': extracted_affiliation,
            'purpose': extracted_purpose,
        }
        for field_name in rejected_fields:
            if field_name not in {'name', 'affiliation', 'purpose'}:
                continue
            if snapshot.phase != 'confirming' and primary_field is not None and field_name != primary_field:
                continue
            if recovered[field_name] is not None:
                continue
            slot_value = self._recover_single_slot(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_field=field_name,
            )
            if slot_value is None:
                continue
            self._trace(f'slot_recovered field={field_name} utterance={_truncate(latest_utterance, 80)}')
            recovered[field_name] = slot_value

        (
            recovered['name'],
            recovered['affiliation'],
            recovered['purpose'],
            _,
        ) = _ground_slot_updates(
            snapshot,
            latest_utterance,
            speech_act='inform',
            extracted_name=recovered['name'],
            extracted_affiliation=recovered['affiliation'],
            extracted_purpose=recovered['purpose'],
        )
        return recovered['name'], recovered['affiliation'], recovered['purpose']

    def _semantic_normalize_slots(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        extracted_name: str | None,
        extracted_affiliation: str | None,
        extracted_purpose: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        populated_fields = sum(
            value is not None for value in (extracted_name, extracted_affiliation, extracted_purpose)
        )
        if extracted_purpose is None and populated_fields <= 1:
            return extracted_name, extracted_affiliation, extracted_purpose
        if not any((extracted_name, extracted_affiliation, extracted_purpose)):
            return extracted_name, extracted_affiliation, extracted_purpose
        try:
            raw = self._invoke_chat(
                f'{request_session_id}:slot-normalize',
                build_reception_slot_normalize_prompt(
                    snapshot,
                    latest_utterance,
                    extracted_name=extracted_name,
                    extracted_affiliation=extracted_affiliation,
                    extracted_purpose=extracted_purpose,
                ),
                RECEPTION_SLOT_NORMALIZE_SYSTEM_PROMPT,
                0.0,
                min(self._max_tokens, 96),
                True,
                RECEPTION_SLOT_NORMALIZE_JSON_SCHEMA,
            )
            payload = _extract_json_object(raw)
        except Exception:
            return extracted_name, extracted_affiliation, extracted_purpose
        if not isinstance(payload, dict):
            return extracted_name, extracted_affiliation, extracted_purpose
        if 'name' in payload:
            extracted_name = _optional_string(payload.get('name'))
        if 'affiliation' in payload:
            extracted_affiliation = _optional_string(payload.get('affiliation'))
        if 'purpose' in payload:
            extracted_purpose = _optional_string(payload.get('purpose'))
        return extracted_name, extracted_affiliation, extracted_purpose

    def _apply_turn_commit_policy(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        speech_act: str,
        correction_target: str,
        extracted_name: str | None,
        extracted_affiliation: str | None,
        extracted_purpose: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        if speech_act in {'deny', 'correction'} or correction_target != 'none':
            return extracted_name, extracted_affiliation, extracted_purpose
        primary_field = _primary_field_for_snapshot(snapshot)
        if snapshot.phase not in {'collecting', 'confirming'}:
            return extracted_name, extracted_affiliation, extracted_purpose
        if not any((extracted_name, extracted_affiliation, extracted_purpose)):
            return extracted_name, extracted_affiliation, extracted_purpose
        if snapshot.phase == 'collecting' and speech_act == 'affirm':
            return None, None, None
        try:
            raw = self._invoke_chat(
                f'{request_session_id}:slot-commit',
                build_reception_slot_commit_prompt(
                    snapshot,
                    latest_utterance,
                    primary_field=primary_field,
                    extracted_name=extracted_name,
                    extracted_affiliation=extracted_affiliation,
                    extracted_purpose=extracted_purpose,
                ),
                RECEPTION_SLOT_COMMIT_SYSTEM_PROMPT,
                0.0,
                min(self._max_tokens, 96),
                True,
                RECEPTION_SLOT_COMMIT_JSON_SCHEMA,
            )
            payload = _extract_json_object(raw)
        except Exception:
            payload = None
        committed_name = extracted_name
        committed_affiliation = extracted_affiliation
        committed_purpose = extracted_purpose
        if isinstance(payload, dict):
            committed_name = _optional_string(payload.get('name')) if 'name' in payload else extracted_name
            committed_affiliation = (
                _optional_string(payload.get('affiliation')) if 'affiliation' in payload else extracted_affiliation
            )
            committed_purpose = _optional_string(payload.get('purpose')) if 'purpose' in payload else extracted_purpose

        validated: dict[FieldName, str | None] = {
            'name': committed_name,
            'affiliation': committed_affiliation,
            'purpose': committed_purpose,
        }
        for field_name, value in tuple(validated.items()):
            if value is None:
                continue
            field_primary = primary_field == field_name
            if (
                snapshot.phase == 'collecting'
                and not field_primary
                and field_name == 'purpose'
                and primary_field != 'purpose'
            ):
                refined = self._commit_single_field(
                    snapshot,
                    latest_utterance,
                    request_session_id=request_session_id,
                    primary_field=primary_field,
                    target_field=field_name,
                    candidate_value=value,
                )
                validated[field_name] = refined
                continue
            validated[field_name] = self._commit_single_field(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                primary_field=primary_field,
                target_field=field_name,
                candidate_value=value,
            )
        should_double_check = snapshot.phase == 'collecting' and (
            sum(value is not None for value in validated.values()) > 1
            or any(
                field_name != primary_field and value is not None
                for field_name, value in validated.items()
            )
        )
        if should_double_check:
            validated['name'] = self._confirm_committed_field_via_slot_extract(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_field='name',
                candidate_value=validated['name'],
            )
            validated['affiliation'] = self._confirm_committed_field_via_slot_extract(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_field='affiliation',
                candidate_value=validated['affiliation'],
            )
            validated['purpose'] = self._confirm_committed_field_via_slot_extract(
                snapshot,
                latest_utterance,
                request_session_id=request_session_id,
                target_field='purpose',
                candidate_value=validated['purpose'],
            )
        return validated['name'], validated['affiliation'], validated['purpose']

    def _commit_single_field(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        primary_field: FieldName | None,
        target_field: FieldName,
        candidate_value: str | None,
    ) -> str | None:
        if candidate_value is None:
            return None
        try:
            raw = self._invoke_chat(
                f'{request_session_id}:field-commit:{target_field}',
                build_reception_field_commit_prompt(
                    snapshot,
                    latest_utterance,
                    primary_field=primary_field,
                    target_field=target_field,
                    candidate_value=candidate_value,
                ),
                RECEPTION_FIELD_COMMIT_SYSTEM_PROMPT,
                0.0,
                min(self._max_tokens, 96),
                True,
                RECEPTION_FIELD_COMMIT_JSON_SCHEMA,
            )
            payload = _extract_json_object(raw)
        except Exception:
            return candidate_value
        if not isinstance(payload, dict):
            return candidate_value
        if 'value' not in payload:
            return candidate_value
        return _optional_string(payload.get('value'))

    def _confirm_committed_field_via_slot_extract(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        target_field: FieldName,
        candidate_value: str | None,
    ) -> str | None:
        if candidate_value is None:
            return None
        recovered = self._recover_single_slot(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            target_field=target_field,
        )
        if recovered is None:
            return candidate_value
        if _normalize_grounding_text(recovered) != _normalize_grounding_text(candidate_value):
            if _prefer_more_specific_slot_value(candidate_value, recovered):
                return recovered
            if _prefer_more_specific_slot_value(recovered, candidate_value):
                return candidate_value
        return recovered

    def _refine_committed_slots(
        self,
        snapshot: SessionSnapshot,
        latest_utterance: str,
        *,
        request_session_id: str,
        extracted_name: str | None,
        extracted_affiliation: str | None,
        extracted_purpose: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        if not extracted_purpose:
            return extracted_name, extracted_affiliation, extracted_purpose
        if snapshot.phase == 'confirming':
            return extracted_name, extracted_affiliation, extracted_purpose
        primary_field = _primary_field_for_snapshot(snapshot)
        if primary_field not in {'name', 'purpose'}:
            return extracted_name, extracted_affiliation, extracted_purpose
        recovered = self._recover_single_slot(
            snapshot,
            latest_utterance,
            request_session_id=request_session_id,
            target_field='purpose',
        )
        if _prefer_more_specific_slot_value(extracted_purpose, recovered):
            return extracted_name, extracted_affiliation, recovered
        return extracted_name, extracted_affiliation, extracted_purpose


def _fallback_decision(snapshot: SessionSnapshot) -> SupervisorDecision:
    return SupervisorDecision(missing_fields=snapshot.visitor_info.missing_fields())


def _primary_field_for_snapshot(snapshot: SessionSnapshot) -> FieldName | None:
    if snapshot.last_dialog_act == 'ask_name':
        return 'name'
    if snapshot.last_dialog_act == 'ask_affiliation':
        return 'affiliation'
    if snapshot.last_dialog_act == 'ask_purpose':
        return 'purpose'
    missing = snapshot.visitor_info.missing_fields()
    return missing[0] if missing else None


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
        key in payload for key in ('speech_act', 'slot_candidates', 'slot_updates', 'confirmation', 'spoken_response')
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
    correction_mode = correction_target != 'none' or speech_act in {'deny', 'correction'}

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
    ignore_input: bool,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> list[FieldName]:
    targets: list[FieldName] = []
    current = snapshot.visitor_info
    if ignore_input:
        return targets
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

    # For normal turns, keep rescue focused on the currently asked field.
    # Multi-slot updates are allowed only when the primary extraction already
    # found them; rescue should not manufacture secondary slots.
    if speech_act in {'inform', 'question', 'unknown', 'greeting'}:
        if snapshot.phase == 'confirming':
            return []
        primary_field = _primary_field_for_snapshot(snapshot)
        low_confidence = slot_confidence < 0.95
        if primary_field == 'name':
            if (
                not any((current.name, current.affiliation, current.purpose))
                and not any((extracted_name, extracted_affiliation, extracted_purpose))
            ):
                targets.extend(['name', 'affiliation', 'purpose'])
            elif not current.name and (
                not extracted_name
                or not _is_grounded_slot_value(extracted_name, latest_utterance)
            ):
                targets.append('name')
        elif primary_field == 'affiliation':
            if not current.affiliation and (
                not extracted_affiliation
                or low_confidence
                or not _is_grounded_slot_value(extracted_affiliation, latest_utterance)
            ):
                targets.append('affiliation')
            if (
                not current.purpose
                and _utterance_likely_contains_purpose(latest_utterance)
                and (
                    not extracted_purpose
                    or low_confidence
                    or not _is_grounded_slot_value(extracted_purpose, latest_utterance)
                )
            ):
                targets.append('purpose')
        elif primary_field == 'purpose':
            if not current.purpose and (
                not extracted_purpose
                or low_confidence
                or not _is_grounded_slot_value(extracted_purpose, latest_utterance)
            ):
                targets.append('purpose')
            if (
                not current.affiliation
                and _utterance_likely_contains_affiliation(latest_utterance)
                and (
                    not extracted_affiliation
                    or low_confidence
                    or not _is_grounded_slot_value(extracted_affiliation, latest_utterance)
                )
            ):
                targets.append('affiliation')
        else:
            if not current.name and (not extracted_name or low_confidence):
                targets.append('name')
    return targets


def _correction_target_for_fields(fields: list[FieldName]) -> str:
    if not fields:
        return 'none'
    if len(fields) == 1:
        return fields[0]
    return 'all'


def _is_noninformative_slot_value(text: str) -> bool:
    normalized = _strip_filler_prefixes(_normalize_grounding_text(text))
    return normalized in _NONINFORMATIVE_NORMALIZED_VALUES


def _is_noninformative_utterance(text: str) -> bool:
    normalized = _strip_filler_prefixes(_normalize_grounding_text(text))
    return normalized in _NONINFORMATIVE_NORMALIZED_VALUES


def _ground_slot_updates(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    speech_act: str,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> tuple[str | None, str | None, str | None, list[str]]:
    rejected_fields: list[str] = []

    if (
        extracted_name
        and extracted_name != snapshot.visitor_info.name
        and not _is_grounded_slot_value(extracted_name, latest_utterance, field_name='name')
    ):
        extracted_name = None
        rejected_fields.append('name')
    if (
        extracted_affiliation
        and extracted_affiliation != snapshot.visitor_info.affiliation
        and not _is_grounded_slot_value(
            extracted_affiliation,
            latest_utterance,
            field_name='affiliation',
        )
    ):
        extracted_affiliation = None
        rejected_fields.append('affiliation')
    if (
        extracted_purpose
        and extracted_purpose != snapshot.visitor_info.purpose
        and not _is_grounded_slot_value(extracted_purpose, latest_utterance, field_name='purpose')
    ):
        extracted_purpose = None
        rejected_fields.append('purpose')

    duplicate_slots = _find_duplicate_slot_values(
        extracted_name=extracted_name,
        extracted_affiliation=extracted_affiliation,
        extracted_purpose=extracted_purpose,
    )
    for field_name in duplicate_slots:
        if field_name == 'name':
            extracted_name = None
        elif field_name == 'affiliation':
            extracted_affiliation = None
        elif field_name == 'purpose':
            extracted_purpose = None
        if field_name not in rejected_fields:
            rejected_fields.append(field_name)

    return extracted_name, extracted_affiliation, extracted_purpose, rejected_fields


def _is_grounded_slot_value(
    candidate: str,
    latest_utterance: str,
    *,
    field_name: FieldName | None = None,
) -> bool:
    normalized_candidate = _normalize_grounding_text(candidate)
    normalized_utterance = _normalize_grounding_text(latest_utterance)
    if not normalized_candidate or not normalized_utterance:
        return False
    if field_name == 'name' and _looks_like_meeting_target_name(normalized_candidate, normalized_utterance):
        return False
    if field_name == 'purpose' and _is_noninformative_utterance(latest_utterance):
        return False
    if field_name == 'purpose' and _is_grounded_purpose_value(normalized_candidate, normalized_utterance):
        return True
    return normalized_candidate in normalized_utterance


def _normalize_grounding_text(text: str) -> str:
    normalized = ''.join(text.split())
    normalized = re.sub(r'[、。,.!！?？「」『』（）()・]', '', normalized)
    normalized = normalized.lower()
    return normalized


def _strip_filler_prefixes(text: str) -> str:
    stripped = text
    changed = True
    while changed and stripped:
        changed = False
        for prefix in _FILLER_PREFIXES:
            normalized_prefix = _normalize_grounding_text(prefix)
            if stripped.startswith(normalized_prefix):
                stripped = stripped[len(normalized_prefix):]
                changed = True
    return stripped


def _looks_like_meeting_target_name(normalized_candidate: str, normalized_utterance: str) -> bool:
    return any(
        f'{normalized_candidate}{suffix}' in normalized_utterance
        for suffix in _MEETING_TARGET_SUFFIXES
    )


def _utterance_likely_contains_purpose(latest_utterance: str) -> bool:
    normalized = _normalize_grounding_text(latest_utterance)
    return any(marker in normalized for marker in _PURPOSE_MARKERS)


def _utterance_likely_contains_affiliation(latest_utterance: str) -> bool:
    normalized = _normalize_grounding_text(latest_utterance)
    return any(marker in normalized for marker in _AFFILIATION_MARKERS)


def _is_grounded_purpose_value(normalized_candidate: str, normalized_utterance: str) -> bool:
    if normalized_candidate in normalized_utterance:
        return True
    canonical_candidate = _strip_purpose_trailing_phrases(normalized_candidate)
    canonical_utterance = _strip_purpose_trailing_phrases(normalized_utterance)
    if not canonical_candidate or not canonical_utterance:
        return False
    if canonical_candidate in canonical_utterance:
        return True
    common_prefix_len = len(os.path.commonprefix((canonical_candidate, canonical_utterance)))
    min_len = min(len(canonical_candidate), len(canonical_utterance))
    return min_len >= 6 and common_prefix_len >= min_len - 1


def _strip_purpose_trailing_phrases(text: str) -> str:
    stripped = text
    changed = True
    while changed and stripped:
        changed = False
        for suffix in _PURPOSE_TRAILING_PHRASES:
            normalized_suffix = _normalize_grounding_text(suffix)
            if stripped.endswith(normalized_suffix) and len(stripped) > len(normalized_suffix):
                stripped = stripped[: -len(normalized_suffix)]
                changed = True
                break
    return stripped


def _find_duplicate_slot_values(
    *,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> list[str]:
    normalized_values = {
        'name': _normalize_grounding_text(extracted_name or ''),
        'affiliation': _normalize_grounding_text(extracted_affiliation or ''),
        'purpose': _normalize_grounding_text(extracted_purpose or ''),
    }
    duplicates: list[str] = []
    seen: dict[str, str] = {}
    for field_name in ('name', 'affiliation', 'purpose'):
        normalized = normalized_values[field_name]
        if not normalized:
            continue
        other = seen.get(normalized)
        if other is None:
            seen[normalized] = field_name
            continue
        if other not in duplicates:
            duplicates.append(other)
        if field_name not in duplicates:
            duplicates.append(field_name)
    return duplicates


def _prefer_more_specific_slot_value(current_value: str, recovered_value: str | None) -> bool:
    if not recovered_value:
        return False
    normalized_current = _normalize_grounding_text(current_value)
    normalized_recovered = _normalize_grounding_text(recovered_value)
    if not normalized_current or not normalized_recovered:
        return False
    if normalized_current == normalized_recovered:
        return False
    if normalized_current in normalized_recovered and len(normalized_recovered) > len(normalized_current):
        return True
    return len(normalized_recovered) > len(normalized_current) + 4
