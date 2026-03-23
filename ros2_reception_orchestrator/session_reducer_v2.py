from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from uuid import uuid4

from .v2_types import BeliefOperationData
from .v2_types import ChatOutboxItemData
from .v2_types import DialogAct
from .v2_types import ReducerOutcomeData
from .v2_types import SecretaryReplyData
from .v2_types import SemanticDecisionData
from .v2_types import SessionStateData
from .v2_types import SlotProvenanceData
from .v2_types import TraceEventData
from .v2_types import VisitorInfoData


_VALID_OPS = {
    'set_slot',
    'replace_slot',
    'clear_slot',
    'confirm_working_state',
    'reject_confirmation',
    'request_clarification',
    'ignore',
}
_VALID_SLOTS = {'name', 'affiliation', 'purpose'}


class SessionReducer:
    """Pure deterministic reducer for the belief-operation reception flow."""

    def __init__(self, *, confidence_threshold: float = 0.55) -> None:
        self._confidence_threshold = float(confidence_threshold)
        self._state: SessionStateData | None = None

    @property
    def state(self) -> SessionStateData:
        if self._state is None:
            self._state = SessionStateData(session_id=uuid4().hex)
        return self._state

    def reset(self) -> None:
        self._state = SessionStateData(session_id=uuid4().hex)

    def apply(
        self,
        *,
        turn_seq: int,
        utterance_text: str,
        decision: SemanticDecisionData,
    ) -> ReducerOutcomeData:
        state = self.state
        state.touch()

        if turn_seq <= state.latest_applied_turn:
            return ReducerOutcomeData(
                session_id=state.session_id,
                turn_seq=turn_seq,
                dialog_act='retry',
                should_render_response=False,
            )

        self._update_response_language(state, decision)
        trace_events = [
            TraceEventData(
                event_type='TURN_PARSED',
                payload_json=json.dumps(
                    {
                        'speech_act': decision.speech_act,
                        'detected_language': decision.detected_language,
                        'target_slot': decision.target_slot,
                        'ambiguity': decision.ambiguity,
                        'requires_confirmation': bool(decision.requires_confirmation),
                        'confidence': float(decision.confidence),
                        'evidence': decision.evidence,
                        'grounded_segments': list(decision.grounded_segments),
                    },
                    ensure_ascii=False,
                ),
            )
        ]

        operations, rejected_reason = self._prepare_operations(state, decision, utterance_text=utterance_text)
        trace_events.append(
            TraceEventData(
                event_type='OPERATIONS_PROPOSED',
                payload_json=json.dumps(
                    {
                        'operations': [self._op_to_payload(op) for op in operations],
                        'rejected_reason': rejected_reason,
                    },
                    ensure_ascii=False,
                ),
            )
        )

        applied_operations: list[BeliefOperationData] = []
        outbox_items: list[ChatOutboxItemData] = []
        requested_clarification_slot = ''
        snapshot_confirmed = False
        working_changed = False

        for operation in operations:
            effective_confidence = max(float(decision.confidence), float(operation.confidence or 0.0))
            if operation.op in {'set_slot', 'replace_slot', 'clear_slot'} and effective_confidence < self._confidence_threshold:
                requested_clarification_slot = requested_clarification_slot or self._preferred_slot(state, decision)
                continue

            if operation.op in {'set_slot', 'replace_slot'}:
                slot = operation.normalized_slot()
                value = str(operation.value or '').strip()
                if slot not in _VALID_SLOTS or not value:
                    continue
                if state.pending_clarification_slot and slot != state.pending_clarification_slot:
                    requested_clarification_slot = state.pending_clarification_slot
                    continue
                current = getattr(state.working_info, slot)
                if current == value:
                    state.pending_clarification_slot = ''
                    continue
                setattr(state.working_info, slot, value)
                state.working_provenance[slot] = SlotProvenanceData(
                    slot=slot,
                    source_turn_seq=turn_seq,
                    grounded_text=str(operation.grounded_text or value),
                    confidence=effective_confidence,
                    updated_at=_utcnow().isoformat(timespec='seconds'),
                )
                state.pending_clarification_slot = ''
                working_changed = True
                applied_operations.append(operation)
                continue

            if operation.op == 'clear_slot':
                slot = operation.normalized_slot()
                if slot not in _VALID_SLOTS:
                    continue
                if getattr(state.working_info, slot):
                    setattr(state.working_info, slot, '')
                    state.working_provenance.pop(slot, None)
                    working_changed = True
                    applied_operations.append(operation)
                continue

            if operation.op == 'request_clarification':
                requested_clarification_slot = operation.normalized_slot()
                if requested_clarification_slot not in _VALID_SLOTS:
                    requested_clarification_slot = self._preferred_slot(state, decision)
                applied_operations.append(operation)
                continue

            if operation.op == 'reject_confirmation':
                state.phase = 'collecting'
                requested_clarification_slot = operation.normalized_slot()
                if requested_clarification_slot not in _VALID_SLOTS:
                    requested_clarification_slot = self._preferred_slot(state, decision)
                applied_operations.append(operation)
                continue

            if operation.op == 'confirm_working_state':
                if self._can_commit(decision=decision, state=state):
                    state.committed_info = state.working_info.copy()
                    state.phase = 'notified_waiting'
                    state.pending_clarification_slot = ''
                    snapshot_confirmed = True
                    applied_operations.append(operation)
                    outbox_item = self._enqueue_confirmed_snapshot(state, turn_seq)
                    if outbox_item is not None:
                        outbox_items.append(outbox_item)
                else:
                    requested_clarification_slot = self._preferred_slot(state, decision)
                continue

        if not operations and decision.requires_confirmation:
            requested_clarification_slot = self._preferred_slot(state, decision)

        if rejected_reason and not requested_clarification_slot:
            requested_clarification_slot = self._preferred_slot(state, decision)

        dialog_act = self._resolve_dialog_act(
            state=state,
            decision=decision,
            requested_clarification_slot=requested_clarification_slot,
            snapshot_confirmed=snapshot_confirmed,
            had_changes=working_changed,
            applied_operations=applied_operations,
        )
        state.focus_slot = _slot_for_dialog_act(dialog_act)
        state.last_system_act = dialog_act
        state.latest_applied_turn = turn_seq
        state.version += 1
        state.turn_journal.append(
            {
                'turn_seq': turn_seq,
                'utterance_text': utterance_text,
                'speech_act': decision.speech_act,
                'target_slot': decision.target_slot,
                'operations': [self._op_to_payload(op) for op in operations],
                'applied_operations': [self._op_to_payload(op) for op in applied_operations],
                'dialog_act': dialog_act,
                'phase': state.phase,
                'working_info': state.working_info.as_dict(),
                'committed_info': state.committed_info.as_dict(),
            }
        )

        if working_changed or snapshot_confirmed:
            trace_events.append(
                TraceEventData(
                    event_type='WORKING_STATE_UPDATED',
                    dialog_act=dialog_act,
                    payload_json=json.dumps(
                        {
                            'working_info': state.working_info.as_dict(),
                            'committed_info': state.committed_info.as_dict(),
                            'applied_operations': [self._op_to_payload(op) for op in applied_operations],
                        },
                        ensure_ascii=False,
                    ),
                )
            )

        if state.pending_clarification_slot:
            trace_events.append(
                TraceEventData(
                    event_type='CLARIFICATION_REQUESTED',
                    dialog_act=dialog_act,
                    payload_json=json.dumps(
                        {'slot': state.pending_clarification_slot},
                        ensure_ascii=False,
                    ),
                )
            )

        if snapshot_confirmed:
            trace_events.append(
                TraceEventData(
                    event_type='SNAPSHOT_CONFIRMED',
                    dialog_act=dialog_act,
                    payload_json=json.dumps(
                        {'committed_info': state.committed_info.as_dict()},
                        ensure_ascii=False,
                    ),
                )
            )

        if outbox_items:
            trace_events.append(
                TraceEventData(
                    event_type='CHAT_OUTBOX_ENQUEUED',
                    dialog_act=dialog_act,
                    payload_json=json.dumps(
                        {'items': [self._outbox_to_payload(item) for item in outbox_items]},
                        ensure_ascii=False,
                    ),
                )
            )

        return ReducerOutcomeData(
            session_id=state.session_id,
            turn_seq=turn_seq,
            dialog_act=dialog_act,
            trace_events=trace_events,
            outbox_items=outbox_items,
            applied_operations=applied_operations,
            should_render_response=True,
        )

    def handle_secretary_reply(self, reply: SecretaryReplyData) -> ReducerOutcomeData | None:
        state = self.state
        if state.phase != 'notified_waiting':
            return None
        if not state.discord_thread_id or reply.thread_id != state.discord_thread_id:
            return None
        state.phase = 'relaying_reply'
        state.version += 1
        next_turn = state.latest_applied_turn + 1
        state.latest_applied_turn = next_turn
        state.last_system_act = 'relay_secretary'
        return ReducerOutcomeData(
            session_id=state.session_id,
            turn_seq=next_turn,
            dialog_act='relay_secretary',
            trace_events=[
                TraceEventData(
                    event_type='SECRETARY_REPLY_RECEIVED',
                    dialog_act='relay_secretary',
                    payload_json=json.dumps(
                        {'thread_id': reply.thread_id, 'message_id': reply.message_id},
                        ensure_ascii=False,
                    ),
                )
            ],
            should_render_response=False,
        )

    def mark_tts_completed(self, *, turn_seq: int, dialog_act: str) -> None:
        state = self.state
        if turn_seq < state.latest_applied_turn:
            return
        state.touch()
        if dialog_act == 'relay_secretary':
            state.phase = 'completed'
            state.version += 1

    def _update_response_language(
        self,
        state: SessionStateData,
        decision: SemanticDecisionData,
    ) -> None:
        detected = str(decision.detected_language or 'unknown').strip().lower()
        if float(decision.confidence) < self._confidence_threshold:
            return
        if detected not in {'ja', 'en'}:
            return
        state.response_language = detected

    def _prepare_operations(
        self,
        state: SessionStateData,
        decision: SemanticDecisionData,
        *,
        utterance_text: str,
    ) -> tuple[list[BeliefOperationData], str]:
        original_operations = [self._sanitize_operation(op) for op in decision.operations]
        operations = [op for op in original_operations if op.op in _VALID_OPS]
        operations = self._filter_incoherent_operations(state, decision, operations, utterance_text=utterance_text)

        if not operations:
            operations = self._derive_meta_operations(state, decision)
            if not operations and original_operations and decision.speech_act not in {'greeting', 'affirm'}:
                return operations, 'ungrounded_or_low_information_rejected'

        slot_ops = [
            op for op in operations if op.op in {'set_slot', 'replace_slot'} and op.normalized_slot() in _VALID_SLOTS and str(op.value or '').strip()
        ]
        if len(slot_ops) >= 2:
            unique_values = {str(op.value or '').strip() for op in slot_ops}
            touched_slots = {op.normalized_slot() for op in slot_ops}
            if len(unique_values) == 1 and len(touched_slots) >= 2:
                safe_ops = [
                    op for op in operations if op.op not in {'set_slot', 'replace_slot', 'clear_slot'}
                ]
                safe_ops.append(
                    BeliefOperationData(
                        op='request_clarification',
                        slot=self._preferred_slot(state, decision),
                        confidence=float(decision.confidence),
                    )
                )
                return safe_ops, 'same_value_multi_slot_rejected'

        return operations, ''

    def _filter_incoherent_operations(
        self,
        state: SessionStateData,
        decision: SemanticDecisionData,
        operations: list[BeliefOperationData],
        *,
        utterance_text: str,
    ) -> list[BeliefOperationData]:
        filtered: list[BeliefOperationData] = []
        low_information_utterance = _is_low_information_utterance(utterance_text)
        for operation in operations:
            slot = operation.normalized_slot()
            value = str(operation.value or '').strip()

            if operation.op in {'set_slot', 'replace_slot'}:
                if slot not in _VALID_SLOTS or not value:
                    continue
                if decision.speech_act in {'greeting', 'affirm'}:
                    continue
                if low_information_utterance:
                    continue
                if not _is_grounded_slot_operation(
                    utterance_text=utterance_text,
                    grounded_text=operation.grounded_text,
                    value=value,
                ):
                    continue
                filtered.append(operation)
                continue

            if operation.op == 'clear_slot':
                if slot not in _VALID_SLOTS:
                    continue
                if decision.speech_act == 'greeting':
                    continue
                filtered.append(operation)
                continue

            if operation.op == 'confirm_working_state':
                if state.phase != 'confirming':
                    continue
                filtered.append(operation)
                continue

            if operation.op == 'reject_confirmation':
                if state.phase != 'confirming':
                    continue
                filtered.append(operation)
                continue

            if operation.op == 'request_clarification':
                if slot not in _VALID_SLOTS:
                    filtered.append(
                        BeliefOperationData(
                            op='request_clarification',
                            slot=self._preferred_slot(state, decision),
                            confidence=float(operation.confidence or decision.confidence),
                        )
                    )
                else:
                    filtered.append(operation)
                continue

            if operation.op == 'ignore':
                filtered.append(operation)

        return filtered

    def _derive_meta_operations(
        self,
        state: SessionStateData,
        decision: SemanticDecisionData,
    ) -> list[BeliefOperationData]:
        if state.phase == 'confirming' and decision.speech_act == 'affirm':
            return [
                BeliefOperationData(
                    op='confirm_working_state',
                    slot='none',
                    confidence=float(decision.confidence),
                )
            ]
        if state.phase == 'confirming' and decision.speech_act in {'deny', 'correction'}:
            return [
                BeliefOperationData(
                    op='reject_confirmation',
                    slot=self._preferred_slot(state, decision),
                    confidence=float(decision.confidence),
                )
            ]
        if decision.requires_confirmation:
            return [
                BeliefOperationData(
                    op='request_clarification',
                    slot=self._preferred_slot(state, decision),
                    confidence=float(decision.confidence),
                )
            ]
        return []

    def _resolve_dialog_act(
        self,
        *,
        state: SessionStateData,
        decision: SemanticDecisionData,
        requested_clarification_slot: str,
        snapshot_confirmed: bool,
        had_changes: bool,
        applied_operations: list[BeliefOperationData],
    ) -> DialogAct:
        preferred_slot = self._preferred_slot(state, decision)

        if snapshot_confirmed:
            return 'notify_waiting'

        if state.phase == 'notified_waiting':
            return 'acknowledge_waiting'

        if requested_clarification_slot in _VALID_SLOTS:
            state.phase = 'collecting'
            state.pending_clarification_slot = requested_clarification_slot
            return _clarify_dialog_for_slot(requested_clarification_slot)

        if state.working_info.has_required_fields():
            state.phase = 'confirming'
            state.pending_clarification_slot = ''
            return 'confirm_snapshot'

        state.phase = 'collecting'
        if not had_changes and not applied_operations and decision.speech_act in {'question', 'complaint', 'unknown'}:
            state.pending_clarification_slot = preferred_slot
            return 'retry'

        state.pending_clarification_slot = ''
        next_slot = preferred_slot
        if next_slot not in state.working_info.missing_fields():
            missing = state.working_info.missing_fields()
            next_slot = missing[0] if missing else 'name'
        return _ask_dialog_for_slot(next_slot)

    def _can_commit(self, *, decision: SemanticDecisionData, state: SessionStateData) -> bool:
        if float(decision.confidence) < self._confidence_threshold:
            return False
        if str(decision.ambiguity or 'high').strip().lower() == 'high':
            return False
        return state.working_info.has_required_fields()

    def _enqueue_confirmed_snapshot(
        self,
        state: SessionStateData,
        turn_seq: int,
    ) -> ChatOutboxItemData | None:
        if not state.committed_info.has_required_fields():
            return None
        state.chat_outbox_cursor += 1
        item = ChatOutboxItemData(
            cursor=state.chat_outbox_cursor,
            item_id=uuid4().hex,
            session_id=state.session_id,
            turn_seq=turn_seq,
            event_type='confirmed_snapshot',
            title=f'受付 {state.session_id[:8]}',
            text=_format_confirmed_post(state.committed_info),
            thread_id=state.discord_thread_id,
            attempt_count=0,
            status='pending',
        )
        state.chat_outbox.append(item)
        state.chat_delivery_state = 'queued'
        return item

    def _preferred_slot(self, state: SessionStateData, decision: SemanticDecisionData) -> str:
        requested = _sanitize_slot(decision.target_slot)
        if state.pending_clarification_slot in _VALID_SLOTS:
            return state.pending_clarification_slot
        if requested in _VALID_SLOTS:
            return requested
        if state.focus_slot in _VALID_SLOTS:
            return state.focus_slot
        missing = state.working_info.missing_fields()
        return missing[0] if missing else 'name'

    @staticmethod
    def _sanitize_operation(operation: BeliefOperationData) -> BeliefOperationData:
        return BeliefOperationData(
            op=str(operation.op or 'ignore').strip(),
            slot=_sanitize_slot(operation.slot),
            value=str(operation.value or '').strip(),
            grounded_text=str(operation.grounded_text or '').strip(),
            confidence=float(operation.confidence or 0.0),
        )

    @staticmethod
    def _op_to_payload(operation: BeliefOperationData) -> dict[str, object]:
        return {
            'op': operation.op,
            'slot': operation.slot,
            'value': operation.value,
            'grounded_text': operation.grounded_text,
            'confidence': float(operation.confidence),
        }

    @staticmethod
    def _outbox_to_payload(item: ChatOutboxItemData) -> dict[str, object]:
        return {
            'cursor': int(item.cursor),
            'item_id': item.item_id,
            'event_type': item.event_type,
            'status': item.status,
            'thread_id': item.thread_id,
            'attempt_count': int(item.attempt_count),
        }


def _ask_dialog_for_slot(slot: str) -> DialogAct:
    if slot == 'name':
        return 'ask_name'
    if slot == 'affiliation':
        return 'ask_affiliation'
    return 'ask_purpose'


def _clarify_dialog_for_slot(slot: str) -> DialogAct:
    if slot == 'name':
        return 'clarify_name'
    if slot == 'affiliation':
        return 'clarify_affiliation'
    return 'clarify_purpose'


def _slot_for_dialog_act(dialog_act: str) -> str:
    if dialog_act in {'ask_name', 'clarify_name'}:
        return 'name'
    if dialog_act in {'ask_affiliation', 'clarify_affiliation'}:
        return 'affiliation'
    if dialog_act in {'ask_purpose', 'clarify_purpose'}:
        return 'purpose'
    return 'none'


def _sanitize_slot(slot: object) -> str:
    candidate = str(slot or 'none').strip().lower()
    if candidate in _VALID_SLOTS:
        return candidate
    return 'none'


def _is_grounded_slot_operation(*, utterance_text: str, grounded_text: str, value: str) -> bool:
    normalized_utterance = _normalize_grounding_text(utterance_text)
    if not normalized_utterance:
        return False

    grounded_candidates = [_normalize_grounding_text(grounded_text), _normalize_grounding_text(value)]
    grounded_candidates = [candidate for candidate in grounded_candidates if candidate]
    if not grounded_candidates:
        return False

    if any(candidate in normalized_utterance for candidate in grounded_candidates):
        return True

    # Reject especially fragile short latin fragments such as "ya" from "iya".
    if any(_is_fragile_latin_fragment(candidate) for candidate in grounded_candidates):
        return False

    return False


def _is_low_information_utterance(text: str) -> bool:
    normalized = _normalize_grounding_text(text)
    if not normalized:
        return True
    if len(normalized) <= 2:
        return True
    if len(normalized) <= 4 and all('a' <= char <= 'z' for char in normalized):
        return True
    return False


def _normalize_grounding_text(text: str) -> str:
    compact = ''.join(char.lower() for char in str(text or '') if char.isalnum())
    return compact


def _is_fragile_latin_fragment(text: str) -> bool:
    return bool(text) and len(text) <= 2 and all('a' <= char <= 'z' for char in text)


def _format_confirmed_post(info: VisitorInfoData) -> str:
    return (
        '【受付内容確定】\n'
        f'名前: {info.name or "未取得"}\n'
        f'所属: {info.affiliation or "未取得"}\n'
        f'用件: {info.purpose or "未取得"}\n'
        '担当者へ連携してください。'
    )


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)
