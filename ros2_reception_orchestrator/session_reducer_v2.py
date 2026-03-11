from __future__ import annotations

import json
from uuid import uuid4

from reception_interfaces.msg import ExecutionCommand

from .v2_types import DialogAct
from .v2_types import OrchestratorCommandData
from .v2_types import ReducerOutcomeData
from .v2_types import SecretaryReplyData
from .v2_types import SemanticDecisionData
from .v2_types import SessionStateData
from .v2_types import VisitorInfoData


class SessionReducer:
    """Pure deterministic reducer for canonical state transitions."""

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

        changed_fields = self._apply_slot_patch(state, decision)
        dialog_act = self._resolve_dialog_act(state, decision, changed_fields)
        commands = self._build_commands(state, turn_seq, dialog_act, changed_fields)

        state.latest_applied_turn = turn_seq
        state.version += 1

        return ReducerOutcomeData(
            session_id=state.session_id,
            turn_seq=turn_seq,
            dialog_act=dialog_act,
            commands=commands,
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
        return ReducerOutcomeData(
            session_id=state.session_id,
            turn_seq=next_turn,
            dialog_act='relay_secretary',
            commands=[],
            should_render_response=False,
        )

    def mark_tts_completed(self, *, turn_seq: int, dialog_act: str) -> None:
        state = self.state
        if turn_seq < state.latest_applied_turn:
            return
        if dialog_act == 'relay_secretary':
            state.phase = 'completed'
            state.version += 1

    def _apply_slot_patch(
        self,
        state: SessionStateData,
        decision: SemanticDecisionData,
    ) -> list[str]:
        if decision.ignore_input:
            return []
        if float(decision.confidence) < self._confidence_threshold:
            return []

        changed: list[str] = []
        patch = decision.slot_patch
        for field_name, value in (
            ('name', patch.name.strip()),
            ('affiliation', patch.affiliation.strip()),
            ('purpose', patch.purpose.strip()),
        ):
            if not value:
                continue
            current = getattr(state.visitor_info, field_name)
            if current == value:
                continue
            setattr(state.visitor_info, field_name, value)
            changed.append(field_name)
        return changed

    def _resolve_dialog_act(
        self,
        state: SessionStateData,
        decision: SemanticDecisionData,
        changed_fields: list[str],
    ) -> DialogAct:
        missing = state.visitor_info.missing_fields()

        if state.phase == 'notified_waiting':
            return 'acknowledge_waiting'

        if state.phase == 'confirming':
            if decision.speech_act == 'affirm' and not changed_fields:
                state.phase = 'notified_waiting'
                state.pending_confirmation = state.visitor_info.copy()
                return 'notify_waiting'
            if changed_fields or decision.correction_target != 'none':
                if missing:
                    state.phase = 'collecting'
                    return _dialog_for_missing(missing)
                state.pending_confirmation = state.visitor_info.copy()
                return 'confirm'
            return 'confirm'

        if state.visitor_info.has_required_fields():
            state.phase = 'confirming'
            state.pending_confirmation = state.visitor_info.copy()
            return 'confirm'

        state.phase = 'collecting'
        if decision.ignore_input and not changed_fields:
            return 'retry'
        return _dialog_for_missing(missing)

    def _build_commands(
        self,
        state: SessionStateData,
        turn_seq: int,
        dialog_act: DialogAct,
        changed_fields: list[str],
    ) -> list[OrchestratorCommandData]:
        commands: list[OrchestratorCommandData] = []

        has_any_slot = bool(
            state.visitor_info.name
            or state.visitor_info.affiliation
            or state.visitor_info.purpose
        )
        if has_any_slot and not state.discord_create_requested:
            state.discord_create_requested = True
            commands.append(
                _command(
                    command_type=ExecutionCommand.COMMAND_DISCORD_CREATE,
                    session_id=state.session_id,
                    turn_seq=turn_seq,
                    payload={
                        'title': f'受付 {state.session_id[:8]}',
                        'initial': _format_initial_post(state),
                    },
                )
            )

        if changed_fields and state.discord_thread_id:
            commands.append(
                _command(
                    command_type=ExecutionCommand.COMMAND_DISCORD_SEND,
                    session_id=state.session_id,
                    turn_seq=turn_seq,
                    payload={'thread_id': state.discord_thread_id, 'text': _format_update_post(state)},
                )
            )

        if dialog_act == 'notify_waiting' and state.discord_thread_id and not state.confirmed_posted:
            state.confirmed_posted = True
            commands.append(
                _command(
                    command_type=ExecutionCommand.COMMAND_DISCORD_SEND,
                    session_id=state.session_id,
                    turn_seq=turn_seq,
                    payload={'thread_id': state.discord_thread_id, 'text': _format_confirmed_post(state)},
                )
            )
        return commands


def _dialog_for_missing(missing: list[str]) -> DialogAct:
    if not missing:
        return 'confirm'
    if missing[0] == 'name':
        return 'ask_name'
    if missing[0] == 'affiliation':
        return 'ask_affiliation'
    return 'ask_purpose'


def _command(*, command_type: int, session_id: str, turn_seq: int, payload: dict[str, str]) -> OrchestratorCommandData:
    return OrchestratorCommandData(
        command_type=command_type,
        command_id=uuid4().hex,
        session_id=session_id,
        turn_seq=turn_seq,
        payload_json=json.dumps(payload, ensure_ascii=False),
    )


def _format_initial_post(state: SessionStateData) -> str:
    return (
        '【受付開始】\n'
        f'名前: {state.visitor_info.name or "未取得"}\n'
        f'所属: {state.visitor_info.affiliation or "未取得"}\n'
        f'用件: {state.visitor_info.purpose or "未取得"}'
    )


def _format_update_post(state: SessionStateData) -> str:
    return (
        '【受付情報更新】\n'
        f'名前: {state.visitor_info.name or "未取得"}\n'
        f'所属: {state.visitor_info.affiliation or "未取得"}\n'
        f'用件: {state.visitor_info.purpose or "未取得"}'
    )


def _format_confirmed_post(state: SessionStateData) -> str:
    return (
        '【確認完了】\n'
        f'名前: {state.visitor_info.name or "未取得"}\n'
        f'所属: {state.visitor_info.affiliation or "未取得"}\n'
        f'用件: {state.visitor_info.purpose or "未取得"}\n'
        '担当者へ連絡済みです。'
    )
