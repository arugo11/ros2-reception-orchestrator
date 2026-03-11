from __future__ import annotations

from dataclasses import asdict
from datetime import UTC
from datetime import datetime
import json
import uuid
from typing import Callable

from .dedup import RecentValueDeduplicator
from .dedup import stable_text_hash
from .formatters import fallback_dialog_text
from .formatters import format_confirmed_post
from .formatters import format_initial_post
from .formatters import format_update_post
from .state_models import DialogAct
from .state_models import DiscordUpdateKind
from .state_models import ReducerOutcome
from .state_models import SessionSnapshot
from .state_models import SessionState
from .state_models import SupervisorDecision
from .state_models import ThreadCreationResult
from .state_models import TurnContext
from .state_models import VisitorInfo


TraceCallable = Callable[[str], None]


class ReceptionOrchestratorCore:
    def __init__(
        self,
        *,
        inactivity_reset_sec: int = 60,
        trace: TraceCallable | None = None,
    ) -> None:
        self._inactivity_reset_sec = int(inactivity_reset_sec)
        self._trace = trace or (lambda message: None)
        self.session: SessionState | None = None
        self._seen_secretary_message_ids = RecentValueDeduplicator()

    def begin_turn(
        self,
        *,
        utterance_id: str,
        text: str,
        now: datetime | None = None,
    ) -> TurnContext | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        timestamp = now or datetime.now(tz=UTC)
        if self.session is None or self.session.phase == 'completed':
            self.reset()
            self._start_session(timestamp)

        session = self.session
        assert session is not None

        session.touch(timestamp)
        session.latest_turn_id += 1
        session.last_user_utterance = cleaned
        session.recent_events.append(f'user:{cleaned}')
        self._trim_recent_events(session)
        self._trace(f'visitor_utterance text={cleaned}')

        return TurnContext(
            session_id=session.session_id,
            turn_id=session.latest_turn_id,
            utterance_id=utterance_id,
            user_text=cleaned,
            snapshot=self._snapshot(session),
            create_thread=False,
            initial_thread_text='',
        )

    def reduce_supervisor_turn(
        self,
        *,
        session_id: str,
        turn_id: int,
        utterance_text: str,
        decision: SupervisorDecision,
        now: datetime | None = None,
    ) -> ReducerOutcome | None:
        session = self.session
        if session is None or session.session_id != session_id:
            return None
        if session.latest_turn_id != turn_id:
            return None

        timestamp = now or datetime.now(tz=UTC)
        session.touch(timestamp)

        changed_fields = self._apply_supervisor_updates(session, utterance_text, decision)
        missing_fields = session.visitor_info.missing_fields()
        self._trace(
            'semantic '
            f'name={decision.extracted_name or "-"} '
            f'affiliation={decision.extracted_affiliation or "-"} '
            f'purpose={decision.extracted_purpose or "-"} '
            f'affirmative={decision.speech_act == "affirm"} '
            f'correction={decision.correction_target != "none"}'
        )
        self._trace(
            'state_after_extraction '
            f'phase={session.phase} '
            f'name={session.visitor_info.name or "-"} '
            f'affiliation={session.visitor_info.affiliation or "-"} '
            f'purpose={session.visitor_info.purpose or "-"} '
            f'missing={",".join(missing_fields) or "none"} '
            f'changed={",".join(changed_fields) or "none"}'
        )

        if (
            decision.ignore_input
            and not changed_fields
            and not (session.visitor_info.name or session.visitor_info.affiliation or session.visitor_info.purpose)
        ):
            self._trace('ignored_input session remains unstarted_for_discord')
            return ReducerOutcome(session_id=session.session_id, turn_id=turn_id)

        has_valid_slot = bool(
            session.visitor_info.name
            or session.visitor_info.affiliation
            or session.visitor_info.purpose
        )
        create_thread = False
        initial_thread_text = ''
        if has_valid_slot and not session.discord.requested:
            session.discord.requested = True
            create_thread = True
            initial_thread_text = format_initial_post(session)

        discord_update_kind: DiscordUpdateKind = 'none'
        if changed_fields:
            discord_update_kind = 'update'

        dialog_act = self._resolve_dialog_act(session, decision, changed_fields=changed_fields)
        if dialog_act == 'confirm':
            session.phase = 'confirming'
            session.pending_confirmation = session.visitor_info.copy()
        elif dialog_act == 'notify_waiting':
            session.phase = 'notified_waiting'
            session.pending_confirmation = session.visitor_info.copy()
            discord_update_kind = 'confirmed'
        elif session.phase == 'confirming' and dialog_act in {'ask_name', 'ask_affiliation', 'ask_purpose', 'clarify', 'retry'}:
            session.phase = 'collecting'
            session.pending_confirmation = None

        discord_text = ''
        if discord_update_kind == 'update' and session.discord.thread_id:
            discord_text = format_update_post(session)
            if self._is_duplicate_discord_text(session, discord_text):
                discord_update_kind = 'none'
                discord_text = ''
        elif discord_update_kind == 'confirmed' and session.discord.thread_id:
            discord_text = format_confirmed_post(session)
            if self._is_duplicate_discord_text(session, discord_text):
                discord_update_kind = 'none'
                discord_text = ''

        return ReducerOutcome(
            session_id=session.session_id,
            turn_id=turn_id,
            dialog_act=dialog_act,
            spoken_response=self._select_spoken_response(
                session,
                dialog_act=dialog_act,
                spoken_response=decision.spoken_response,
                changed_fields=changed_fields,
            ),
            discord_update_kind=discord_update_kind,
            discord_text=discord_text,
            create_thread=create_thread,
            initial_thread_text=initial_thread_text,
        )

    def accept_spoken_response(
        self,
        *,
        session_id: str,
        turn_id: int,
        dialog_act: DialogAct,
        text: str,
        now: datetime | None = None,
    ) -> str | None:
        session = self.session
        if session is None or session.session_id != session_id:
            return None
        if dialog_act != 'relay_secretary' and turn_id != session.latest_turn_id:
            return None

        cleaned = text.strip() or fallback_dialog_text(dialog_act, session.visitor_info)
        timestamp = now or datetime.now(tz=UTC)
        session.touch(timestamp)
        session.last_dialog_act = dialog_act
        session.last_spoken_text = cleaned
        session.latest_spoken_turn_id = max(session.latest_spoken_turn_id, turn_id)
        session.recent_events.append(f'assistant:{dialog_act}:{cleaned}')
        self._trim_recent_events(session)
        return cleaned

    def handle_thread_created(self, result: ThreadCreationResult) -> str | None:
        session = self.session
        if session is None or session.session_id != result.session_id or not result.success:
            return None

        session.discord.thread_id = result.thread_id
        session.discord.channel_id = result.channel_id
        if session.phase in {'notified_waiting', 'relaying_reply', 'completed'}:
            return self._record_discord_hash(session, format_confirmed_post(session))
        if session.visitor_info.name or session.visitor_info.affiliation or session.visitor_info.purpose:
            return self._record_discord_hash(session, format_update_post(session))
        return self._record_discord_hash(session, format_initial_post(session))

    def handle_secretary_reply(
        self,
        *,
        thread_id: str,
        message_id: str,
        text: str,
        now: datetime | None = None,
    ) -> ReducerOutcome | None:
        session = self.session
        if session is None:
            return None
        if session.phase != 'notified_waiting':
            return None
        if not session.discord.thread_id or session.discord.thread_id != thread_id:
            return None
        if not text.strip():
            return None
        if not self._seen_secretary_message_ids.mark(message_id):
            return None

        timestamp = now or datetime.now(tz=UTC)
        session.touch(timestamp)
        session.phase = 'relaying_reply'
        session.latest_turn_id += 1
        return ReducerOutcome(
            session_id=session.session_id,
            turn_id=session.latest_turn_id,
            dialog_act='relay_secretary',
            spoken_response=text.strip(),
        )

    def handle_tts_completed(
        self,
        *,
        session_id: str,
        turn_id: int,
        dialog_act: DialogAct,
        now: datetime | None = None,
    ) -> bool:
        session = self.session
        if session is None or session.session_id != session_id:
            return False
        if turn_id < session.latest_spoken_turn_id:
            return False
        session.touch(now or datetime.now(tz=UTC))
        if dialog_act == 'relay_secretary':
            session.phase = 'completed'
            session.secretary_replied = True
        return True

    def handle_inactivity(self, *, now: datetime | None = None) -> bool:
        session = self.session
        if session is None:
            return False
        timestamp = now or datetime.now(tz=UTC)
        inactive_sec = (timestamp - session.last_activity_at).total_seconds()
        if inactive_sec < self._inactivity_reset_sec:
            return False
        self.reset()
        self._trace('session_reset reason=inactivity')
        return True

    def debug_state_payload(self) -> str:
        if self.session is None:
            return json.dumps({'session': None}, ensure_ascii=False)
        return json.dumps({'session': asdict(self.session)}, ensure_ascii=False, default=str)

    def reset(self) -> None:
        self.session = None
        self._seen_secretary_message_ids = RecentValueDeduplicator()

    def _start_session(self, now: datetime) -> None:
        session_id = uuid.uuid4().hex
        self.session = SessionState(
            session_id=session_id,
            started_at=now,
            last_activity_at=now,
            phase='collecting',
        )
        self._trace(f'session_started session_id={session_id}')

    def _snapshot(self, session: SessionState) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=session.session_id,
            phase=session.phase,
            visitor_info=session.visitor_info.copy(),
            last_user_utterance=session.last_user_utterance,
            last_dialog_act=session.last_dialog_act,
            last_spoken_text=session.last_spoken_text,
            pending_confirmation=(session.pending_confirmation.copy() if session.pending_confirmation else None),
            latest_turn_id=session.latest_turn_id,
        )

    def _apply_supervisor_updates(
        self,
        session: SessionState,
        utterance_text: str,
        decision: SupervisorDecision,
    ) -> list[str]:
        changed_fields: list[str] = []
        updates = {
            'name': decision.extracted_name,
            'affiliation': decision.extracted_affiliation,
            'purpose': decision.extracted_purpose,
        }

        for field_name, value in updates.items():
            if not value:
                continue
            current = getattr(session.visitor_info, field_name)
            if current == value:
                continue
            setattr(session.visitor_info, field_name, value)
            changed_fields.append(field_name)
        return changed_fields

    def _resolve_dialog_act(
        self,
        session: SessionState,
        decision: SupervisorDecision,
        *,
        changed_fields: list[str],
    ) -> DialogAct:
        if session.phase == 'notified_waiting':
            return 'acknowledge_waiting'

        missing_fields = session.visitor_info.missing_fields()

        if session.phase == 'confirming':
            if decision.speech_act == 'affirm' and not changed_fields:
                return 'notify_waiting'
            if changed_fields or decision.correction_target != 'none':
                if missing_fields:
                    return _dialog_act_for_missing(missing_fields)
                return 'confirm'
            if decision.speech_act == 'affirm':
                return 'notify_waiting'
            if decision.speech_act in {'deny', 'correction', 'complaint'}:
                return 'clarify'
            if decision.ignore_input:
                return 'confirm'
            if any(
                (
                    decision.extracted_name,
                    decision.extracted_affiliation,
                    decision.extracted_purpose,
                )
            ):
                if missing_fields:
                    return _dialog_act_for_missing(missing_fields)
                return 'confirm'
            return 'confirm'

        if session.visitor_info.has_required_fields() or decision.should_confirm:
            return 'confirm'

        if decision.ignore_input and not any(
            (
                decision.extracted_name,
                decision.extracted_affiliation,
                decision.extracted_purpose,
            )
        ):
            return 'retry'
        return _dialog_act_for_missing(missing_fields)

    def _select_spoken_response(
        self,
        session: SessionState,
        *,
        dialog_act: DialogAct,
        spoken_response: str | None,
        changed_fields: list[str],
    ) -> str:
        fallback = fallback_dialog_text(dialog_act, session.visitor_info)
        if dialog_act != 'relay_secretary':
            return fallback
        cleaned = (spoken_response or '').strip()
        if not cleaned:
            return fallback
        if session.last_spoken_text and cleaned == session.last_spoken_text:
            return fallback
        if changed_fields and session.last_spoken_text and cleaned == session.last_spoken_text:
            return fallback
        return cleaned

    def _is_duplicate_discord_text(self, session: SessionState, text: str) -> bool:
        content_hash = stable_text_hash(text)
        if session.discord.last_post_hash == content_hash:
            return True
        session.discord.last_post_hash = content_hash
        return False

    def _record_discord_hash(self, session: SessionState, text: str) -> str:
        self._is_duplicate_discord_text(session, text)
        return text

    @staticmethod
    def _trim_recent_events(session: SessionState) -> None:
        if len(session.recent_events) > 32:
            del session.recent_events[:-32]


def _dialog_act_for_missing(missing_fields: list[str]) -> DialogAct:
    if not missing_fields:
        return 'confirm'
    first = missing_fields[0]
    if first == 'name':
        return 'ask_name'
    if first == 'affiliation':
        return 'ask_affiliation'
    return 'ask_purpose'
