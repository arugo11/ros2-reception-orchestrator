from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Literal


Phase = Literal[
    'idle',
    'collecting',
    'confirming',
    'notified_waiting',
    'relaying_reply',
    'completed',
]

FieldName = Literal['name', 'affiliation', 'purpose']
DialogAct = Literal[
    'ask_name',
    'ask_affiliation',
    'ask_purpose',
    'confirm',
    'notify_waiting',
    'acknowledge_waiting',
    'clarify',
    'retry',
    'relay_secretary',
]
SpeechAct = Literal[
    'inform',
    'affirm',
    'deny',
    'correction',
    'question',
    'complaint',
    'greeting',
    'unknown',
]
CorrectionTarget = Literal['none', 'name', 'affiliation', 'purpose', 'all']
DiscordUpdateKind = Literal['initial', 'update', 'confirmed', 'none']


@dataclass(slots=True)
class VisitorInfo:
    name: str | None = None
    affiliation: str | None = None
    purpose: str | None = None

    def missing_fields(self) -> list[FieldName]:
        missing: list[FieldName] = []
        if not self.name:
            missing.append('name')
        if not self.affiliation:
            missing.append('affiliation')
        if not self.purpose:
            missing.append('purpose')
        return missing

    def has_required_fields(self) -> bool:
        return not self.missing_fields()

    def copy(self) -> 'VisitorInfo':
        return VisitorInfo(
            name=self.name,
            affiliation=self.affiliation,
            purpose=self.purpose,
        )


@dataclass(slots=True)
class DiscordThreadState:
    requested: bool = False
    thread_id: str | None = None
    channel_id: str | None = None
    last_post_hash: str | None = None
    confirmed_posted: bool = False


@dataclass(slots=True)
class SupervisorDecision:
    speech_act: SpeechAct = 'unknown'
    extracted_name: str | None = None
    extracted_affiliation: str | None = None
    extracted_purpose: str | None = None
    slot_confidence: float = 0.0
    missing_fields: list[FieldName] = field(default_factory=list)
    next_dialog_act: DialogAct | None = None
    should_confirm: bool = False
    correction_target: CorrectionTarget = 'none'
    discord_update_kind: DiscordUpdateKind = 'none'
    ignore_input: bool = False


@dataclass(slots=True)
class SessionState:
    session_id: str
    started_at: datetime
    last_activity_at: datetime
    phase: Phase = 'collecting'
    visitor_info: VisitorInfo = field(default_factory=VisitorInfo)
    last_user_utterance: str = ''
    last_dialog_act: DialogAct | None = None
    pending_confirmation: VisitorInfo | None = None
    discord: DiscordThreadState = field(default_factory=DiscordThreadState)
    recent_events: list[str] = field(default_factory=list)
    latest_turn_id: int = 0
    latest_spoken_turn_id: int = 0
    secretary_replied: bool = False

    def touch(self, now: datetime) -> None:
        self.last_activity_at = now


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    phase: Phase
    visitor_info: VisitorInfo
    last_user_utterance: str
    last_dialog_act: DialogAct | None
    pending_confirmation: VisitorInfo | None
    latest_turn_id: int


@dataclass(slots=True)
class TurnContext:
    session_id: str
    turn_id: int
    utterance_id: str
    user_text: str
    snapshot: SessionSnapshot
    create_thread: bool = False
    initial_thread_text: str = ''


@dataclass(slots=True)
class DialogRenderRequest:
    session_id: str
    turn_id: int
    dialog_act: DialogAct
    phase: Phase
    visitor_info: VisitorInfo
    pending_confirmation: VisitorInfo | None = None
    secretary_reply_text: str = ''


@dataclass(slots=True)
class ReducerOutcome:
    session_id: str
    turn_id: int
    dialog_request: DialogRenderRequest | None = None
    discord_update_kind: DiscordUpdateKind = 'none'
    discord_text: str = ''
    create_thread: bool = False
    initial_thread_text: str = ''


@dataclass(slots=True)
class ThreadCreationResult:
    session_id: str
    success: bool
    thread_id: str = ''
    channel_id: str = ''
    error_message: str = ''
