from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
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

DialogAct = Literal[
    'ask_name',
    'ask_affiliation',
    'ask_purpose',
    'confirm',
    'notify_waiting',
    'acknowledge_waiting',
    'relay_secretary',
    'retry',
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


@dataclass(slots=True)
class VisitorInfoData:
    name: str = ''
    affiliation: str = ''
    purpose: str = ''

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.name.strip():
            missing.append('name')
        if not self.affiliation.strip():
            missing.append('affiliation')
        if not self.purpose.strip():
            missing.append('purpose')
        return missing

    def has_required_fields(self) -> bool:
        return not self.missing_fields()

    def copy(self) -> 'VisitorInfoData':
        return VisitorInfoData(
            name=self.name,
            affiliation=self.affiliation,
            purpose=self.purpose,
        )


@dataclass(slots=True)
class SemanticDecisionData:
    turn_seq: int
    speech_act: str = 'unknown'
    slot_patch: VisitorInfoData = field(default_factory=VisitorInfoData)
    correction_target: str = 'none'
    ignore_input: bool = False
    confidence: float = 0.0
    evidence: str = ''


@dataclass(slots=True)
class SessionStateData:
    session_id: str
    phase: Phase = 'collecting'
    visitor_info: VisitorInfoData = field(default_factory=VisitorInfoData)
    pending_confirmation: VisitorInfoData = field(default_factory=VisitorInfoData)
    latest_applied_turn: int = 0
    version: int = 0
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    discord_thread_id: str = ''
    discord_channel_id: str = ''
    discord_create_requested: bool = False
    confirmed_posted: bool = False

    def touch(self) -> None:
        self.last_activity_at = datetime.now(tz=UTC)


@dataclass(slots=True)
class OrchestratorCommandData:
    command_type: int
    command_id: str
    session_id: str
    turn_seq: int
    payload_json: str
    dialog_act: str = ''


@dataclass(slots=True)
class ReducerOutcomeData:
    session_id: str
    turn_seq: int
    dialog_act: DialogAct
    commands: list[OrchestratorCommandData] = field(default_factory=list)
    should_render_response: bool = True


@dataclass(slots=True)
class TurnEnvelopeData:
    session_id: str
    turn_seq: int
    utterance_id: str
    text: str
    captured_during_tts: bool
    asr_confidence: float


@dataclass(slots=True)
class SecretaryReplyData:
    thread_id: str
    message_id: str
    text: str
