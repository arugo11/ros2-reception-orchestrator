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
    'clarify_name',
    'clarify_affiliation',
    'clarify_purpose',
    'confirm_snapshot',
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

ResponseLanguage = Literal['ja', 'en', 'unknown']
SlotName = Literal['name', 'affiliation', 'purpose', 'none']
Ambiguity = Literal['low', 'medium', 'high']
OperationType = Literal[
    'set_slot',
    'replace_slot',
    'clear_slot',
    'confirm_working_state',
    'reject_confirmation',
    'request_clarification',
    'ignore',
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

    def as_dict(self) -> dict[str, str]:
        return {
            'name': self.name,
            'affiliation': self.affiliation,
            'purpose': self.purpose,
        }


@dataclass(slots=True)
class BeliefOperationData:
    op: str = 'ignore'
    slot: str = 'none'
    value: str = ''
    grounded_text: str = ''
    confidence: float = 0.0

    def normalized_slot(self) -> str:
        slot = str(self.slot or 'none').strip().lower()
        if slot in {'name', 'affiliation', 'purpose'}:
            return slot
        return 'none'


@dataclass(slots=True)
class SlotProvenanceData:
    slot: str
    source_turn_seq: int
    grounded_text: str = ''
    confidence: float = 0.0
    updated_at: str = ''


@dataclass(slots=True)
class ChatOutboxItemData:
    cursor: int
    item_id: str
    session_id: str
    turn_seq: int
    event_type: str
    title: str
    text: str
    thread_id: str = ''
    attempt_count: int = 0
    status: str = 'pending'


@dataclass(slots=True)
class TraceEventData:
    event_type: str
    text: str = ''
    dialog_act: str = ''
    role: str = 'system'
    payload_json: str = ''


@dataclass(slots=True)
class SemanticDecisionData:
    turn_seq: int
    speech_act: str = 'unknown'
    detected_language: str = 'unknown'
    target_slot: str = 'none'
    ambiguity: str = 'high'
    requires_confirmation: bool = False
    confidence: float = 0.0
    evidence: str = ''
    operations: list[BeliefOperationData] = field(default_factory=list)
    grounded_segments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionStateData:
    session_id: str
    phase: Phase = 'collecting'
    response_language: str = 'ja'
    working_info: VisitorInfoData = field(default_factory=VisitorInfoData)
    committed_info: VisitorInfoData = field(default_factory=VisitorInfoData)
    focus_slot: str = 'name'
    last_system_act: str = ''
    pending_clarification_slot: str = ''
    working_provenance: dict[str, SlotProvenanceData] = field(default_factory=dict)
    turn_journal: list[dict[str, object]] = field(default_factory=list)
    chat_outbox: list[ChatOutboxItemData] = field(default_factory=list)
    chat_outbox_cursor: int = 0
    chat_delivery_state: str = 'idle'
    latest_applied_turn: int = 0
    version: int = 0
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    discord_thread_id: str = ''
    discord_channel_id: str = ''

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
    trace_events: list[TraceEventData] = field(default_factory=list)
    outbox_items: list[ChatOutboxItemData] = field(default_factory=list)
    applied_operations: list[BeliefOperationData] = field(default_factory=list)
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
