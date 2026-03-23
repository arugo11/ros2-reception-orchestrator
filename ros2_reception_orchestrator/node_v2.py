from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import os
import queue
import threading
from typing import Any
from uuid import uuid4

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from asr_interfaces.msg import Utterance
from reception_interfaces.action import ExtractTurn
from reception_interfaces.msg import BeliefOperation
from reception_interfaces.msg import ChatOutboxItem
from reception_interfaces.msg import ConversationTrace
from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent
from reception_interfaces.msg import SessionStateV2
from reception_interfaces.msg import SlotProvenance
from reception_interfaces.msg import TurnEnvelope
from reception_interfaces.msg import VisitorInfo
from ros2_chat_interfaces.msg import ChatMessage
from ros2_chat_interfaces.msg import ChatTarget
from ros2_chat_interfaces.srv import CreateThread
from ros2_chat_interfaces.srv import SendMessage
from ros2_vllm.utils import extract_completion_text
from ros2_vllm.vllm_client import VllmClient
from ros2_vllm_interfaces.action import Chat
from ros2_vllm_interfaces.msg import LlmStatus
from tts_msgs.action import Speak

from .conversation_log import ConversationLogWriter
from .conversation_trace import build_conversation_trace_message
from .effect_executor_v2 import EffectExecutor
from .llm_stage_utils import extract_json_object
from .llm_stage_utils import wait_future
from .prompt_templates import RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA
from .prompt_templates import RECEPTION_REPAIR_SYSTEM_PROMPT
from .prompt_templates import RECEPTION_SLOT_EXTRACT_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_NORMALIZE_JSON_SCHEMA
from .prompt_templates import RECEPTION_SLOT_NORMALIZE_SYSTEM_PROMPT
from .prompt_templates import build_reception_confirmation_rescue_prompt
from .prompt_templates import build_reception_slot_extract_prompt
from .prompt_templates import build_reception_slot_normalize_prompt
from .response_planner_server import _fallback_dialog_text
from .semantic_extractor_server import _STAGE1_JSON_SCHEMA
from .semantic_extractor_server import _STAGE1_SYSTEM_PROMPT
from .session_reducer_v2 import SessionReducer
from .state_models import SessionSnapshot
from .state_models import VisitorInfo
from .turn_ingestor_v2 import TurnIngestor
from .v2_types import BeliefOperationData
from .v2_types import ChatOutboxItemData
from .v2_types import OrchestratorCommandData
from .v2_types import SecretaryReplyData
from .v2_types import SemanticDecisionData
from .v2_types import SlotProvenanceData
from .v2_types import TraceEventData
from .v2_types import TurnEnvelopeData
from .v2_types import VisitorInfoData


@dataclass(slots=True)
class _QueueTurnEvent:
    turn: TurnEnvelopeData


@dataclass(slots=True)
class _QueueSecretaryEvent:
    reply: SecretaryReplyData


class ReceptionOrchestratorNodeV2(Node):
    _READY_MARKER = 'Conversation backends ready: ASR, LLM, and TTS are available'
    _READY_ANNOUNCEMENT_SEGMENTS = (
        ('こんにちは。', 'ja'),
        ('Welcome to S I T.', 'en'),
    )
    _DEFAULT_TTS_VOLUME = 0.1

    def __init__(self) -> None:
        super().__init__('reception_orchestrator')
        self._tts_volume = self._load_tts_volume()

        self._declare_parameters()
        self._load_parameters()
        self._conversation_log_writer = ConversationLogWriter(
            enabled=self._conversation_log_enabled,
            output_dir=self._conversation_log_output_dir,
            log_format=self._conversation_log_format,
            scope=self._conversation_log_scope,
            flush_on_session_switch=self._conversation_log_flush_on_session_switch,
        )
        self._client_callback_group = ReentrantCallbackGroup()

        self._extract_client = ActionClient(
            self,
            ExtractTurn,
            self._extract_action_name,
            callback_group=self._client_callback_group,
        )
        self._chat_client = ActionClient(
            self,
            Chat,
            self._llm_chat_action_name,
            callback_group=self._client_callback_group,
        )
        self._tts_client = ActionClient(
            self,
            Speak,
            self._tts_action_name,
            callback_group=self._client_callback_group,
        )

        self._create_thread_client = self.create_client(
            CreateThread,
            '/chat_bridge/create_thread',
            callback_group=self._client_callback_group,
        )
        self._send_message_client = self.create_client(
            SendMessage,
            '/chat_bridge/send_message',
            callback_group=self._client_callback_group,
        )

        self._utterance_subscription = self.create_subscription(
            Utterance,
            self._asr_utterance_topic,
            self._on_utterance,
            30,
        )
        self._incoming_subscription = self.create_subscription(
            ChatMessage,
            '/chat_bridge/incoming',
            self._on_chat_incoming,
            30,
        )
        self._llm_status_subscription = self.create_subscription(
            LlmStatus,
            self._llm_status_topic,
            self._on_llm_status,
            30,
        )

        self._session_state_publisher = self.create_publisher(SessionStateV2, self._session_state_topic, 20)
        self._event_publisher = self.create_publisher(ExecutionEvent, self._execution_event_topic, 50)
        self._conversation_trace_publisher = self.create_publisher(
            ConversationTrace,
            self._conversation_trace_topic,
            50,
        )

        self._event_queue: queue.Queue[object] = queue.Queue()
        self._chat_outbox_queue: queue.Queue[str] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._seen_secretary_messages: set[str] = set()
        self._pending_turn_events: list[TurnEnvelopeData] = []
        self._state_lock = threading.RLock()
        self._ready_logged = False
        self._ready_announcement_sent = False
        self._llm_backend_ready = False
        self._llm_model_name = ''
        self._direct_vllm_client: VllmClient | None = None
        self._active_tts_goal_lock = threading.RLock()
        self._active_tts_goal_handle = None
        self._active_tts_command_id = ''

        self._reducer = SessionReducer(confidence_threshold=self._semantic_confidence_threshold)
        self._ingestor = TurnIngestor(merge_window_sec=self._followup_merge_window_sec)

        self._effect_executor = EffectExecutor(
            invoke_tts=self._invoke_tts_command,
            publish_event=self._publish_execution_event,
            completion_hook=self._on_command_completed,
            same_turn_replace=self._same_turn_replace,
        )

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._outbox_worker = threading.Thread(target=self._chat_outbox_loop, daemon=True)
        self._outbox_worker.start()

        self._tick_timer = self.create_timer(0.1, self._on_tick)
        self.get_logger().info('reception_orchestrator_v2 ready')
        self._publish_state()

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._outbox_worker.is_alive():
            self._outbox_worker.join(timeout=2.0)
        self._effect_executor.shutdown()
        self._conversation_log_writer.flush_all()
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter('discord.adapter_name', 'discord')
        self.declare_parameter('discord.parent_channel_id', '')
        self.declare_parameter('session.inactivity_reset_sec', 60)
        self.declare_parameter('asr.utterance_topic', '/asr/utterances')
        self.declare_parameter('llm.status_topic', '/llm/status')
        self.declare_parameter('llm.chat_action_name', '/llm/chat')
        self.declare_parameter('llm.vllm_host', '127.0.0.1')
        self.declare_parameter('llm.vllm_port', 8000)
        self.declare_parameter('llm.request_timeout_sec', 60.0)
        self.declare_parameter('extract.action_name', '/reception/extract_turn')
        self.declare_parameter('tts.action_name', '/tts/speak')
        self.declare_parameter('session.state_topic', '/reception/session_state')
        self.declare_parameter('execution.event_topic', '/reception/events')
        self.declare_parameter('conversation.trace_topic', '/reception/conversation_trace')
        self.declare_parameter('conversation_log.enabled', True)
        self.declare_parameter(
            'conversation_log.output_dir',
            '/workspaces/ros2-workspace-template/logs/conversations',
        )
        self.declare_parameter('conversation_log.format', 'text')
        self.declare_parameter('conversation_log.scope', 'utterances')
        self.declare_parameter('conversation_log.flush_on_session_switch', True)
        self.declare_parameter('response.followup_merge_window_ms', 1200)
        self.declare_parameter('queue.same_turn_replace', True)
        self.declare_parameter('semantic.confidence_threshold', 0.55)

    def _load_parameters(self) -> None:
        self._discord_adapter_name = str(self.get_parameter('discord.adapter_name').value)
        self._discord_parent_channel_id = str(self.get_parameter('discord.parent_channel_id').value).strip()
        self._session_inactivity_reset_sec = int(self.get_parameter('session.inactivity_reset_sec').value)
        if not self._discord_parent_channel_id:
            raise ValueError('discord.parent_channel_id is required')

        self._asr_utterance_topic = str(self.get_parameter('asr.utterance_topic').value)
        self._llm_status_topic = str(self.get_parameter('llm.status_topic').value)
        self._llm_chat_action_name = str(self.get_parameter('llm.chat_action_name').value)
        self._llm_vllm_host = str(self.get_parameter('llm.vllm_host').value).strip() or '127.0.0.1'
        self._llm_vllm_port = int(self.get_parameter('llm.vllm_port').value)
        self._llm_request_timeout_sec = float(self.get_parameter('llm.request_timeout_sec').value)
        self._extract_action_name = str(self.get_parameter('extract.action_name').value)
        self._tts_action_name = str(self.get_parameter('tts.action_name').value)
        self._session_state_topic = str(self.get_parameter('session.state_topic').value)
        self._execution_event_topic = str(self.get_parameter('execution.event_topic').value)
        self._conversation_trace_topic = str(self.get_parameter('conversation.trace_topic').value)
        self._conversation_log_enabled = bool(self.get_parameter('conversation_log.enabled').value)
        self._conversation_log_output_dir = str(self.get_parameter('conversation_log.output_dir').value).strip()
        self._conversation_log_format = str(self.get_parameter('conversation_log.format').value).strip()
        self._conversation_log_scope = str(self.get_parameter('conversation_log.scope').value).strip()
        self._conversation_log_flush_on_session_switch = bool(
            self.get_parameter('conversation_log.flush_on_session_switch').value
        )

        self._followup_merge_window_sec = float(self.get_parameter('response.followup_merge_window_ms').value) / 1000.0
        self._same_turn_replace = bool(self.get_parameter('queue.same_turn_replace').value)
        self._semantic_confidence_threshold = float(self.get_parameter('semantic.confidence_threshold').value)

    def _on_tick(self) -> None:
        with self._state_lock:
            if not self._ready_logged and self._dependencies_ready():
                self.get_logger().info(self._READY_MARKER)
                self._ready_logged = True
            if self._dependencies_ready() and not self._ready_announcement_sent:
                self._submit_ready_announcement()
                self._ready_announcement_sent = True

            session_id = self._reducer.state.session_id
            flushed = self._ingestor.flush_due(session_id=session_id)
            if flushed is not None:
                self._enqueue_or_buffer_turn(flushed)

            if self._dependencies_ready() and self._pending_turn_events:
                pending = self._pending_turn_events
                self._pending_turn_events = []
                for turn in pending:
                    self._event_queue.put(_QueueTurnEvent(turn))
                self.get_logger().info(
                    f'[PIPELINE] resumed {len(pending)} buffered turn(s) after backend ready'
                )

            self._handle_session_inactivity()
            self._publish_state()

    def _handle_session_inactivity(self) -> None:
        state = self._reducer.state
        if not self._session_has_user_activity(state):
            return

        now = datetime.now(tz=UTC)
        idle_sec = (now - state.last_activity_at).total_seconds()
        if idle_sec < float(self._session_inactivity_reset_sec):
            return

        expired_session_id = state.session_id
        expired_turn_seq = state.latest_applied_turn
        self.get_logger().info(
            '[SESSION] inactivity timeout '
            f'session_id={expired_session_id} idle_sec={idle_sec:.1f} '
            f'threshold_sec={self._session_inactivity_reset_sec}'
        )
        self._effect_executor.cancel_pending_tts(detail='session_timeout_pending_tts')
        self._publish_conversation_trace(
            role=ConversationTrace.ROLE_SYSTEM,
            session_id=expired_session_id,
            turn_seq=expired_turn_seq,
            text='',
            dialog_act='',
            phase=state.phase,
            utterance_id='',
            asr_confidence=0.0,
            event_type='SESSION_TIMEOUT',
            event_payload=json.dumps(
                {
                    'idle_sec': round(idle_sec, 3),
                    'timeout_sec': int(self._session_inactivity_reset_sec),
                },
                ensure_ascii=False,
            ),
            payload_json=json.dumps(
                {
                    'idle_sec': round(idle_sec, 3),
                    'timeout_sec': int(self._session_inactivity_reset_sec),
                },
                ensure_ascii=False,
            ),
        )
        self._conversation_log_writer.flush_all()
        self._pending_turn_events = []
        self._ingestor.reset()
        self._reducer.reset()

    @staticmethod
    def _session_has_user_activity(state: SessionStateData) -> bool:
        return bool(
            state.latest_applied_turn > 0
            or state.working_info.name
            or state.working_info.affiliation
            or state.working_info.purpose
            or state.committed_info.name
            or state.committed_info.affiliation
            or state.committed_info.purpose
            or state.chat_outbox
            or state.discord_thread_id
            or state.discord_channel_id
        )

    def _dependencies_ready(self) -> bool:
        return (
            self._llm_backend_ready
            and self._chat_client.server_is_ready()
            and self._tts_client.server_is_ready()
        )

    def _on_utterance(self, msg: Utterance) -> None:
        local_tts_active = self._effect_executor.is_tts_active()
        if local_tts_active:
            self._effect_executor.cancel_pending_tts(detail='barge_in_pending_tts')
            self._cancel_active_tts_for_barge_in()
        self.get_logger().info(
            '[ASR] utterance received '
            f'id={msg.utterance_id} conf={float(msg.confidence):.2f} '
            f'interrupted_tts={bool(msg.interrupted_tts or local_tts_active)} text={self._short(msg.text)}'
        )
        with self._state_lock:
            turns = self._ingestor.accept(
                utterance_id=msg.utterance_id,
                text=msg.text,
                confidence=float(msg.confidence),
                captured_during_tts=bool(msg.interrupted_tts or local_tts_active),
                session_id=self._reducer.state.session_id,
            )
            for turn in turns:
                self._enqueue_or_buffer_turn(turn)

    def _cancel_active_tts_for_barge_in(self) -> None:
        with self._active_tts_goal_lock:
            goal_handle = self._active_tts_goal_handle
            command_id = self._active_tts_command_id
        if goal_handle is None:
            return
        try:
            self.get_logger().info(f'[TTS] cancel requested for active command id={command_id} due to barge-in')
            goal_handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'[TTS] barge-in cancel request failed: {exc}')

    def _enqueue_or_buffer_turn(self, turn: TurnEnvelopeData) -> None:
        if self._dependencies_ready():
            self._event_queue.put(_QueueTurnEvent(turn))
            return
        self._pending_turn_events.append(turn)
        self.get_logger().warn(
            f'[PIPELINE] buffered turn seq={turn.turn_seq} until conversation backends are ready'
        )

    def _on_llm_status(self, msg: LlmStatus) -> None:
        ready = msg.status == LlmStatus.READY
        with self._state_lock:
            if ready == self._llm_backend_ready:
                return
            self._llm_backend_ready = ready
            self._llm_model_name = str(msg.model_name or '').strip()
            if ready and self._llm_model_name:
                self._direct_vllm_client = VllmClient(
                    host=self._llm_vllm_host,
                    port=self._llm_vllm_port,
                    model_name=self._llm_model_name,
                    timeout=self._llm_request_timeout_sec,
                )
            state = 'READY' if ready else 'NOT_READY'
            self.get_logger().info(
                f'[LLM] backend status changed: {state} model={msg.model_name} message={msg.message}'
            )

    def _on_chat_incoming(self, msg: ChatMessage) -> None:
        text = msg.text.strip()
        if not text:
            return
        if msg.message_id and msg.message_id in self._seen_secretary_messages:
            return
        if msg.message_id:
            self._seen_secretary_messages.add(msg.message_id)
        self._event_queue.put(
            _QueueSecretaryEvent(
                SecretaryReplyData(
                    thread_id=msg.thread_id,
                    message_id=msg.message_id,
                    text=text,
                )
            )
        )

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                item = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                with self._state_lock:
                    if isinstance(item, _QueueTurnEvent):
                        self._process_turn(item.turn)
                    elif isinstance(item, _QueueSecretaryEvent):
                        self._process_secretary_reply(item.reply)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'worker error: {exc}')

    def _chat_outbox_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                item_id = self._chat_outbox_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_chat_outbox_item(item_id)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'chat outbox worker error: {exc}')

    def _process_turn(self, turn: TurnEnvelopeData) -> None:
        if not self._dependencies_ready():
            self._pending_turn_events.append(turn)
            self.get_logger().warn(
                f'[PIPELINE] re-buffered turn seq={turn.turn_seq} because backend became unavailable'
            )
            return

        self._publish_user_trace(turn)
        self.get_logger().info(
            '[PIPELINE] turn finalized '
            f'seq={turn.turn_seq} utterance_id={turn.utterance_id} '
            f'captured_during_tts={turn.captured_during_tts} text={self._short(turn.text)}'
        )
        decision = self._call_extract_stage(turn)
        outcome = self._reducer.apply(
            turn_seq=turn.turn_seq,
            utterance_text=turn.text,
            decision=decision,
        )
        state = self._reducer.state
        self.get_logger().info(
            '[REDUCER] applied '
            f'seq={turn.turn_seq} phase={state.phase} '
            f'response_language={state.response_language} '
            f'dialog_act={outcome.dialog_act} '
            f'working=name:{state.working_info.name or "-"},'
            f'affiliation:{state.working_info.affiliation or "-"},'
            f'purpose:{state.working_info.purpose or "-"} '
            f'committed=name:{state.committed_info.name or "-"},'
            f'affiliation:{state.committed_info.affiliation or "-"},'
            f'purpose:{state.committed_info.purpose or "-"}'
        )
        self._publish_state()
        self._publish_reducer_traces(outcome, turn_seq=turn.turn_seq)

        for item in outcome.outbox_items:
            self._chat_outbox_queue.put(item.item_id)

        if not outcome.should_render_response:
            self.get_logger().info(
                f'[PIPELINE] skip render seq={turn.turn_seq} (stale or no response needed)'
            )
            return

        text = self._call_render_stage(
            turn_seq=turn.turn_seq,
            dialog_act=outcome.dialog_act,
            latest_user_text=turn.text,
            secretary_reply_text='',
        )

        tts_command = OrchestratorCommandData(
            command_type=ExecutionCommand.COMMAND_TTS,
            command_id=uuid4().hex,
            session_id=self._reducer.state.session_id,
            turn_seq=turn.turn_seq,
            payload_json=json.dumps(
                {
                    'text': text,
                    'dialog_act': outcome.dialog_act,
                    'language': self._reducer.state.response_language,
                },
                ensure_ascii=False,
            ),
            dialog_act=outcome.dialog_act,
        )
        self._publish_assistant_trace(tts_command)
        self._effect_executor.submit(tts_command, immediate_non_tts=self._invoke_non_tts_command)

    def _process_secretary_reply(self, reply: SecretaryReplyData) -> None:
        self.get_logger().info(
            f'[SECRETARY] incoming thread={reply.thread_id} text={self._short(reply.text)}'
        )
        outcome = self._reducer.handle_secretary_reply(reply)
        if outcome is None:
            self.get_logger().info('[SECRETARY] ignored (phase/thread mismatch or duplicate)')
            return
        self._publish_reducer_traces(outcome, turn_seq=outcome.turn_seq)
        self._publish_state()

        tts_command = OrchestratorCommandData(
            command_type=ExecutionCommand.COMMAND_TTS,
            command_id=uuid4().hex,
            session_id=outcome.session_id,
            turn_seq=outcome.turn_seq,
            payload_json=json.dumps(
                {
                    'text': reply.text,
                    'dialog_act': 'relay_secretary',
                    'language': self._reducer.state.response_language,
                },
                ensure_ascii=False,
            ),
            dialog_act='relay_secretary',
        )
        self._publish_assistant_trace(tts_command)
        self._effect_executor.submit(tts_command, immediate_non_tts=self._invoke_non_tts_command)

    def _submit_ready_announcement(self) -> None:
        self.get_logger().info('[TTS] ready announcement submitting bilingual segments')
        for index, (text, language) in enumerate(self._READY_ANNOUNCEMENT_SEGMENTS):
            tts_command = OrchestratorCommandData(
                command_type=ExecutionCommand.COMMAND_TTS,
                command_id=uuid4().hex,
                session_id=self._reducer.state.session_id,
                turn_seq=index,
                payload_json=json.dumps(
                    {'text': text, 'dialog_act': 'system_ready', 'language': language},
                    ensure_ascii=False,
                ),
                dialog_act='system_ready',
            )
            self._publish_system_trace(tts_command)
            self._effect_executor.submit(tts_command, immediate_non_tts=self._invoke_non_tts_command)

    def _process_chat_outbox_item(self, item_id: str) -> None:
        with self._state_lock:
            item = self._find_chat_outbox_item(item_id)
            if item is None or item.status == 'sent':
                return
            item.status = 'sending'
            item.attempt_count += 1
            state = self._reducer.state
            state.chat_delivery_state = 'sending'
            title = item.title
            text = item.text
            thread_id = item.thread_id or state.discord_thread_id
            self._publish_state()

        try:
            if not thread_id:
                response = self._call_create_thread_service(thread_title=title, initial_text=text)
                if not response.success:
                    raise RuntimeError(response.error_message or 'create_thread failed')
                thread_id = response.thread_id
                with self._state_lock:
                    state = self._reducer.state
                    state.discord_thread_id = response.thread_id
                    state.discord_channel_id = response.channel_id
                    item = self._find_chat_outbox_item(item_id)
                    if item is not None:
                        item.thread_id = response.thread_id
                    self._publish_state()
                    self.get_logger().info(
                        f'[DISCORD] thread created thread_id={response.thread_id} channel_id={response.channel_id}'
                    )
                message_id = response.message_id
            else:
                response = self._call_send_message_service(thread_id, text)
                message_id = response.message_id

            with self._state_lock:
                state = self._reducer.state
                item = self._find_chat_outbox_item(item_id)
                if item is not None:
                    item.status = 'sent'
                    item.thread_id = thread_id
                state.chat_delivery_state = 'sent'
                self._publish_state()
                self._publish_trace_event(
                    TraceEventData(
                        event_type='CHAT_SENT',
                        payload_json=json.dumps(
                            {
                                'item_id': item_id,
                                'thread_id': thread_id,
                                'message_id': message_id,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                    turn_seq=getattr(item, 'turn_seq', state.latest_applied_turn),
                )
                self.get_logger().info(
                    f'[DISCORD] message sent thread_id={thread_id} message_id={message_id}'
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[DISCORD] outbox failed item_id={item_id}: {exc}')
            with self._state_lock:
                state = self._reducer.state
                item = self._find_chat_outbox_item(item_id)
                if item is None:
                    return
                if item.attempt_count < 3:
                    item.status = 'retry_pending'
                    state.chat_delivery_state = 'retry_pending'
                    self._publish_state()
                    self._schedule_outbox_retry(item_id)
                else:
                    item.status = 'failed'
                    state.chat_delivery_state = 'failed'
                    self._publish_state()

    def _schedule_outbox_retry(self, item_id: str) -> None:
        if self._shutdown_event.is_set():
            return

        def _requeue() -> None:
            if not self._shutdown_event.is_set():
                self._chat_outbox_queue.put(item_id)

        timer = threading.Timer(1.0, _requeue)
        timer.daemon = True
        timer.start()

    def _find_chat_outbox_item(self, item_id: str) -> ChatOutboxItemData | None:
        for item in self._reducer.state.chat_outbox:
            if item.item_id == item_id:
                return item
        return None

    def _call_extract_stage(self, turn: TurnEnvelopeData) -> SemanticDecisionData:
        cmd = OrchestratorCommandData(
            command_type=ExecutionCommand.COMMAND_UNKNOWN,
            command_id=f'extract-{turn.turn_seq}-{uuid4().hex[:8]}',
            session_id=turn.session_id,
            turn_seq=turn.turn_seq,
            payload_json='{}',
            dialog_act='',
        )
        self._publish_execution_event(cmd, ExecutionEvent.STATUS_STARTED, ExecutionEvent.REASON_NONE, 'extract_started')
        self.get_logger().info(
            f'[LLM-S1] extract start seq={turn.turn_seq} text={self._short(turn.text)}'
        )

        try:
            decision = self._normalize_semantic_decision(
                self._call_extract_stage_action(turn),
                utterance_text=turn.text,
            )
            decision = self._normalize_semantic_decision(
                self._apply_confirmation_rescue_if_needed(turn, decision),
                utterance_text=turn.text,
            )
            if self._decision_needs_stage_rescue(decision):
                rescued = self._rescue_direct_semantic_decision(turn, decision)
                if rescued is not None:
                    decision = self._normalize_semantic_decision(rescued, utterance_text=turn.text)
                else:
                    raise RuntimeError('direct extract requires semantic rescue')
            decision = self._normalize_semantic_decision(self._refine_long_slot_decision(turn, decision), utterance_text=turn.text)
            decision = self._normalize_semantic_decision(self._normalize_slot_operation_values(turn, decision), utterance_text=turn.text)
            self._publish_execution_event(cmd, ExecutionEvent.STATUS_SUCCEEDED, ExecutionEvent.REASON_NONE, 'extract_succeeded')
            self.get_logger().info(
                '[LLM-S1] stage result '
                f'seq={turn.turn_seq} speech_act={decision.speech_act} '
                f'conf={decision.confidence:.2f} '
                f'target_slot={decision.target_slot} '
                f'ops={self._short(json.dumps([self._operation_to_dict(op) for op in decision.operations], ensure_ascii=False))}'
            )
            return decision
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[LLM-S1] stage failed seq={turn.turn_seq}: {exc}')
            try:
                decision = self._normalize_semantic_decision(self._call_extract_direct_llm(turn), utterance_text=turn.text)
                decision = self._normalize_semantic_decision(
                    self._apply_confirmation_rescue_if_needed(turn, decision),
                    utterance_text=turn.text,
                )
                if self._decision_needs_stage_rescue(decision):
                    rescued = self._rescue_direct_semantic_decision(turn, decision)
                    if rescued is not None:
                        decision = self._normalize_semantic_decision(
                            rescued,
                            utterance_text=turn.text,
                        )
                    else:
                        raise RuntimeError('direct extract requires semantic rescue')
                decision = self._normalize_semantic_decision(
                    self._refine_long_slot_decision(turn, decision),
                    utterance_text=turn.text,
                )
                decision = self._normalize_semantic_decision(
                    self._normalize_slot_operation_values(turn, decision),
                    utterance_text=turn.text,
                )
                self._publish_execution_event(
                    cmd,
                    ExecutionEvent.STATUS_SUCCEEDED,
                    ExecutionEvent.REASON_NONE,
                    'extract_succeeded_via_direct_fallback',
                )
                self.get_logger().info(
                    '[LLM-S1] direct fallback result '
                    f'seq={turn.turn_seq} speech_act={decision.speech_act} '
                    f'conf={decision.confidence:.2f} '
                    f'target_slot={decision.target_slot} '
                    f'ops={self._short(json.dumps([self._operation_to_dict(op) for op in decision.operations], ensure_ascii=False))}'
                )
                return decision
            except Exception as action_exc:  # noqa: BLE001
                self._publish_execution_event(
                    cmd,
                    ExecutionEvent.STATUS_FAILED,
                    ExecutionEvent.REASON_INTERNAL_ERROR,
                    f'extract_failed:{action_exc}',
                )
                self.get_logger().error(
                    f'[LLM-S1] action fallback failed seq={turn.turn_seq}: {action_exc}'
                )
                heuristic = self._heuristic_extract_decision(turn.turn_seq, turn.text)
                self.get_logger().warn(
                    '[LLM-S1] heuristic fallback '
                    f'seq={turn.turn_seq} speech_act={heuristic.speech_act} '
                    f'target_slot={heuristic.target_slot}'
                )
                return heuristic

    def _apply_confirmation_rescue_if_needed(
        self,
        turn: TurnEnvelopeData,
        decision: SemanticDecisionData,
    ) -> SemanticDecisionData:
        if self._reducer.state.phase != 'confirming':
            return decision
        rescued = self._rescue_direct_semantic_decision(turn, decision)
        if rescued is None:
            return decision
        if rescued.speech_act in {'affirm', 'deny', 'correction'}:
            return rescued
        return decision

    def _refine_long_slot_decision(
        self,
        turn: TurnEnvelopeData,
        decision: SemanticDecisionData,
    ) -> SemanticDecisionData:
        substantive_ops = [
            op for op in decision.operations
            if op.op in {'set_slot', 'replace_slot'} and op.slot in {'name', 'affiliation', 'purpose'} and op.value
        ]
        if self._reducer.state.phase != 'collecting':
            return decision
        if len(substantive_ops) != 1:
            return decision
        if len(turn.text.strip()) < 12:
            return decision

        snapshot = self._legacy_snapshot_for_rescue(turn)
        try:
            raw = self._invoke_chat_action(
                session_id=f'{turn.session_id}:extract-long-slot-refine:{turn.turn_seq}',
                user_message=build_reception_slot_extract_prompt(
                    snapshot,
                    turn.text,
                    target_fields=['name', 'affiliation', 'purpose'],
                ),
                system_prompt=RECEPTION_REPAIR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=96,
                timeout_sec=20.0,
                response_json_schema=RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
            )
            payload = extract_json_object(raw)
        except Exception:  # noqa: BLE001
            return decision
        if not isinstance(payload, dict):
            return decision

        extracted = {
            slot: str(payload.get(slot) or '').strip()
            for slot in ('name', 'affiliation', 'purpose')
            if str(payload.get(slot) or '').strip()
        }
        refined_slot, refined_value = self._choose_refined_slot_candidate(
            turn=turn,
            extracted=extracted,
        )
        if refined_slot == 'none' or not refined_value:
            return decision

        current_op = substantive_ops[0]
        if refined_slot == current_op.slot and refined_value == current_op.value:
            return decision

        updated_operations: list[BeliefOperationData] = []
        for operation in decision.operations:
            if operation is current_op:
                updated_operations.append(
                    BeliefOperationData(
                        op=operation.op,
                        slot=refined_slot,
                        value=refined_value,
                        grounded_text=turn.text,
                        confidence=operation.confidence,
                    )
                )
            else:
                updated_operations.append(operation)

        return SemanticDecisionData(
            turn_seq=decision.turn_seq,
            speech_act=decision.speech_act,
            detected_language=decision.detected_language,
            target_slot=refined_slot,
            ambiguity=decision.ambiguity,
            requires_confirmation=decision.requires_confirmation,
            confidence=decision.confidence,
            evidence=f'{decision.evidence}|long_slot_refine',
            operations=updated_operations,
            grounded_segments=[refined_value],
        )

    def _normalize_slot_operation_values(
        self,
        turn: TurnEnvelopeData,
        decision: SemanticDecisionData,
    ) -> SemanticDecisionData:
        substantive_ops = [
            op for op in decision.operations
            if op.op in {'set_slot', 'replace_slot'} and op.slot in {'name', 'affiliation', 'purpose'} and op.value
        ]
        if not substantive_ops:
            return decision

        snapshot = self._legacy_snapshot_for_rescue(turn)
        extracted_name = None
        extracted_affiliation = None
        extracted_purpose = None
        for operation in substantive_ops:
            if operation.slot == 'name':
                extracted_name = operation.value
            elif operation.slot == 'affiliation':
                extracted_affiliation = operation.value
            elif operation.slot == 'purpose':
                extracted_purpose = operation.value

        try:
            raw = self._invoke_chat_action(
                session_id=f'{turn.session_id}:slot-normalize:{turn.turn_seq}',
                user_message=build_reception_slot_normalize_prompt(
                    snapshot,
                    turn.text,
                    extracted_name=extracted_name,
                    extracted_affiliation=extracted_affiliation,
                    extracted_purpose=extracted_purpose,
                ),
                system_prompt=RECEPTION_SLOT_NORMALIZE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=96,
                timeout_sec=20.0,
                response_json_schema=RECEPTION_SLOT_NORMALIZE_JSON_SCHEMA,
            )
            payload = extract_json_object(raw)
        except Exception:  # noqa: BLE001
            return decision
        if not isinstance(payload, dict):
            return decision

        normalized_values = {
            'name': self._postprocess_normalized_slot_value('name', payload.get('name')),
            'affiliation': self._postprocess_normalized_slot_value('affiliation', payload.get('affiliation')),
            'purpose': self._postprocess_normalized_slot_value('purpose', payload.get('purpose')),
        }
        if not any(value is not None for value in normalized_values.values()):
            return decision

        updated = False
        updated_operations: list[BeliefOperationData] = []
        grounded_segments = list(decision.grounded_segments)
        for operation in decision.operations:
            if operation.op in {'set_slot', 'replace_slot'} and operation.slot in normalized_values:
                normalized_value = normalized_values[operation.slot]
                if normalized_value is None:
                    updated = True
                    continue
                if normalized_value != operation.value:
                    updated = True
                    updated_operations.append(
                        BeliefOperationData(
                            op=operation.op,
                            slot=operation.slot,
                            value=normalized_value,
                            grounded_text=operation.grounded_text or turn.text,
                            confidence=operation.confidence,
                        )
                    )
                    grounded_segments.append(normalized_value)
                    continue
            updated_operations.append(operation)

        if not updated:
            return decision

        return SemanticDecisionData(
            turn_seq=decision.turn_seq,
            speech_act=decision.speech_act,
            detected_language=decision.detected_language,
            target_slot=decision.target_slot,
            ambiguity=decision.ambiguity,
            requires_confirmation=decision.requires_confirmation,
            confidence=decision.confidence,
            evidence=f'{decision.evidence}|slot_normalized',
            operations=updated_operations,
            grounded_segments=grounded_segments,
        )

    def _choose_refined_slot_candidate(
        self,
        *,
        turn: TurnEnvelopeData,
        extracted: dict[str, str],
    ) -> tuple[str, str]:
        if len(extracted) == 1:
            return next(iter(extracted.items()))

        per_slot_candidates: dict[str, str] = {}
        for slot in ('name', 'affiliation', 'purpose'):
            recovered = self._recover_single_slot_value(turn, slot)
            if recovered:
                per_slot_candidates[slot] = recovered

        if len(per_slot_candidates) == 1:
            return next(iter(per_slot_candidates.items()))

        return ('none', '')

    def _recover_single_slot_value(self, turn: TurnEnvelopeData, target_slot: str) -> str | None:
        snapshot = self._legacy_snapshot_for_rescue(turn)
        try:
            raw = self._invoke_chat_action(
                session_id=f'{turn.session_id}:extract-long-slot-refine:{turn.turn_seq}:{target_slot}',
                user_message=build_reception_slot_extract_prompt(
                    snapshot,
                    turn.text,
                    target_fields=[target_slot],
                ),
                system_prompt=RECEPTION_REPAIR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=64,
                timeout_sec=20.0,
                response_json_schema=RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
            )
            payload = extract_json_object(raw)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        value = str(payload.get(target_slot) or '').strip()
        return value or None

    def _rescue_direct_semantic_decision(
        self,
        turn: TurnEnvelopeData,
        decision: SemanticDecisionData,
    ) -> SemanticDecisionData | None:
        snapshot = self._legacy_snapshot_for_rescue(turn)
        if self._reducer.state.phase == 'confirming':
            try:
                raw = self._invoke_chat_action(
                    session_id=f'{turn.session_id}:extract-confirm-direct-rescue:{turn.turn_seq}',
                    user_message=build_reception_confirmation_rescue_prompt(snapshot, turn.text),
                    system_prompt=RECEPTION_REPAIR_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=96,
                    timeout_sec=20.0,
                    response_json_schema=RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA,
                )
                payload = extract_json_object(raw)
            except Exception:  # noqa: BLE001
                payload = None
            if isinstance(payload, dict):
                speech_act = str(payload.get('speech_act', '')).strip()
                correction = payload.get('correction', {})
                confirmation = payload.get('confirmation', {})
                target = self._sanitize_slot(correction.get('target', 'none')) if isinstance(correction, dict) else 'none'
                accepted = bool(confirmation.get('accepted')) if isinstance(confirmation, dict) else False
                if speech_act == 'affirm' or accepted:
                    return SemanticDecisionData(
                        turn_seq=turn.turn_seq,
                        speech_act='affirm',
                        detected_language=decision.detected_language,
                        target_slot='none',
                        ambiguity='low',
                        confidence=decision.confidence,
                        evidence='direct_confirmation_rescue',
                        operations=[BeliefOperationData(op='confirm_working_state', slot='none', grounded_text=turn.text, confidence=decision.confidence)],
                        grounded_segments=[turn.text],
                    )
                if speech_act in {'deny', 'correction'}:
                    return SemanticDecisionData(
                        turn_seq=turn.turn_seq,
                        speech_act=speech_act,
                        detected_language=decision.detected_language,
                        target_slot=target,
                        ambiguity='medium',
                        requires_confirmation=True,
                        confidence=decision.confidence,
                        evidence='direct_confirmation_rescue',
                        operations=[BeliefOperationData(op='reject_confirmation', slot=target, grounded_text=turn.text, confidence=decision.confidence)],
                        grounded_segments=[turn.text],
                    )

        target_slot = decision.target_slot
        if target_slot not in {'name', 'affiliation', 'purpose'}:
            return None
        try:
            raw = self._invoke_chat_action(
                session_id=f'{turn.session_id}:extract-slot-direct-rescue:{turn.turn_seq}',
                user_message=build_reception_slot_extract_prompt(snapshot, turn.text, target_fields=[target_slot]),
                system_prompt=RECEPTION_REPAIR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=96,
                timeout_sec=20.0,
                response_json_schema=RECEPTION_SLOT_EXTRACT_JSON_SCHEMA,
            )
            payload = extract_json_object(raw)
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict):
            return None
        rescued_value = str(payload.get(target_slot) or '').strip()
        if not rescued_value:
            return None
        current_value = getattr(self._reducer.state.working_info, target_slot)
        op_name = 'replace_slot' if current_value else 'set_slot'
        return SemanticDecisionData(
            turn_seq=turn.turn_seq,
            speech_act=decision.speech_act,
            detected_language=decision.detected_language,
            target_slot=target_slot,
            ambiguity='low' if decision.ambiguity == 'high' else decision.ambiguity,
            requires_confirmation=False,
            confidence=decision.confidence,
            evidence='direct_slot_rescue',
            operations=[BeliefOperationData(op=op_name, slot=target_slot, value=rescued_value, grounded_text=rescued_value, confidence=decision.confidence)],
            grounded_segments=[rescued_value],
        )

    def _legacy_snapshot_for_rescue(self, turn: TurnEnvelopeData) -> SessionSnapshot:
        state = self._reducer.state
        return SessionSnapshot(
            session_id=turn.session_id,
            phase=state.phase,
            visitor_info=VisitorInfo(
                name=state.working_info.name or None,
                affiliation=state.working_info.affiliation or None,
                purpose=state.working_info.purpose or None,
            ),
            last_user_utterance=turn.text,
            last_dialog_act=state.last_system_act or None,
            last_spoken_text='',
            pending_confirmation=(
                VisitorInfo(
                    name=state.committed_info.name or None,
                    affiliation=state.committed_info.affiliation or None,
                    purpose=state.committed_info.purpose or None,
                )
                if state.phase == 'confirming'
                else None
            ),
            latest_turn_id=turn.turn_seq,
        )

    def _call_render_stage(
        self,
        *,
        turn_seq: int,
        dialog_act: str,
        latest_user_text: str,
        secretary_reply_text: str,
    ) -> str:
        cmd = OrchestratorCommandData(
            command_type=ExecutionCommand.COMMAND_UNKNOWN,
            command_id=f'render-{turn_seq}-{uuid4().hex[:8]}',
            session_id=self._reducer.state.session_id,
            turn_seq=turn_seq,
            payload_json='{}',
            dialog_act=dialog_act,
        )
        self._publish_execution_event(cmd, ExecutionEvent.STATUS_STARTED, ExecutionEvent.REASON_NONE, 'render_started')
        self.get_logger().info(
            f'[LLM-S2] render start seq={turn_seq} dialog_act={dialog_act}'
        )

        fallback = _fallback_dialog_text(
            dialog_act,
            self._reducer.state.working_info.name,
            self._reducer.state.working_info.affiliation,
            self._reducer.state.working_info.purpose,
            self._reducer.state.response_language,
        )
        if dialog_act != 'relay_secretary':
            self._publish_execution_event(
                cmd,
                ExecutionEvent.STATUS_SUCCEEDED,
                ExecutionEvent.REASON_NONE,
                'render_template_fallback',
            )
            self.get_logger().info(
                f'[LLM-S2] template result seq={turn_seq} text={self._short(fallback)}'
            )
            return fallback

        text = secretary_reply_text.strip() or fallback
        self._publish_execution_event(
            cmd,
            ExecutionEvent.STATUS_SUCCEEDED,
            ExecutionEvent.REASON_NONE,
            'render_secretary_reply',
        )
        self.get_logger().info(
            f'[LLM-S2] secretary relay result seq={turn_seq} text={self._short(text)}'
        )
        return text

    def _invoke_non_tts_command(self, command: OrchestratorCommandData) -> tuple[bool, str]:
        del command
        return True, 'noop'

    def _invoke_tts_command(self, command: OrchestratorCommandData) -> tuple[bool, str]:
        try:
            payload = json.loads(command.payload_json or '{}')
        except json.JSONDecodeError:
            payload = {}
        text = str(payload.get('text', '')).strip()
        if not text:
            return False, 'tts text missing'
        language = self._sanitize_detected_language(
            payload.get('language', self._reducer.state.response_language)
        )
        if language == 'unknown':
            language = 'ja'
        self.get_logger().info(
            f'[TTS] start seq={command.turn_seq} dialog_act={payload.get("dialog_act", command.dialog_act)} '
            f'language={language} text={self._short(text)}'
        )

        if not self._tts_client.wait_for_server(timeout_sec=5.0):
            return False, 'tts action unavailable'

        goal = Speak.Goal()
        goal.request_id = command.command_id
        goal.session_id = command.session_id
        goal.text = text
        goal.language = language
        goal.voice = ''
        goal.volume = self._tts_volume
        goal.speed = 0.0
        goal.pitch = 0.0
        goal.priority = 0
        goal.interrupt = False
        goal.allow_streaming = False
        goal.save_wav = False

        try:
            goal_handle = wait_future(self._tts_client.send_goal_async(goal), timeout_sec=10.0)
            if goal_handle is None or not goal_handle.accepted:
                return False, 'tts goal rejected'
            with self._active_tts_goal_lock:
                self._active_tts_goal_handle = goal_handle
                self._active_tts_command_id = command.command_id
            wrapped = wait_future(goal_handle.get_result_async(), timeout_sec=180.0)
            if wrapped is None:
                return False, 'tts timeout'
            ok = bool(wrapped.result.ok)
            if not ok:
                self.get_logger().error(
                    f'[TTS] failed seq={command.turn_seq}: {wrapped.result.error_message or "tts failed"}'
                )
                return False, wrapped.result.error_message or 'tts failed'
            self.get_logger().info(f'[TTS] completed seq={command.turn_seq}')
            return True, 'tts_ok'
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[TTS] exception seq={command.turn_seq}: {exc}')
            return False, str(exc)
        finally:
            with self._active_tts_goal_lock:
                if self._active_tts_command_id == command.command_id:
                    self._active_tts_goal_handle = None
                    self._active_tts_command_id = ''

    def _load_tts_volume(self) -> float:
        raw = os.environ.get('RECEPTION_TTS_VOLUME', str(self._DEFAULT_TTS_VOLUME)).strip()
        try:
            volume = float(raw)
        except ValueError:
            self.get_logger().warning(
                f'Invalid RECEPTION_TTS_VOLUME={raw!r}; falling back to {self._DEFAULT_TTS_VOLUME}'
            )
            return self._DEFAULT_TTS_VOLUME
        return max(0.0, min(1.0, volume))

    def _on_command_completed(self, command: OrchestratorCommandData, ok: bool, detail: str) -> None:
        self.get_logger().info(
            f'[EFFECT] command completed id={command.command_id} type={command.command_type} '
            f'seq={command.turn_seq} ok={ok} detail={detail}'
        )
        if command.command_type != ExecutionCommand.COMMAND_TTS:
            return
        try:
            payload = json.loads(command.payload_json or '{}')
        except json.JSONDecodeError:
            payload = {}
        dialog_act = str(payload.get('dialog_act', command.dialog_act or ''))
        self._publish_trace_event(
            TraceEventData(
                event_type='TTS_COMPLETED' if ok else 'TTS_FAILED',
                dialog_act=dialog_act,
                payload_json=json.dumps(
                    {'ok': bool(ok), 'detail': detail},
                    ensure_ascii=False,
                ),
            ),
            turn_seq=command.turn_seq,
        )
        if ok:
            self._reducer.mark_tts_completed(turn_seq=command.turn_seq, dialog_act=dialog_act)
            self._publish_state()

    def _call_create_thread_service(self, *, thread_title: str, initial_text: str) -> CreateThread.Response:
        if not self._create_thread_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/chat_bridge/create_thread service is unavailable')
        request = CreateThread.Request()
        request.parent_target.target_type = ChatTarget.CHANNEL
        request.parent_target.adapter_name = self._discord_adapter_name
        request.parent_target.target_id = self._discord_parent_channel_id
        request.thread_title = thread_title
        request.initial_text = initial_text
        request.subscribe = True
        response = wait_future(self._create_thread_client.call_async(request), timeout_sec=10.0)
        if response is None:
            raise TimeoutError('create_thread timeout')
        return response

    def _call_send_message_service(self, thread_id: str, text: str) -> SendMessage.Response:
        if not self._send_message_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/chat_bridge/send_message service is unavailable')
        request = SendMessage.Request()
        request.target.target_type = ChatTarget.THREAD
        request.target.adapter_name = self._discord_adapter_name
        request.target.target_id = thread_id
        request.text = text
        response = wait_future(self._send_message_client.call_async(request), timeout_sec=6.0)
        if response is None:
            raise TimeoutError('send_message timeout')
        if not response.success:
            raise RuntimeError(response.error_message or 'send_message failed')
        return response

    def _call_extract_direct_llm(self, turn: TurnEnvelopeData) -> SemanticDecisionData:
        state = self._reducer.state
        prompt = (
            'Task: semantic extraction for reception flow.\n'
            f'phase={state.phase}\n'
            f'current_response_language={self._stage1_language_hint()}\n'
            f'working_name={state.working_info.name}\n'
            f'working_affiliation={state.working_info.affiliation}\n'
            f'working_purpose={state.working_info.purpose}\n'
            f'committed_name={state.committed_info.name}\n'
            f'committed_affiliation={state.committed_info.affiliation}\n'
            f'committed_purpose={state.committed_info.purpose}\n'
            f'focus_slot={state.focus_slot}\n'
            f'last_system_act={state.last_system_act}\n'
            f'pending_clarification_slot={state.pending_clarification_slot}\n'
            f'latest_utterance={turn.text}\n'
        )
        raw = self._invoke_chat_action(
            session_id=f'{turn.session_id}:extract-direct:{turn.turn_seq}',
            user_message=prompt,
            system_prompt=_STAGE1_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=220,
            timeout_sec=45.0,
            response_json_schema=_STAGE1_JSON_SCHEMA,
        )
        payload = extract_json_object(raw)
        if not isinstance(payload, dict):
            raise RuntimeError('direct extract returned non-JSON response')
        return self._decision_from_payload(turn.turn_seq, payload)

    def _invoke_chat_action(
        self,
        *,
        session_id: str,
        user_message: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
        response_json_schema: str = '',
    ) -> str:
        del session_id, timeout_sec
        client = self._direct_vllm_client
        if client is None:
            raise RuntimeError('direct vLLM client is not ready')
        completion = client.chat_completion(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stream=False,
            extra_body={},
            response_format=(
                {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'reception_turn_decision',
                        'schema': json.loads(response_json_schema),
                    },
                }
                if response_json_schema
                else None
            ),
        )
        text = extract_completion_text(completion).strip()
        if not text:
            raise RuntimeError('direct vLLM returned empty response')
        return text

    def _call_extract_stage_action(self, turn: TurnEnvelopeData) -> SemanticDecisionData:
        if not self._extract_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'{self._extract_action_name} action unavailable')

        goal = ExtractTurn.Goal()
        goal.turn = self._to_turn_msg(turn)
        goal.phase = self._reducer.state.phase
        goal.working_info = self._to_visitor_msg(self._reducer.state.working_info)
        goal.committed_info = self._to_visitor_msg(self._reducer.state.committed_info)
        goal.focus_slot = self._reducer.state.focus_slot
        goal.last_system_act = self._reducer.state.last_system_act
        goal.pending_clarification_slot = self._reducer.state.pending_clarification_slot
        goal.current_response_language = self._stage1_language_hint()

        goal_handle = wait_future(self._extract_client.send_goal_async(goal), timeout_sec=10.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f'{self._extract_action_name} goal rejected')

        wrapped = wait_future(goal_handle.get_result_async(), timeout_sec=90.0)
        if wrapped is None:
            raise RuntimeError(f'{self._extract_action_name} timeout waiting result')

        result = wrapped.result
        if result is None:
            raise RuntimeError(f'{self._extract_action_name} returned no result')

        decision = result.decision
        return SemanticDecisionData(
            turn_seq=int(decision.turn_seq),
            speech_act=str(decision.speech_act),
            detected_language=self._sanitize_detected_language(decision.detected_language),
            target_slot=self._sanitize_slot(decision.target_slot),
            ambiguity=self._sanitize_ambiguity(decision.ambiguity),
            requires_confirmation=bool(decision.requires_confirmation),
            confidence=float(decision.confidence),
            evidence=str(decision.evidence or 'extract_stage_action'),
            operations=[self._belief_operation_data_from_msg(op) for op in decision.operations],
            grounded_segments=[str(item).strip() for item in decision.grounded_segments if str(item).strip()],
        )

    def _publish_state(self) -> None:
        state = self._reducer.state
        msg = SessionStateV2()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.session_id = state.session_id
        msg.phase = state.phase
        msg.response_language = state.response_language
        msg.working_info = self._to_visitor_msg(state.working_info)
        msg.committed_info = self._to_visitor_msg(state.committed_info)
        msg.focus_slot = state.focus_slot
        msg.last_system_act = state.last_system_act
        msg.pending_clarification_slot = state.pending_clarification_slot
        msg.working_provenance = [
            self._to_slot_provenance_msg(item)
            for _, item in sorted(state.working_provenance.items(), key=lambda pair: pair[0])
        ]
        msg.chat_outbox = [self._to_chat_outbox_msg(item) for item in state.chat_outbox]
        msg.chat_delivery_state = state.chat_delivery_state
        msg.discord_thread_id = state.discord_thread_id
        msg.discord_channel_id = state.discord_channel_id
        msg.latest_applied_turn = int(state.latest_applied_turn)
        msg.version = int(state.version)
        self._session_state_publisher.publish(msg)

    def _publish_user_trace(self, turn: TurnEnvelopeData) -> None:
        payload = json.dumps(
            {
                'captured_during_tts': bool(turn.captured_during_tts),
                'asr_confidence': float(turn.asr_confidence),
            },
            ensure_ascii=False,
        )
        self._publish_conversation_trace(
            role=ConversationTrace.ROLE_USER,
            session_id=turn.session_id,
            turn_seq=turn.turn_seq,
            text=turn.text,
            dialog_act='',
            phase=self._reducer.state.phase,
            utterance_id=turn.utterance_id,
            asr_confidence=turn.asr_confidence,
            event_type='UTTERANCE_RECEIVED',
            event_payload=payload,
            payload_json=payload,
        )

    def _publish_assistant_trace(self, command: OrchestratorCommandData) -> None:
        payload = self._decode_command_payload(command)
        payload_json = json.dumps(payload, ensure_ascii=False)
        self._publish_conversation_trace(
            role=ConversationTrace.ROLE_ASSISTANT,
            session_id=command.session_id,
            turn_seq=command.turn_seq,
            text=str(payload.get('text', '')),
            dialog_act=str(payload.get('dialog_act', command.dialog_act or '')),
            phase=self._reducer.state.phase,
            utterance_id='',
            asr_confidence=0.0,
            event_type='TTS_REQUESTED',
            event_payload=payload_json,
            payload_json=payload_json,
        )

    def _publish_system_trace(self, command: OrchestratorCommandData) -> None:
        payload = self._decode_command_payload(command)
        payload_json = json.dumps(payload, ensure_ascii=False)
        self._publish_conversation_trace(
            role=ConversationTrace.ROLE_SYSTEM,
            session_id=command.session_id,
            turn_seq=command.turn_seq,
            text=str(payload.get('text', '')),
            dialog_act=str(payload.get('dialog_act', command.dialog_act or '')),
            phase=self._reducer.state.phase,
            utterance_id='',
            asr_confidence=0.0,
            event_type='TTS_REQUESTED',
            event_payload=payload_json,
            payload_json=payload_json,
        )

    def _publish_reducer_traces(self, outcome: Any, *, turn_seq: int) -> None:
        for trace in outcome.trace_events:
            self._publish_trace_event(trace, turn_seq=turn_seq)

    def _publish_trace_event(self, trace: TraceEventData, *, turn_seq: int) -> None:
        role = ConversationTrace.ROLE_SYSTEM
        if trace.role == 'assistant':
            role = ConversationTrace.ROLE_ASSISTANT
        elif trace.role == 'user':
            role = ConversationTrace.ROLE_USER
        payload_json = trace.payload_json or ''
        self._publish_conversation_trace(
            role=role,
            session_id=self._reducer.state.session_id,
            turn_seq=turn_seq,
            text=trace.text,
            dialog_act=trace.dialog_act,
            phase=self._reducer.state.phase,
            utterance_id='',
            asr_confidence=0.0,
            event_type=trace.event_type,
            event_payload=payload_json,
            payload_json=payload_json,
        )

    def _publish_conversation_trace(
        self,
        *,
        role: int,
        session_id: str,
        turn_seq: int,
        text: str,
        dialog_act: str,
        phase: str,
        utterance_id: str,
        asr_confidence: float,
        event_type: str,
        event_payload: str,
        payload_json: str,
    ) -> None:
        msg = build_conversation_trace_message(
            timestamp=self.get_clock().now().to_msg(),
            session_id=session_id,
            turn_seq=turn_seq,
            role=role,
            text=text,
            dialog_act=dialog_act,
            phase=phase,
            utterance_id=utterance_id,
            asr_confidence=asr_confidence,
            event_type=event_type,
            event_payload=event_payload,
            payload_json=payload_json,
        )
        self._conversation_trace_publisher.publish(msg)
        try:
            self._conversation_log_writer.record(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'conversation log writer failed: {exc}')

    def _publish_execution_event(
        self,
        command: OrchestratorCommandData,
        status: int,
        reason_code: int,
        detail: str,
    ) -> None:
        event = ExecutionEvent()
        event.timestamp = self.get_clock().now().to_msg()
        event.command_id = command.command_id
        event.command_type = int(command.command_type)
        event.session_id = command.session_id
        event.turn_seq = int(command.turn_seq)
        event.status = int(status)
        event.reason_code = int(reason_code)
        event.detail = str(detail)
        self._event_publisher.publish(event)

    @staticmethod
    def _decode_command_payload(command: OrchestratorCommandData) -> dict[str, Any]:
        try:
            payload = json.loads(command.payload_json or '{}')
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _to_turn_msg(turn: TurnEnvelopeData) -> TurnEnvelope:
        msg = TurnEnvelope()
        msg.session_id = turn.session_id
        msg.turn_seq = int(turn.turn_seq)
        msg.utterance_id = turn.utterance_id
        msg.text = turn.text
        msg.captured_during_tts = bool(turn.captured_during_tts)
        msg.asr_confidence = float(turn.asr_confidence)
        return msg

    @staticmethod
    def _to_visitor_msg(info: VisitorInfoData) -> VisitorInfo:
        msg = VisitorInfo()
        msg.name = info.name
        msg.affiliation = info.affiliation
        msg.purpose = info.purpose
        return msg

    @staticmethod
    def _to_slot_provenance_msg(provenance: SlotProvenanceData) -> SlotProvenance:
        msg = SlotProvenance()
        msg.slot = provenance.slot
        msg.source_turn_seq = int(provenance.source_turn_seq)
        msg.grounded_text = provenance.grounded_text
        msg.confidence = float(provenance.confidence)
        msg.updated_at = provenance.updated_at
        return msg

    @staticmethod
    def _to_chat_outbox_msg(item: ChatOutboxItemData) -> ChatOutboxItem:
        msg = ChatOutboxItem()
        msg.cursor = int(item.cursor)
        msg.item_id = item.item_id
        msg.session_id = item.session_id
        msg.turn_seq = int(item.turn_seq)
        msg.event_type = item.event_type
        msg.thread_id = item.thread_id
        msg.title = item.title
        msg.text = item.text
        msg.attempt_count = int(item.attempt_count)
        msg.status = item.status
        return msg

    @staticmethod
    def _belief_operation_data_from_msg(msg: BeliefOperation) -> BeliefOperationData:
        return BeliefOperationData(
            op=str(msg.op or 'ignore').strip(),
            slot=ReceptionOrchestratorNodeV2._sanitize_slot(msg.slot),
            value=str(msg.value or '').strip(),
            grounded_text=str(msg.grounded_text or '').strip(),
            confidence=float(msg.confidence),
        )

    @staticmethod
    def _decision_from_payload(turn_seq: int, payload: dict[str, Any]) -> SemanticDecisionData:
        raw_operations = payload.get('operations', [])
        operations: list[BeliefOperationData] = []
        if isinstance(raw_operations, list):
            for raw_operation in raw_operations:
                if not isinstance(raw_operation, dict):
                    continue
                operations.append(
                    BeliefOperationData(
                        op=str(raw_operation.get('op', 'ignore')).strip(),
                        slot=ReceptionOrchestratorNodeV2._sanitize_slot(raw_operation.get('slot', 'none')),
                        value=str(raw_operation.get('value') or '').strip(),
                        grounded_text=str(raw_operation.get('grounded_text') or '').strip(),
                        confidence=float(raw_operation.get('confidence', 0.0) or 0.0),
                    )
                )
        return SemanticDecisionData(
            turn_seq=turn_seq,
            speech_act=str(payload.get('speech_act', 'unknown')),
            detected_language=ReceptionOrchestratorNodeV2._sanitize_detected_language(
                payload.get('detected_language', 'unknown')
            ),
            target_slot=ReceptionOrchestratorNodeV2._sanitize_slot(payload.get('target_slot', 'none')),
            ambiguity=ReceptionOrchestratorNodeV2._sanitize_ambiguity(payload.get('ambiguity', 'high')),
            requires_confirmation=bool(payload.get('requires_confirmation', False)),
            confidence=float(payload.get('confidence', 0.0) or 0.0),
            evidence=str(payload.get('evidence', 'direct_llm')),
            operations=operations,
            grounded_segments=[str(item).strip() for item in payload.get('grounded_segments', []) if str(item).strip()],
        )

    @staticmethod
    def _heuristic_extract_decision(turn_seq: int, text: str) -> SemanticDecisionData:
        utterance = str(text or '').strip()
        if not utterance:
            return SemanticDecisionData(
                turn_seq=turn_seq,
                speech_act='unknown',
                detected_language='unknown',
                target_slot='none',
                ambiguity='high',
                requires_confirmation=False,
                confidence=0.0,
                evidence='orchestrator_heuristic_ignore',
                operations=[BeliefOperationData(op='ignore', slot='none', confidence=0.0)],
            )
        return SemanticDecisionData(
            turn_seq=turn_seq,
            speech_act='unknown',
            detected_language='unknown',
            target_slot='none',
            ambiguity='high',
            requires_confirmation=True,
            confidence=0.0,
            evidence='orchestrator_heuristic_request_clarification',
            operations=[BeliefOperationData(op='request_clarification', slot='none', confidence=0.0)],
        )

    @staticmethod
    def _normalize_semantic_decision(
        decision: SemanticDecisionData,
        *,
        utterance_text: str = '',
    ) -> SemanticDecisionData:
        operations: list[BeliefOperationData] = []
        for operation in decision.operations:
            op_name = str(operation.op or 'ignore').strip()
            if op_name not in {
                'set_slot',
                'replace_slot',
                'clear_slot',
                'confirm_working_state',
                'reject_confirmation',
                'request_clarification',
                'ignore',
            }:
                op_name = 'ignore'
            operations.append(
                BeliefOperationData(
                    op=op_name,
                    slot=ReceptionOrchestratorNodeV2._sanitize_slot(operation.slot),
                    value=str(operation.value or '').strip(),
                    grounded_text=str(operation.grounded_text or '').strip(),
                    confidence=float(operation.confidence or 0.0),
                )
            )
        detected_language = ReceptionOrchestratorNodeV2._sanitize_detected_language(decision.detected_language)
        inferred_language = ReceptionOrchestratorNodeV2._infer_detected_language_from_utterance(utterance_text)
        if inferred_language == 'en':
            detected_language = 'en'
        elif detected_language == 'unknown' and inferred_language == 'ja':
            detected_language = 'ja'

        return SemanticDecisionData(
            turn_seq=decision.turn_seq,
            speech_act=str(decision.speech_act or 'unknown'),
            detected_language=detected_language,
            target_slot=ReceptionOrchestratorNodeV2._sanitize_slot(decision.target_slot),
            ambiguity=ReceptionOrchestratorNodeV2._sanitize_ambiguity(decision.ambiguity),
            requires_confirmation=bool(decision.requires_confirmation),
            confidence=float(decision.confidence or 0.0),
            evidence=str(decision.evidence or ''),
            operations=operations,
            grounded_segments=[str(item).strip() for item in decision.grounded_segments if str(item).strip()],
        )

    @staticmethod
    def _decision_needs_stage_rescue(decision: SemanticDecisionData) -> bool:
        has_substantive_op = False
        for operation in decision.operations:
            if operation.op in {'set_slot', 'replace_slot'} and operation.slot in {'name', 'affiliation', 'purpose'} and operation.value:
                has_substantive_op = True
                break
            if operation.op in {'confirm_working_state', 'reject_confirmation'}:
                has_substantive_op = True
                break
        if has_substantive_op:
            return False
        if decision.speech_act in {'inform', 'affirm', 'deny', 'correction'}:
            return True
        return False

    @staticmethod
    def _operation_to_dict(operation: BeliefOperationData) -> dict[str, object]:
        return {
            'op': operation.op,
            'slot': operation.slot,
            'value': operation.value,
            'grounded_text': operation.grounded_text,
            'confidence': float(operation.confidence),
        }

    @staticmethod
    def _sanitize_detected_language(value: object) -> str:
        candidate = str(value or 'unknown').strip().lower()
        if candidate in {'ja', 'en'}:
            return candidate
        return 'unknown'

    @staticmethod
    def _infer_detected_language_from_utterance(value: object) -> str:
        text = str(value or '').strip()
        if not text:
            return 'unknown'
        latin_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        japanese_chars = sum(
            1
            for ch in text
            if ('\u3040' <= ch <= '\u30ff') or ('\u4e00' <= ch <= '\u9fff')
        )
        if latin_letters >= 4 and japanese_chars == 0:
            return 'en'
        if japanese_chars >= 2 and latin_letters == 0:
            return 'ja'
        return 'unknown'

    @staticmethod
    def _sanitize_slot(value: object) -> str:
        candidate = str(value or 'none').strip().lower()
        if candidate in {'name', 'affiliation', 'purpose'}:
            return candidate
        return 'none'

    @staticmethod
    def _sanitize_ambiguity(value: object) -> str:
        candidate = str(value or 'high').strip().lower()
        if candidate in {'low', 'medium', 'high'}:
            return candidate
        return 'high'

    @staticmethod
    def _postprocess_normalized_slot_value(slot: str, value: object) -> str | None:
        normalized = str(value or '').strip()
        if not normalized:
            return None
        normalized = normalized.replace('\n', ' ').strip()
        normalized = normalized.rstrip('。．.、, ')
        if not normalized:
            return None
        if slot == 'name':
            normalized = normalized.removesuffix('様').strip()
            normalized = normalized.removesuffix('さん').strip()
            normalized = normalized.removesuffix('です').strip()
        return normalized or None

    def _stage1_language_hint(self) -> str:
        state = self._reducer.state
        if state.latest_applied_turn == 0 and state.response_language == 'ja':
            return 'unknown'
        return self._sanitize_detected_language(state.response_language)

    @staticmethod
    def _short(text: str, *, limit: int = 160) -> str:
        collapsed = ' '.join(str(text or '').split())
        if len(collapsed) <= limit:
            return collapsed
        return f'{collapsed[:limit - 1]}…'


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ReceptionOrchestratorNodeV2()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
