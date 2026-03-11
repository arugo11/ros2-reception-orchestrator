from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import queue
import re
import threading
import time
from typing import Any
from uuid import uuid4

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from asr_interfaces.msg import Utterance
from reception_interfaces.action import ExtractTurn
from reception_interfaces.action import RenderDialog
from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent
from reception_interfaces.msg import SessionStateV2
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

from .effect_executor_v2 import EffectExecutor
from .llm_stage_utils import extract_json_object
from .llm_stage_utils import wait_future
from .session_reducer_v2 import SessionReducer
from .turn_ingestor_v2 import TurnIngestor
from .v2_types import OrchestratorCommandData
from .v2_types import SecretaryReplyData
from .v2_types import SemanticDecisionData
from .v2_types import TurnEnvelopeData
from .v2_types import VisitorInfoData


@dataclass(slots=True)
class _QueueTurnEvent:
    turn: TurnEnvelopeData


@dataclass(slots=True)
class _QueueSecretaryEvent:
    reply: SecretaryReplyData


class ReceptionOrchestratorNodeV2(Node):
    _READY_MARKER = 'All backends ready: ASR, LLM, TTS, and chat bridge are available'
    _READY_ANNOUNCEMENT_TEXT = 'こんにちは, Welcome to SIT'
    _STAGE1_JSON_SCHEMA = json.dumps(
        {
            'type': 'object',
            'properties': {
                'speech_act': {
                    'type': 'string',
                    'enum': ['inform', 'affirm', 'deny', 'correction', 'question', 'complaint', 'greeting', 'unknown'],
                },
                'slot_updates': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': ['string', 'null']},
                        'affiliation': {'type': ['string', 'null']},
                        'purpose': {'type': ['string', 'null']},
                    },
                    'required': ['name', 'affiliation', 'purpose'],
                    'additionalProperties': False,
                },
                'correction_target': {
                    'type': 'string',
                    'enum': ['none', 'name', 'affiliation', 'purpose', 'all'],
                },
                'ignore_input': {'type': 'boolean'},
                'confidence': {'type': 'number'},
                'evidence': {'type': 'string'},
            },
            'required': ['speech_act', 'slot_updates', 'correction_target', 'ignore_input', 'confidence', 'evidence'],
            'additionalProperties': False,
        },
        ensure_ascii=False,
    )

    def __init__(self) -> None:
        super().__init__('reception_orchestrator')

        self._declare_parameters()
        self._load_parameters()

        self._extract_client = ActionClient(self, ExtractTurn, self._extract_action_name)
        self._render_client = ActionClient(self, RenderDialog, self._render_action_name)
        self._chat_client = ActionClient(self, Chat, self._llm_chat_action_name)
        self._tts_client = ActionClient(self, Speak, self._tts_action_name)

        self._create_thread_client = self.create_client(CreateThread, '/chat_bridge/create_thread')
        self._send_message_client = self.create_client(SendMessage, '/chat_bridge/send_message')

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

        self._event_queue: queue.Queue[object] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._seen_secretary_messages: set[str] = set()
        self._pending_turn_events: list[TurnEnvelopeData] = []
        self._state_lock = threading.RLock()
        self._ready_logged = False
        self._ready_announcement_sent = False
        self._llm_backend_ready = False
        self._llm_model_name = ''
        self._direct_vllm_client: VllmClient | None = None
        self._extract_action_backoff_until = 0.0
        self._render_action_backoff_until = 0.0

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

        self._tick_timer = self.create_timer(0.1, self._on_tick)
        self._bg = ThreadPoolExecutor(max_workers=4)

        self.get_logger().info('reception_orchestrator_v2 ready')
        self._publish_state()

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._effect_executor.shutdown()
        self._bg.shutdown(wait=False, cancel_futures=True)
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
        self.declare_parameter('render.action_name', '/reception/render_dialog')
        self.declare_parameter('tts.action_name', '/tts/speak')
        self.declare_parameter('session.state_topic', '/reception/session_state')
        self.declare_parameter('execution.event_topic', '/reception/events')
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
        self._render_action_name = str(self.get_parameter('render.action_name').value)
        self._tts_action_name = str(self.get_parameter('tts.action_name').value)
        self._session_state_topic = str(self.get_parameter('session.state_topic').value)
        self._execution_event_topic = str(self.get_parameter('execution.event_topic').value)

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

            self._publish_state()

    def _dependencies_ready(self) -> bool:
        return (
            self._llm_backend_ready
            and self._chat_client.server_is_ready()
            and self._tts_client.server_is_ready()
            and self._create_thread_client.service_is_ready()
            and self._send_message_client.service_is_ready()
        )

    def _on_utterance(self, msg: Utterance) -> None:
        self.get_logger().info(
            '[ASR] utterance received '
            f'id={msg.utterance_id} conf={float(msg.confidence):.2f} '
            f'interrupted_tts={bool(msg.interrupted_tts)} text={self._short(msg.text)}'
        )
        with self._state_lock:
            turns = self._ingestor.accept(
                utterance_id=msg.utterance_id,
                text=msg.text,
                confidence=float(msg.confidence),
                captured_during_tts=bool(msg.interrupted_tts),
                session_id=self._reducer.state.session_id,
            )
            for turn in turns:
                self._enqueue_or_buffer_turn(turn)

    def _enqueue_or_buffer_turn(self, turn: TurnEnvelopeData) -> None:
        if self._dependencies_ready():
            self._event_queue.put(_QueueTurnEvent(turn))
            return
        self._pending_turn_events.append(turn)
        self.get_logger().warn(
            f'[PIPELINE] buffered turn seq={turn.turn_seq} until dependencies are ready'
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

    def _process_turn(self, turn: TurnEnvelopeData) -> None:
        if not self._dependencies_ready():
            self._pending_turn_events.append(turn)
            self.get_logger().warn(
                f'[PIPELINE] re-buffered turn seq={turn.turn_seq} because backend became unavailable'
            )
            return

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
        self.get_logger().info(
            '[REDUCER] applied '
            f'seq={turn.turn_seq} phase={self._reducer.state.phase} '
            f'dialog_act={outcome.dialog_act} '
            f'slots=name:{self._reducer.state.visitor_info.name or "-"},'
            f'affiliation:{self._reducer.state.visitor_info.affiliation or "-"},'
            f'purpose:{self._reducer.state.visitor_info.purpose or "-"}'
        )
        self._publish_state()

        for command in outcome.commands:
            self._effect_executor.submit(command, immediate_non_tts=self._invoke_non_tts_command)

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
                {'text': text, 'dialog_act': outcome.dialog_act},
                ensure_ascii=False,
            ),
            dialog_act=outcome.dialog_act,
        )
        self._effect_executor.submit(tts_command, immediate_non_tts=self._invoke_non_tts_command)

    def _process_secretary_reply(self, reply: SecretaryReplyData) -> None:
        self.get_logger().info(
            f'[SECRETARY] incoming thread={reply.thread_id} text={self._short(reply.text)}'
        )
        outcome = self._reducer.handle_secretary_reply(reply)
        if outcome is None:
            self.get_logger().info('[SECRETARY] ignored (phase/thread mismatch or duplicate)')
            return
        self._publish_state()

        tts_command = OrchestratorCommandData(
            command_type=ExecutionCommand.COMMAND_TTS,
            command_id=uuid4().hex,
            session_id=outcome.session_id,
            turn_seq=outcome.turn_seq,
            payload_json=json.dumps(
                {'text': reply.text, 'dialog_act': 'relay_secretary'},
                ensure_ascii=False,
            ),
            dialog_act='relay_secretary',
        )
        self._effect_executor.submit(tts_command, immediate_non_tts=self._invoke_non_tts_command)

    def _submit_ready_announcement(self) -> None:
        self.get_logger().info(
            f'[TTS] ready announcement text={self._short(self._READY_ANNOUNCEMENT_TEXT)}'
        )
        tts_command = OrchestratorCommandData(
            command_type=ExecutionCommand.COMMAND_TTS,
            command_id=uuid4().hex,
            session_id=self._reducer.state.session_id,
            turn_seq=0,
            payload_json=json.dumps(
                {'text': self._READY_ANNOUNCEMENT_TEXT, 'dialog_act': 'system_ready'},
                ensure_ascii=False,
            ),
            dialog_act='system_ready',
        )
        self._effect_executor.submit(tts_command, immediate_non_tts=self._invoke_non_tts_command)

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
            decision = self._normalize_semantic_decision(turn.text, self._call_extract_direct_llm(turn))
            self._publish_execution_event(cmd, ExecutionEvent.STATUS_SUCCEEDED, ExecutionEvent.REASON_NONE, 'extract_succeeded')
            self.get_logger().info(
                '[LLM-S1] direct result '
                f'seq={turn.turn_seq} speech_act={decision.speech_act} '
                f'conf={decision.confidence:.2f} '
                f'slots=name:{decision.slot_patch.name or "-"},'
                f'affiliation:{decision.slot_patch.affiliation or "-"},'
                f'purpose:{decision.slot_patch.purpose or "-"}'
            )
            return decision
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[LLM-S1] direct failed seq={turn.turn_seq}: {exc}')
            try:
                decision = self._normalize_semantic_decision(
                    turn.text,
                    self._call_extract_stage_action(turn),
                )
                self._publish_execution_event(
                    cmd,
                    ExecutionEvent.STATUS_SUCCEEDED,
                    ExecutionEvent.REASON_NONE,
                    'extract_succeeded_via_stage_action',
                )
                self.get_logger().info(
                    '[LLM-S1] action fallback result '
                    f'seq={turn.turn_seq} speech_act={decision.speech_act} '
                    f'conf={decision.confidence:.2f} '
                    f'slots=name:{decision.slot_patch.name or "-"},'
                    f'affiliation:{decision.slot_patch.affiliation or "-"},'
                    f'purpose:{decision.slot_patch.purpose or "-"}'
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
                heuristic = self._normalize_semantic_decision(
                    turn.text,
                    self._heuristic_extract_decision(turn.turn_seq, turn.text),
                )
                self.get_logger().warn(
                    '[LLM-S1] heuristic fallback '
                    f'seq={turn.turn_seq} speech_act={heuristic.speech_act} '
                    f'slots=name:{heuristic.slot_patch.name or "-"},'
                    f'affiliation:{heuristic.slot_patch.affiliation or "-"},'
                    f'purpose:{heuristic.slot_patch.purpose or "-"}'
                )
                return heuristic

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
            self._reducer.state.visitor_info.name,
            self._reducer.state.visitor_info.affiliation,
            self._reducer.state.visitor_info.purpose,
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
        try:
            text = self._call_render_direct_llm(
                turn_seq=turn_seq,
                dialog_act=dialog_act,
                latest_user_text=latest_user_text,
                fallback=fallback,
            )
            self._publish_execution_event(cmd, ExecutionEvent.STATUS_SUCCEEDED, ExecutionEvent.REASON_NONE, 'render_succeeded')
            self.get_logger().info(
                f'[LLM-S2] direct result seq={turn_seq} text={self._short(text)}'
            )
            return text
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[LLM-S2] direct failed seq={turn_seq}: {exc}')
            try:
                text = self._call_render_stage_action(
                    turn_seq=turn_seq,
                    dialog_act=dialog_act,
                    latest_user_text=latest_user_text,
                    secretary_reply_text=secretary_reply_text,
                )
                self._publish_execution_event(
                    cmd,
                    ExecutionEvent.STATUS_SUCCEEDED,
                    ExecutionEvent.REASON_NONE,
                    'render_succeeded_via_stage_action',
                )
                self.get_logger().info(
                    f'[LLM-S2] action fallback result seq={turn_seq} text={self._short(text)}'
                )
                return text or fallback
            except Exception as action_exc:  # noqa: BLE001
                self._publish_execution_event(
                    cmd,
                    ExecutionEvent.STATUS_FAILED,
                    ExecutionEvent.REASON_INTERNAL_ERROR,
                    f'render_failed:{action_exc}',
                )
                self.get_logger().error(
                    f'[LLM-S2] action fallback failed seq={turn_seq}: {action_exc}'
                )
                return fallback

    def _invoke_non_tts_command(self, command: OrchestratorCommandData) -> tuple[bool, str]:
        try:
            payload = json.loads(command.payload_json or '{}')
        except json.JSONDecodeError:
            payload = {}

        if command.command_type == ExecutionCommand.COMMAND_DISCORD_CREATE:
            return self._invoke_discord_create(payload)
        if command.command_type == ExecutionCommand.COMMAND_DISCORD_SEND:
            return self._invoke_discord_send(payload)
        return True, 'noop'

    def _invoke_discord_create(self, payload: dict[str, Any]) -> tuple[bool, str]:
        title = str(payload.get('title', f'受付 {self._reducer.state.session_id[:8]}'))
        initial = str(payload.get('initial', ''))
        try:
            response = self._call_create_thread_service(thread_title=title, initial_text=initial)
            if not response.success:
                return False, response.error_message or 'create_thread failed'
            self._reducer.state.discord_thread_id = response.thread_id
            self._reducer.state.discord_channel_id = response.channel_id
            self._reducer.state.version += 1
            self._publish_state()
            self.get_logger().info(
                f'[DISCORD] thread created thread_id={response.thread_id} channel_id={response.channel_id}'
            )
            return True, f'thread_created:{response.thread_id}'
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[DISCORD] create failed: {exc}')
            return False, str(exc)

    def _invoke_discord_send(self, payload: dict[str, Any]) -> tuple[bool, str]:
        thread_id = str(payload.get('thread_id', '')).strip() or self._reducer.state.discord_thread_id
        text = str(payload.get('text', '')).strip()
        if not thread_id or not text:
            return False, 'thread_id/text missing'
        try:
            response = self._call_send_message_service(thread_id, text)
            self.get_logger().info(
                f'[DISCORD] message sent thread_id={thread_id} message_id={response.message_id}'
            )
            return True, f'message_sent:{response.message_id}'
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[DISCORD] send failed thread_id={thread_id}: {exc}')
            return False, str(exc)

    def _invoke_tts_command(self, command: OrchestratorCommandData) -> tuple[bool, str]:
        try:
            payload = json.loads(command.payload_json or '{}')
        except json.JSONDecodeError:
            payload = {}
        text = str(payload.get('text', '')).strip()
        if not text:
            return False, 'tts text missing'
        self.get_logger().info(
            f'[TTS] start seq={command.turn_seq} dialog_act={payload.get("dialog_act", command.dialog_act)} '
            f'text={self._short(text)}'
        )

        if not self._tts_client.wait_for_server(timeout_sec=5.0):
            return False, 'tts action unavailable'

        goal = Speak.Goal()
        goal.request_id = command.command_id
        goal.session_id = command.session_id
        goal.text = text
        goal.language = 'ja'
        goal.voice = ''
        goal.volume = 1.0
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

    def _on_command_completed(self, command: OrchestratorCommandData, ok: bool, detail: str) -> None:
        self.get_logger().info(
            f'[EFFECT] command completed id={command.command_id} type={command.command_type} '
            f'seq={command.turn_seq} ok={ok} detail={detail}'
        )
        if command.command_type == ExecutionCommand.COMMAND_TTS and ok:
            try:
                payload = json.loads(command.payload_json or '{}')
            except json.JSONDecodeError:
                payload = {}
            dialog_act = str(payload.get('dialog_act', command.dialog_act or ''))
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
            f'current_name={state.visitor_info.name}\n'
            f'current_affiliation={state.visitor_info.affiliation}\n'
            f'current_purpose={state.visitor_info.purpose}\n'
            f'pending_name={state.pending_confirmation.name}\n'
            f'pending_affiliation={state.pending_confirmation.affiliation}\n'
            f'pending_purpose={state.pending_confirmation.purpose}\n'
            f'latest_utterance={turn.text}\n'
        )
        system_prompt = (
            'You are a strict receptionist semantic extractor. '
            'Return JSON only. Do not output prose. '
            'Infer speech_act and slot_updates from the latest utterance. '
            'Never fabricate names, affiliations, or purposes if not present.'
        )
        raw = self._invoke_chat_action(
            session_id=f'{turn.session_id}:extract-direct:{turn.turn_seq}',
            user_message=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=140,
            timeout_sec=45.0,
            response_json_schema=self._STAGE1_JSON_SCHEMA,
        )
        payload = extract_json_object(raw)
        if not isinstance(payload, dict):
            raise RuntimeError('direct extract returned non-JSON response')

        updates = payload.get('slot_updates', {}) if isinstance(payload.get('slot_updates'), dict) else {}
        decision = SemanticDecisionData(
            turn_seq=turn.turn_seq,
            speech_act=str(payload.get('speech_act', 'unknown')),
            slot_patch=VisitorInfoData(
                name=str(updates.get('name') or '').strip(),
                affiliation=str(updates.get('affiliation') or '').strip(),
                purpose=str(updates.get('purpose') or '').strip(),
            ),
            correction_target=str(payload.get('correction_target', 'none')),
            ignore_input=bool(payload.get('ignore_input', False)),
            confidence=float(payload.get('confidence', 0.0)),
            evidence=str(payload.get('evidence', 'direct_llm')),
        )
        return decision

    def _call_render_direct_llm(
        self,
        *,
        turn_seq: int,
        dialog_act: str,
        latest_user_text: str,
        fallback: str,
    ) -> str:
        state = self._reducer.state
        prompt = (
            '大学受付として次に話す一文だけ返してください。説明やJSONは禁止。\n'
            f'dialog_act={dialog_act}\n'
            f'phase={state.phase}\n'
            f'name={state.visitor_info.name}\n'
            f'affiliation={state.visitor_info.affiliation}\n'
            f'purpose={state.visitor_info.purpose}\n'
            f'latest_user_text={latest_user_text}\n'
            f'fallback={fallback}\n'
        )
        text = self._invoke_chat_action(
            session_id=f'{state.session_id}:render-direct:{turn_seq}',
            user_message=prompt,
            system_prompt='あなたは丁寧な日本語の大学受付です。1文で簡潔に答えてください。',
            temperature=0.2,
            max_tokens=72,
            timeout_sec=30.0,
        ).strip()
        return text or fallback

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
        goal.visitor_info = self._to_visitor_msg(self._reducer.state.visitor_info)
        goal.pending_confirmation = self._to_visitor_msg(self._reducer.state.pending_confirmation)

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
            slot_patch=VisitorInfoData(
                name=str(decision.slot_patch.name or '').strip(),
                affiliation=str(decision.slot_patch.affiliation or '').strip(),
                purpose=str(decision.slot_patch.purpose or '').strip(),
            ),
            correction_target=str(decision.correction_target or 'none'),
            ignore_input=bool(decision.ignore_input),
            confidence=float(decision.confidence),
            evidence=str(decision.evidence or 'extract_stage_action'),
        )

    def _call_render_stage_action(
        self,
        *,
        turn_seq: int,
        dialog_act: str,
        latest_user_text: str,
        secretary_reply_text: str,
    ) -> str:
        if not self._render_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'{self._render_action_name} action unavailable')

        goal = RenderDialog.Goal()
        goal.session_id = self._reducer.state.session_id
        goal.turn_seq = int(turn_seq)
        goal.dialog_act = str(dialog_act)
        goal.phase = self._reducer.state.phase
        goal.visitor_info = self._to_visitor_msg(self._reducer.state.visitor_info)
        goal.pending_confirmation = self._to_visitor_msg(self._reducer.state.pending_confirmation)
        goal.latest_user_text = latest_user_text
        goal.secretary_reply_text = secretary_reply_text

        goal_handle = wait_future(self._render_client.send_goal_async(goal), timeout_sec=10.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f'{self._render_action_name} goal rejected')

        wrapped = wait_future(goal_handle.get_result_async(), timeout_sec=90.0)
        if wrapped is None:
            raise RuntimeError(f'{self._render_action_name} timeout waiting result')

        result = wrapped.result
        if result is None:
            raise RuntimeError(f'{self._render_action_name} returned no result')
        return str(result.text or '').strip()

    def _publish_state(self) -> None:
        state = self._reducer.state
        msg = SessionStateV2()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.session_id = state.session_id
        msg.phase = state.phase
        msg.visitor_info = self._to_visitor_msg(state.visitor_info)
        msg.pending_confirmation = self._to_visitor_msg(state.pending_confirmation)
        msg.latest_applied_turn = int(state.latest_applied_turn)
        msg.version = int(state.version)
        self._session_state_publisher.publish(msg)

    def _publish_execution_event(
        self,
        command: OrchestratorCommandData,
        status: int,
        reason_code: int,
        detail: str,
    ) -> None:
        msg = ExecutionEvent()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.command_id = command.command_id
        msg.command_type = int(command.command_type)
        msg.session_id = command.session_id
        msg.turn_seq = int(command.turn_seq)
        msg.status = int(status)
        msg.reason_code = int(reason_code)
        msg.detail = str(detail)
        self._event_publisher.publish(msg)

    @staticmethod
    def _to_visitor_msg(info: VisitorInfoData) -> VisitorInfo:
        msg = VisitorInfo()
        msg.name = info.name
        msg.affiliation = info.affiliation
        msg.purpose = info.purpose
        return msg

    def _to_turn_msg(self, turn: TurnEnvelopeData) -> TurnEnvelope:
        msg = TurnEnvelope()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.session_id = turn.session_id
        msg.turn_seq = int(turn.turn_seq)
        msg.utterance_id = turn.utterance_id
        msg.text = turn.text
        msg.captured_during_tts = bool(turn.captured_during_tts)
        msg.asr_confidence = float(turn.asr_confidence)
        return msg

    @staticmethod
    def _short(text: str, limit: int = 120) -> str:
        compact = ' '.join((text or '').split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + '...'

    @staticmethod
    def _heuristic_extract_decision(turn_seq: int, text: str) -> SemanticDecisionData:
        utterance = (text or '').strip()
        normalized = utterance.replace('　', ' ')
        lower = normalized.lower()
        segments = [s.strip() for s in re.split(r'[、。,.!?！？\n]+', normalized) if s.strip()]
        normalized_no_space = normalized.replace(' ', '')

        speech_act = 'inform'
        if not utterance:
            speech_act = 'unknown'
        elif normalized_no_space in ('はい', 'はい。', 'ええ', 'そうです', 'その通りです', 'お願いします'):
            speech_act = 'affirm'
        elif normalized_no_space in ('いいえ', '違います', '違う', 'いや'):
            speech_act = 'deny'
        elif any(token in lower for token in ('訂正', '違います', 'ではなく', 'じゃなく', '修正')):
            speech_act = 'correction'
        elif any(token in lower for token in ('こんにちは', 'こんばんは', 'おはよう', 'はじめまして')):
            speech_act = 'greeting'
        elif '?' in utterance or '？' in utterance:
            speech_act = 'question'

        affiliation_markers = (
            '株式会社',
            '有限会社',
            '合同会社',
            '大学',
            '学部',
            '学科',
            '研究室',
            '研究所',
            '病院',
            '銀行',
            '商事',
            'コーポレーション',
        )
        purpose_markers = (
            '打ち合わせ',
            '面談',
            '訪問',
            '会議',
            '相談',
            '商談',
            '納品',
            '手続き',
            '説明',
            '挨拶',
        )

        slot = VisitorInfoData()
        for pattern in (
            r'(?:私(?:の)?名前は|名前は|わたしは|私は)\s*([^\s、。,.]{1,20})\s*(?:です|と申します|といいます)?',
            r'([一-龥ぁ-んァ-ン]{2,8})\s*です',
        ):
            match = re.search(pattern, normalized)
            if not match:
                continue
            candidate = match.group(1).strip()
            if (
                candidate
                and 'の' not in candidate
                and not any(marker in candidate for marker in affiliation_markers + purpose_markers)
            ):
                slot.name = candidate
                break

        for pattern in (r'(?:所属(?:は)?|会社(?:名)?(?:は)?|勤務先(?:は)?|学校(?:名)?(?:は)?)\s*([^\n、。]{1,40})',):
            match = re.search(pattern, normalized)
            if match:
                slot.affiliation = match.group(1).strip()
                break
        if not slot.affiliation:
            for seg in segments:
                if any(marker in seg for marker in affiliation_markers):
                    slot.affiliation = seg
                    break

        for pattern in (r'(?:用件(?:は)?|目的(?:は)?|本日(?:の)?(?:用件|目的)(?:は)?)\s*([^\n、。]{1,40})',):
            match = re.search(pattern, normalized)
            if match:
                slot.purpose = match.group(1).strip()
                break
        if not slot.purpose:
            for seg in segments:
                if any(marker in seg for marker in purpose_markers):
                    slot.purpose = seg
                    break

        correction_target = 'none'
        if speech_act == 'correction':
            has_name = any(token in lower for token in ('名前', '氏名'))
            has_aff = any(token in lower for token in ('所属', '会社', '勤務先', '学校'))
            has_purpose = any(token in lower for token in ('用件', '目的', '要件'))
            count = int(has_name) + int(has_aff) + int(has_purpose)
            if count >= 2:
                correction_target = 'all'
            elif has_name:
                correction_target = 'name'
            elif has_aff:
                correction_target = 'affiliation'
            elif has_purpose:
                correction_target = 'purpose'
            else:
                correction_target = 'all'

        extracted = sum(1 for v in (slot.name, slot.affiliation, slot.purpose) if v)
        if extracted:
            confidence = 0.86
        elif speech_act == 'greeting':
            confidence = 0.80
        elif speech_act == 'question':
            confidence = 0.55
        else:
            confidence = 0.40

        return SemanticDecisionData(
            turn_seq=turn_seq,
            speech_act=speech_act,
            slot_patch=slot,
            correction_target=correction_target,
            ignore_input=False,
            confidence=confidence,
            evidence='orchestrator_heuristic_fallback',
        )

    @staticmethod
    def _normalize_semantic_decision(text: str, decision: SemanticDecisionData) -> SemanticDecisionData:
        utterance = (text or '').strip()
        normalized = utterance.replace('　', ' ')
        compact = normalized.replace(' ', '')
        slot = decision.slot_patch.copy()

        for field_name in ('name', 'affiliation', 'purpose'):
            value = getattr(slot, field_name).strip()
            if value in {'未指定', '未取得', '不明', 'unknown', 'なし', '無し', 'N/A'}:
                setattr(slot, field_name, '')

        if compact in ('はい', 'はい。', 'ええ', 'そうです', 'その通りです', 'お願いします'):
            return SemanticDecisionData(
                turn_seq=decision.turn_seq,
                speech_act='affirm',
                slot_patch=VisitorInfoData(),
                correction_target='none',
                ignore_input=False,
                confidence=max(float(decision.confidence), 0.95),
                evidence='normalized_affirm',
            )

        if compact in ('いいえ', '違います', '違う', 'いや'):
            return SemanticDecisionData(
                turn_seq=decision.turn_seq,
                speech_act='deny',
                slot_patch=VisitorInfoData(),
                correction_target='none',
                ignore_input=False,
                confidence=max(float(decision.confidence), 0.95),
                evidence='normalized_deny',
            )

        if decision.speech_act == 'greeting' and not any(
            (slot.name.strip(), slot.affiliation.strip(), slot.purpose.strip())
        ):
            return SemanticDecisionData(
                turn_seq=decision.turn_seq,
                speech_act='greeting',
                slot_patch=VisitorInfoData(),
                correction_target='none',
                ignore_input=False,
                confidence=max(float(decision.confidence), 0.9),
                evidence='normalized_greeting',
            )

        if any(token in normalized for token in ('に会いに来ました', 'に会いにきました', 'に用があって来ました', 'にお会いしたく', '面会')) and not any(
            token in normalized for token in ('私の名前', '名前は', '私は', 'わたしは')
        ):
            slot.name = ''
            slot.affiliation = ''
            slot.purpose = utterance
            return SemanticDecisionData(
                turn_seq=decision.turn_seq,
                speech_act='inform',
                slot_patch=slot,
                correction_target='none',
                ignore_input=False,
                confidence=max(float(decision.confidence), 0.95),
                evidence='normalized_purpose_first',
            )

        if re.fullmatch(r'[一-龥ぁ-んァ-ン]{2,12}です[。]?$', compact):
            candidate = compact.removesuffix('。').removesuffix('です')
            if any(marker in candidate for marker in ('研究室', '研究所', '大学', '学部', '学科', '株式会社', '合同会社', '有限会社')):
                slot.affiliation = candidate
                slot.name = ''
                return SemanticDecisionData(
                    turn_seq=decision.turn_seq,
                    speech_act='inform',
                    slot_patch=slot,
                    correction_target='none',
                    ignore_input=False,
                    confidence=max(float(decision.confidence), 0.9),
                    evidence='normalized_affiliation',
                )
            slot.name = candidate
            slot.affiliation = ''
            return SemanticDecisionData(
                turn_seq=decision.turn_seq,
                speech_act='inform',
                slot_patch=slot,
                correction_target='none',
                ignore_input=False,
                confidence=max(float(decision.confidence), 0.9),
                evidence='normalized_name',
            )

        forbidden_name_terms = ('学長', '教授', '先生', '部長', '社長')
        if slot.name and any(term in slot.name for term in forbidden_name_terms) and not any(
            token in normalized for token in ('私の名前', '名前は', '私は', 'わたしは')
        ):
            slot.name = ''

        return SemanticDecisionData(
            turn_seq=decision.turn_seq,
            speech_act=decision.speech_act,
            slot_patch=slot,
            correction_target=decision.correction_target,
            ignore_input=decision.ignore_input,
            confidence=decision.confidence,
            evidence=decision.evidence,
        )


def _fallback_dialog_text(dialog_act: str, name: str, affiliation: str, purpose: str) -> str:
    if dialog_act == 'ask_name':
        return 'お名前を教えてください。'
    if dialog_act == 'ask_affiliation':
        honor = f'{name}様、' if name else ''
        return f'{honor}ご所属を教えてください。'
    if dialog_act == 'ask_purpose':
        return '本日のご用件を教えてください。'
    if dialog_act == 'confirm':
        n = name or '未取得'
        a = affiliation or '未取得'
        p = purpose or '未取得'
        return f'お名前は{n}様、ご所属は{a}、ご用件は{p}でお間違いないでしょうか。'
    if dialog_act == 'notify_waiting':
        return '担当者へ連絡しました。少々お待ちください。'
    if dialog_act == 'acknowledge_waiting':
        return '担当者へ連絡済みです。少々お待ちください。'
    if dialog_act == 'retry':
        return 'もう一度お願いいたします。'
    if dialog_act == 'relay_secretary':
        return '担当者から連絡が入りました。'
    return '少々お待ちください。'


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


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
