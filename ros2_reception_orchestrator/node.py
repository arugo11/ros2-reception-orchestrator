from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import queue
import threading
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String

from asr_interfaces.msg import SpeechEvent
from asr_interfaces.msg import Utterance
from asr_interfaces.srv import GetStatus as AsrGetStatus
from ros2_chat_interfaces.msg import ChatMessage
from ros2_chat_interfaces.msg import ChatTarget
from ros2_chat_interfaces.srv import CreateThread
from ros2_chat_interfaces.srv import SendMessage
from ros2_vllm_interfaces.action import Chat
from ros2_vllm_interfaces.msg import LlmStatus
from tts_msgs.action import Speak
from tts_msgs.srv import GetStatus as TtsGetStatus

from .session_manager import ReceptionOrchestratorCore
from .state_models import DialogAct
from .state_models import SupervisorDecision
from .state_models import ThreadCreationResult
from .supervisor_adapter import SupervisorAdapter


@dataclass(slots=True)
class _UtteranceEvent:
    utterance_id: str
    text: str
    confidence: float
    interrupted_tts: bool


@dataclass(slots=True)
class _SpeechEventEvent:
    utterance_id: str
    event_type: int
    confidence: float


@dataclass(slots=True)
class _SupervisorResultEvent:
    session_id: str
    turn_id: int
    utterance_id: str
    utterance_text: str
    decision: SupervisorDecision


@dataclass(slots=True)
class _ThreadCreatedEvent:
    result: ThreadCreationResult


@dataclass(slots=True)
class _SecretaryReplyEvent:
    thread_id: str
    message_id: str
    text: str


@dataclass(slots=True)
class _TtsCompletedEvent:
    session_id: str
    turn_id: int
    dialog_act: DialogAct
    success: bool
    error_message: str = ''


@dataclass(slots=True)
class _PendingSemanticTurn:
    utterance_id: str
    text: str
    confidence: float
    interrupted_tts: bool
    due_monotonic: float


@dataclass(slots=True)
class _PendingResponse:
    session_id: str
    turn_id: int
    dialog_act: DialogAct
    text: str


class ReceptionOrchestratorNode(Node):
    _DEPENDENCY_WARN_INTERVAL_SEC = 5.0
    _HEALTH_POLL_INTERVAL_SEC = 2.0

    def __init__(self) -> None:
        super().__init__('reception_orchestrator')

        self._declare_parameters()
        self._load_parameters()

        self._ros_group = ReentrantCallbackGroup()
        self._llm_client = ActionClient(
            self,
            Chat,
            self._llm_chat_action_name,
            callback_group=self._ros_group,
        )
        self._tts_client = ActionClient(self, Speak, '/tts/speak', callback_group=self._ros_group)
        self._asr_status_client = self.create_client(
            AsrGetStatus,
            '/asr/get_status',
            callback_group=self._ros_group,
        )
        self._tts_status_client = self.create_client(
            TtsGetStatus,
            '/tts/get_status',
            callback_group=self._ros_group,
        )
        self._send_message_client = self.create_client(
            SendMessage,
            '/chat_bridge/send_message',
            callback_group=self._ros_group,
        )
        self._create_thread_client = self.create_client(
            CreateThread,
            '/chat_bridge/create_thread',
            callback_group=self._ros_group,
        )

        self._incoming_subscription = self.create_subscription(
            ChatMessage,
            '/chat_bridge/incoming',
            self._on_chat_incoming,
            10,
            callback_group=self._ros_group,
        )
        llm_status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._llm_status_subscription = self.create_subscription(
            LlmStatus,
            self._llm_status_topic,
            self._on_llm_status,
            llm_status_qos,
            callback_group=self._ros_group,
        )
        self._speech_event_subscription = self.create_subscription(
            SpeechEvent,
            '/asr/speech_events',
            self._on_speech_event,
            10,
            callback_group=self._ros_group,
        )
        self._utterance_subscription = self.create_subscription(
            Utterance,
            '/asr/utterances',
            self._on_utterance,
            10,
            callback_group=self._ros_group,
        )

        self._session_state_publisher = self.create_publisher(String, '/reception/session_state', 10)
        self._event_publisher = self.create_publisher(String, '/reception/events', 10)

        self._background = ThreadPoolExecutor(max_workers=8)
        self._event_queue: queue.Queue[object] = queue.Queue()
        self._control_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._dependency_warn_state: dict[str, float] = {}
        self._dependency_ready_state: dict[str, bool] = {}
        self._all_dependencies_ready_logged = False
        self._llm_backend_ready = False
        self._asr_status_ok = False
        self._tts_status_ok = False
        self._asr_status_inflight = False
        self._tts_status_inflight = False
        self._last_health_poll_monotonic = 0.0
        self._active_speech_starts: dict[str, float] = {}
        self._last_utterance_text = ''
        self._last_utterance_monotonic = 0.0
        self._pending_semantic_turn: _PendingSemanticTurn | None = None
        self._pending_response_queue: list[_PendingResponse] = []

        self._tts_state_lock = threading.RLock()
        self._tts_busy = False
        self._current_tts_goal_handle: Any | None = None
        self._current_tts_session_id = ''
        self._current_tts_turn_id = 0
        self._current_tts_dialog_act: DialogAct | None = None
        self._tts_cancel_requested = False

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._tick_timer = self.create_timer(0.1, self._on_tick, callback_group=self._ros_group)

        self._supervisor_adapter = SupervisorAdapter(
            self._invoke_llm_chat_action,
            temperature=self._dialog_temperature,
            max_tokens=self._dialog_max_tokens,
            trace=self._trace_pipeline,
        )
        self._core = ReceptionOrchestratorCore(
            inactivity_reset_sec=self._session_inactivity_reset_sec,
            trace=self._trace_pipeline,
        )

        self.get_logger().info('reception_orchestrator ready')
        self._report_dependency_state()

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._background.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        self.declare_parameter('discord.adapter_name', 'discord')
        self.declare_parameter('discord.parent_channel_id', '')
        self.declare_parameter('session.inactivity_reset_sec', 60)
        self.declare_parameter('response.followup_merge_window_ms', 1200)
        self.declare_parameter('llm.chat_action_name', '/llm/chat')
        self.declare_parameter('llm.status_topic', '/llm/status')
        self.declare_parameter('llm.temperature', 0.1)
        self.declare_parameter('llm.max_tokens', 96)

    def _load_parameters(self) -> None:
        self._discord_adapter_name = str(self.get_parameter('discord.adapter_name').value)
        self._discord_parent_channel_id = str(self.get_parameter('discord.parent_channel_id').value).strip()
        if not self._discord_parent_channel_id:
            raise ValueError('discord.parent_channel_id is required')
        self._session_inactivity_reset_sec = int(
            self.get_parameter('session.inactivity_reset_sec').value
        )
        self._followup_merge_window_sec = (
            float(self.get_parameter('response.followup_merge_window_ms').value) / 1000.0
        )
        self._llm_chat_action_name = str(
            self.get_parameter('llm.chat_action_name').value
        )
        self._llm_status_topic = str(self.get_parameter('llm.status_topic').value)
        self._dialog_temperature = float(self.get_parameter('llm.temperature').value)
        self._dialog_max_tokens = int(self.get_parameter('llm.max_tokens').value)

    def _on_tick(self) -> None:
        if not self._control_lock.acquire(blocking=False):
            return
        try:
            self._report_dependency_state()
            self._poll_health_services()
            self._flush_pending_semantic_turn_if_due()
            self._core.handle_inactivity(now=_utcnow())
            self._publish_session_state()
        finally:
            self._control_lock.release()

    def _report_dependency_state(self) -> None:
        now = time.monotonic()
        asr_ready = (
            self.count_publishers('/asr/utterances') > 0
            and self.count_publishers('/asr/speech_events') > 0
            and self._asr_status_ok
        )
        self._report_dependency(
            key='asr',
            ready=asr_ready,
            ready_message='ASR utterance streams are ready',
            waiting_message='Waiting for ASR topics /asr/utterances and /asr/speech_events and healthy /asr/get_status.',
            now=now,
        )
        self._report_dependency(
            key='llm',
            ready=self._llm_client.server_is_ready() and self._llm_backend_ready,
            ready_message=f'LLM action server is ready: {self._llm_chat_action_name}',
            waiting_message=(
                f'Waiting for LLM action server {self._llm_chat_action_name} '
                f'and backend READY state on {self._llm_status_topic}.'
            ),
            now=now,
        )
        self._report_dependency(
            key='tts',
            ready=self._tts_client.server_is_ready() and self._tts_status_ok,
            ready_message='TTS action server is ready: /tts/speak',
            waiting_message='Waiting for TTS action server /tts/speak and healthy /tts/get_status.',
            now=now,
        )
        self._report_dependency(
            key='chat_bridge',
            ready=self._send_message_client.service_is_ready() and self._create_thread_client.service_is_ready(),
            ready_message='Chat bridge services are ready',
            waiting_message='Waiting for chat bridge services /chat_bridge/send_message and /chat_bridge/create_thread.',
            now=now,
        )
        all_ready = all(
            self._dependency_ready_state.get(key, False)
            for key in ('asr', 'llm', 'tts', 'chat_bridge')
        )
        if all_ready and not self._all_dependencies_ready_logged:
            self._all_dependencies_ready_logged = True
            self.get_logger().info(
                'All backends ready: ASR, LLM, TTS, and chat bridge are available'
            )
            self._publish_event('all_backends_ready')
        if not all_ready:
            self._all_dependencies_ready_logged = False

    def _report_dependency(
        self,
        *,
        key: str,
        ready: bool,
        ready_message: str,
        waiting_message: str,
        now: float,
    ) -> None:
        previous_ready = self._dependency_ready_state.get(key)
        self._dependency_ready_state[key] = ready
        if ready:
            if previous_ready is False or previous_ready is None:
                self.get_logger().info(ready_message)
            return
        last_warn = self._dependency_warn_state.get(key, 0.0)
        if previous_ready is None or now - last_warn >= self._DEPENDENCY_WARN_INTERVAL_SEC:
            self.get_logger().warn(waiting_message)
            self._dependency_warn_state[key] = now

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                event = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                with self._control_lock:
                    self._process_event(event)
                    self._publish_session_state()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'Failed to process orchestrator event: {exc}')
                self._publish_event('worker_error', error=str(exc))

    def _process_event(self, event: object) -> None:
        if isinstance(event, _UtteranceEvent):
            self._handle_utterance_event(event)
            return
        if isinstance(event, _SpeechEventEvent):
            self._handle_speech_event_record(event)
            return
        if isinstance(event, _SupervisorResultEvent):
            self._handle_supervisor_result_event(event)
            return
        if isinstance(event, _ThreadCreatedEvent):
            self._handle_thread_created_event(event)
            return
        if isinstance(event, _SecretaryReplyEvent):
            self._handle_secretary_reply_event(event)
            return
        if isinstance(event, _TtsCompletedEvent):
            self._handle_tts_completed_event(event)
            return

    def _handle_utterance_event(self, event: _UtteranceEvent) -> None:
        now = time.monotonic()
        pending = self._pending_semantic_turn
        if pending is None:
            self._pending_semantic_turn = _PendingSemanticTurn(
                utterance_id=event.utterance_id,
                text=event.text,
                confidence=event.confidence,
                interrupted_tts=event.interrupted_tts,
                due_monotonic=now + self._followup_merge_window_sec,
            )
            return

        if now <= pending.due_monotonic:
            merged = ' '.join(part for part in (pending.text.strip(), event.text.strip()) if part).strip()
            self._pending_semantic_turn = _PendingSemanticTurn(
                utterance_id=event.utterance_id,
                text=merged,
                confidence=max(pending.confidence, event.confidence),
                interrupted_tts=(pending.interrupted_tts or event.interrupted_tts),
                due_monotonic=now + self._followup_merge_window_sec,
            )
            self._publish_event(
                'semantic_turn_merged',
                utterance_id=event.utterance_id,
                text=merged,
            )
            return

        self._flush_pending_semantic_turn()
        self._pending_semantic_turn = _PendingSemanticTurn(
            utterance_id=event.utterance_id,
            text=event.text,
            confidence=event.confidence,
            interrupted_tts=event.interrupted_tts,
            due_monotonic=now + self._followup_merge_window_sec,
        )

    def _handle_speech_event_record(self, event: _SpeechEventEvent) -> None:
        if event.event_type == SpeechEvent.STARTED:
            self._active_speech_starts[event.utterance_id] = time.monotonic()
        else:
            self._active_speech_starts.pop(event.utterance_id, None)
        self._publish_event(
            'speech_event',
            utterance_id=event.utterance_id,
            event_type=event.event_type,
            confidence=event.confidence,
        )

    def _handle_supervisor_result_event(self, event: _SupervisorResultEvent) -> None:
        outcome = self._core.reduce_supervisor_turn(
            session_id=event.session_id,
            turn_id=event.turn_id,
            utterance_text=event.utterance_text,
            decision=event.decision,
            now=_utcnow(),
        )
        if outcome is None:
            self._publish_event('stale_supervisor', session_id=event.session_id, turn_id=event.turn_id)
            return
        if outcome.create_thread:
            self._schedule_create_thread(outcome.session_id, outcome.initial_thread_text)
        if outcome.discord_text and self._core.session is not None and self._core.session.discord.thread_id:
            self._send_thread_message(self._core.session.discord.thread_id, outcome.discord_text)
        if outcome.dialog_act is not None and outcome.spoken_response:
            self._enqueue_or_speak_response(
                session_id=outcome.session_id,
                turn_id=outcome.turn_id,
                dialog_act=outcome.dialog_act,
                text=outcome.spoken_response,
            )

    def _handle_thread_created_event(self, event: _ThreadCreatedEvent) -> None:
        text = self._core.handle_thread_created(event.result)
        if self._core.session is None or not text or not self._core.session.discord.thread_id:
            return
        self._publish_event(
            'thread_created',
            session_id=event.result.session_id,
            thread_id=event.result.thread_id,
            channel_id=event.result.channel_id,
        )
        self._send_thread_message(self._core.session.discord.thread_id, text)

    def _handle_secretary_reply_event(self, event: _SecretaryReplyEvent) -> None:
        outcome = self._core.handle_secretary_reply(
            thread_id=event.thread_id,
            message_id=event.message_id,
            text=event.text,
            now=_utcnow(),
        )
        if outcome is None:
            return
        self._publish_event(
            'secretary_reply',
            thread_id=event.thread_id,
            message_id=event.message_id,
            text=event.text,
        )
        if outcome.dialog_act is not None and outcome.spoken_response:
            self._enqueue_or_speak_response(
                session_id=outcome.session_id,
                turn_id=outcome.turn_id,
                dialog_act=outcome.dialog_act,
                text=outcome.spoken_response,
            )

    def _handle_tts_completed_event(self, event: _TtsCompletedEvent) -> None:
        self._core.handle_tts_completed(
            session_id=event.session_id,
            turn_id=event.turn_id,
            dialog_act=event.dialog_act,
            now=_utcnow(),
        )
        self._publish_event(
            'tts_completed',
            session_id=event.session_id,
            turn_id=event.turn_id,
            dialog_act=event.dialog_act,
            success=event.success,
            error_message=event.error_message,
        )
        self._maybe_dispatch_buffered_response()

    def _on_utterance(self, msg: Utterance) -> None:
        text = msg.text.strip()
        if not text:
            return
        now = time.monotonic()
        if text == self._last_utterance_text and now - self._last_utterance_monotonic < 1.0:
            self.get_logger().info(f'ASR duplicate transcript ignored: {text}')
            return
        self._last_utterance_text = text
        self._last_utterance_monotonic = now
        self.get_logger().info(f'ASR final transcript: {text}')
        self._event_queue.put(
            _UtteranceEvent(
                utterance_id=msg.utterance_id,
                text=text,
                confidence=float(msg.confidence),
                interrupted_tts=bool(msg.interrupted_tts),
            )
        )

    def _on_speech_event(self, msg: SpeechEvent) -> None:
        self._event_queue.put(
            _SpeechEventEvent(
                utterance_id=msg.utterance_id,
                event_type=int(msg.event_type),
                confidence=float(msg.confidence),
            )
        )

    def _on_chat_incoming(self, msg: ChatMessage) -> None:
        text = msg.text.strip()
        if not text:
            return
        self._event_queue.put(
            _SecretaryReplyEvent(
                thread_id=msg.thread_id,
                message_id=msg.message_id,
                text=text,
            )
        )

    def _on_llm_status(self, msg: LlmStatus) -> None:
        self._llm_backend_ready = msg.status == LlmStatus.READY

    def _schedule_supervisor(
        self,
        session_id: str,
        turn_id: int,
        utterance_id: str,
        snapshot,
        utterance_text: str,
        *,
        captured_during_tts: bool,
    ) -> None:
        def _run() -> None:
            decision = self._supervisor_adapter.analyze(
                snapshot,
                utterance_text,
                currently_speaking=self._is_tts_busy(),
                captured_during_tts=captured_during_tts,
            )
            self._event_queue.put(
                _SupervisorResultEvent(
                    session_id=session_id,
                    turn_id=turn_id,
                    utterance_id=utterance_id,
                    utterance_text=utterance_text,
                    decision=decision,
                )
            )

        self._background.submit(_run)

    def _schedule_create_thread(self, session_id: str, initial_text: str) -> None:
        def _run() -> None:
            result = ThreadCreationResult(session_id=session_id, success=False)
            try:
                response = self._call_create_thread_service(
                    thread_title=f'受付 {session_id[:8]}',
                    initial_text=initial_text,
                )
                result.success = response.success
                result.thread_id = response.thread_id
                result.channel_id = response.channel_id
                result.error_message = response.error_message
            except Exception as exc:  # noqa: BLE001
                result.error_message = str(exc)
            self._event_queue.put(_ThreadCreatedEvent(result))

        self._background.submit(_run)

    def _send_thread_message(self, thread_id: str, text: str) -> None:
        def _run() -> None:
            try:
                response = self._call_send_message_service(thread_id, text)
                self.get_logger().info(
                    f'Discord thread message sent: thread_id={thread_id} message_id={response.message_id}'
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'Discord send failed: {exc}')

        self._background.submit(_run)

    def _invoke_llm_chat_action(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stateless: bool,
        response_json_schema: str | None = None,
    ) -> str:
        return self._invoke_chat_action(
            self._llm_client,
            self._llm_chat_action_name,
            session_id,
            user_message,
            system_prompt,
            temperature,
            max_tokens,
            stateless,
            response_json_schema,
        )

    def _invoke_chat_action(
        self,
        client: ActionClient,
        action_name: str,
        session_id: str,
        user_message: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stateless: bool,
        response_json_schema: str | None,
    ) -> str:
        deadline = time.monotonic() + 30.0
        while True:
            if not client.wait_for_server(timeout_sec=5.0):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{action_name} action server is unavailable')
                time.sleep(0.2)
                continue

            goal = Chat.Goal()
            goal.session_id = session_id
            goal.user_message = user_message
            goal.system_prompt = system_prompt
            goal.temperature = float(temperature)
            goal.max_tokens = int(max_tokens)
            goal.stateless = bool(stateless)
            if hasattr(goal, 'response_json_schema'):
                goal.response_json_schema = response_json_schema or ''

            self.get_logger().info(
                f'LLM request: session={session_id} max_tokens={goal.max_tokens} '
                f'stateless={goal.stateless} text={self._summarize_text(user_message)}'
            )
            goal_handle = self._wait_for_future(client.send_goal_async(goal), timeout_sec=10.0)
            if goal_handle is None or not goal_handle.accepted:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{action_name} goal rejected')
                time.sleep(0.2)
                continue

            wrapped = self._wait_for_future(goal_handle.get_result_async(), timeout_sec=90.0)
            result = wrapped.result
            if not result.success:
                message = result.error_message or 'llm chat failed'
                if 'not ready' in message.lower() and time.monotonic() < deadline:
                    time.sleep(0.2)
                    continue
                raise RuntimeError(message)
            self.get_logger().info(
                f'LLM response: session={session_id} prompt_tokens={result.prompt_tokens} '
                f'completion_tokens={result.completion_tokens} text={self._summarize_text(result.assistant_message)}'
            )
            return result.assistant_message

    def _speak_text_async(
        self,
        *,
        session_id: str,
        turn_id: int,
        dialog_act: DialogAct,
        text: str,
    ) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.get_logger().info(f'TTS speak: {cleaned}')
        self._publish_event(
            'tts_started',
            session_id=session_id,
            turn_id=turn_id,
            dialog_act=dialog_act,
            text=cleaned,
        )

        def _run() -> None:
            if not self._tts_client.wait_for_server(timeout_sec=5.0):
                self._event_queue.put(
                    _TtsCompletedEvent(
                        session_id=session_id,
                        turn_id=turn_id,
                        dialog_act=dialog_act,
                        success=False,
                        error_message='/tts/speak action server is unavailable',
                    )
                )
                return

            goal = Speak.Goal()
            goal.request_id = ''
            goal.session_id = session_id
            goal.text = cleaned
            goal.language = 'ja'
            goal.voice = ''
            goal.volume = 1.0
            goal.speed = 0.0
            goal.pitch = 0.0
            goal.priority = 0
            goal.interrupt = False
            goal.allow_streaming = False
            goal.save_wav = False

            with self._tts_state_lock:
                self._tts_busy = True
                self._current_tts_session_id = session_id
                self._current_tts_turn_id = turn_id
                self._current_tts_dialog_act = dialog_act
                self._tts_cancel_requested = False

            try:
                goal_handle = self._wait_for_future(self._tts_client.send_goal_async(goal), timeout_sec=30.0)
                if goal_handle is None or not goal_handle.accepted:
                    raise RuntimeError('/tts/speak goal rejected')
                with self._tts_state_lock:
                    self._current_tts_goal_handle = goal_handle
                wrapped = self._wait_for_future(goal_handle.get_result_async(), timeout_sec=180.0)
                ok = bool(wrapped.result.ok)
                error_message = '' if ok else (wrapped.result.error_message or 'tts failed')
                self._event_queue.put(
                    _TtsCompletedEvent(
                        session_id=session_id,
                        turn_id=turn_id,
                        dialog_act=dialog_act,
                        success=ok,
                        error_message=error_message,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._event_queue.put(
                    _TtsCompletedEvent(
                        session_id=session_id,
                        turn_id=turn_id,
                        dialog_act=dialog_act,
                        success=False,
                        error_message=str(exc),
                    )
                )
            finally:
                with self._tts_state_lock:
                    self._tts_busy = False
                    self._current_tts_goal_handle = None
                    self._current_tts_session_id = ''
                    self._current_tts_turn_id = 0
                    self._current_tts_dialog_act = None
                    self._tts_cancel_requested = False

        self._background.submit(_run)

    def _enqueue_or_speak_response(
        self,
        *,
        session_id: str,
        turn_id: int,
        dialog_act: DialogAct,
        text: str,
    ) -> None:
        if self._is_tts_busy():
            if self._pending_response_queue:
                self._pending_response_queue.clear()
                self._publish_event(
                    'pending_response_collapsed',
                    session_id=session_id,
                    turn_id=turn_id,
                    dialog_act=dialog_act,
                )
            self._pending_response_queue.append(
                _PendingResponse(
                    session_id=session_id,
                    turn_id=turn_id,
                    dialog_act=dialog_act,
                    text=text.strip(),
                )
            )
            self._publish_event(
                'llm_result_buffered',
                session_id=session_id,
                turn_id=turn_id,
                dialog_act=dialog_act,
                text=text.strip(),
            )
            return

        cleaned = self._core.accept_spoken_response(
            session_id=session_id,
            turn_id=turn_id,
            dialog_act=dialog_act,
            text=text,
            now=_utcnow(),
        )
        if not cleaned:
            self._publish_event(
                'stale_llm_response_drop',
                session_id=session_id,
                turn_id=turn_id,
                dialog_act=dialog_act,
            )
            return
        self._speak_text_async(
            session_id=session_id,
            turn_id=turn_id,
            dialog_act=dialog_act,
            text=cleaned,
        )

    def _maybe_dispatch_buffered_response(self) -> None:
        if self._is_tts_busy():
            return
        if not self._pending_response_queue:
            return
        pending = self._pending_response_queue.pop(-1)
        self._publish_event(
            'buffered_turn_dequeued',
            session_id=pending.session_id,
            turn_id=pending.turn_id,
            dialog_act=pending.dialog_act,
        )
        cleaned = self._core.accept_spoken_response(
            session_id=pending.session_id,
            turn_id=pending.turn_id,
            dialog_act=pending.dialog_act,
            text=pending.text,
            now=_utcnow(),
        )
        if not cleaned:
            self._publish_event(
                'stale_tts_drop',
                session_id=pending.session_id,
                turn_id=pending.turn_id,
                dialog_act=pending.dialog_act,
            )
            return
        self._speak_text_async(
            session_id=pending.session_id,
            turn_id=pending.turn_id,
            dialog_act=pending.dialog_act,
            text=cleaned,
        )

    def _is_tts_busy(self) -> bool:
        with self._tts_state_lock:
            return self._tts_busy

    def _flush_pending_semantic_turn_if_due(self) -> None:
        pending = self._pending_semantic_turn
        if pending is None or time.monotonic() < pending.due_monotonic:
            return
        self._flush_pending_semantic_turn()

    def _flush_pending_semantic_turn(self) -> None:
        pending = self._pending_semantic_turn
        if pending is None:
            return
        if not (
            self._llm_client.server_is_ready()
            and self._llm_backend_ready
        ):
            pending.due_monotonic = time.monotonic() + 0.5
            return
        self._pending_semantic_turn = None
        turn = self._core.begin_turn(
            utterance_id=pending.utterance_id,
            text=pending.text,
            now=_utcnow(),
        )
        if turn is None:
            return
        self._publish_event(
            'utterance_received',
            utterance_id=pending.utterance_id,
            text=pending.text,
            confidence=pending.confidence,
            interrupted_tts=pending.interrupted_tts,
            session_id=turn.session_id,
            turn_id=turn.turn_id,
        )
        if turn.create_thread:
            self._schedule_create_thread(turn.session_id, turn.initial_thread_text)
        self._publish_event(
            'llm_dispatched',
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            utterance_id=turn.utterance_id,
            captured_during_tts=pending.interrupted_tts,
        )
        self._schedule_supervisor(
            turn.session_id,
            turn.turn_id,
            turn.utterance_id,
            turn.snapshot,
            turn.user_text,
            captured_during_tts=pending.interrupted_tts,
        )

    def _poll_health_services(self) -> None:
        now = time.monotonic()
        if now - self._last_health_poll_monotonic < self._HEALTH_POLL_INTERVAL_SEC:
            return
        self._last_health_poll_monotonic = now
        if self._asr_status_client.service_is_ready() and not self._asr_status_inflight:
            self._asr_status_inflight = True
            future = self._asr_status_client.call_async(AsrGetStatus.Request())
            future.add_done_callback(self._on_asr_status_response)
        if self._tts_status_client.service_is_ready() and not self._tts_status_inflight:
            self._tts_status_inflight = True
            request = TtsGetStatus.Request()
            request.request_id = ''
            future = self._tts_status_client.call_async(request)
            future.add_done_callback(self._on_tts_status_response)

    def _on_asr_status_response(self, future: Future[Any]) -> None:
        self._asr_status_inflight = False
        try:
            response = future.result()
        except Exception:
            self._asr_status_ok = False
            return
        self._asr_status_ok = not bool(response.last_error)

    def _on_tts_status_response(self, future: Future[Any]) -> None:
        self._tts_status_inflight = False
        try:
            future.result()
        except Exception:
            self._tts_status_ok = False
            return
        self._tts_status_ok = True

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
        return self._wait_for_future(self._create_thread_client.call_async(request), timeout_sec=20.0)

    def _call_send_message_service(self, thread_id: str, text: str) -> SendMessage.Response:
        if not self._send_message_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/chat_bridge/send_message service is unavailable')
        request = SendMessage.Request()
        request.target.target_type = ChatTarget.THREAD
        request.target.adapter_name = self._discord_adapter_name
        request.target.target_id = thread_id
        request.text = text
        response = self._wait_for_future(self._send_message_client.call_async(request), timeout_sec=20.0)
        if not response.success:
            raise RuntimeError(response.error_message or 'send_message failed')
        return response

    @staticmethod
    def _wait_for_future(future: Future[Any], *, timeout_sec: float) -> Any:
        completed = threading.Event()
        holder: dict[str, Any] = {}

        def _done(done_future: Future[Any]) -> None:
            try:
                holder['result'] = done_future.result()
            except Exception as exc:  # noqa: BLE001
                holder['exception'] = exc
            finally:
                completed.set()

        future.add_done_callback(_done)
        if not completed.wait(timeout=timeout_sec):
            raise TimeoutError(f'future timed out after {timeout_sec} seconds')
        if 'exception' in holder:
            raise holder['exception']
        return holder.get('result')

    def _publish_session_state(self) -> None:
        msg = String()
        msg.data = self._core.debug_state_payload()
        self._session_state_publisher.publish(msg)

    def _publish_event(self, category: str, **payload: object) -> None:
        message = String()
        data = {'event_type': category, **payload}
        message.data = json.dumps(data, ensure_ascii=False, default=str)
        self._event_publisher.publish(message)

    def _trace_pipeline(self, message: str) -> None:
        self.get_logger().info(f'Pipeline: {message}')
        self._publish_event('pipeline', message=message)

    @staticmethod
    def _summarize_text(text: str, limit: int = 160) -> str:
        compact = ' '.join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + '...'


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ReceptionOrchestratorNode()
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
