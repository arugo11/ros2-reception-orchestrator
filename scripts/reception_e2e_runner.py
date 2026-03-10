#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

import rclpy
from asr_interfaces.msg import SpeechEvent
from asr_interfaces.msg import Utterance
from asr_interfaces.srv import GetStatus as AsrGetStatus
from builtin_interfaces.msg import Time as RosTime
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from ros2_chat_interfaces.msg import ChatMessage
from ros2_chat_interfaces.srv import CreateThread
from ros2_chat_interfaces.srv import SendMessage
from std_msgs.msg import String


@dataclass(slots=True)
class ScenarioResult:
    name: str
    success: bool
    details: str


@dataclass(slots=True)
class ScenarioContext:
    name: str
    event_index: int
    sent_index: int
    thread_index: int


@dataclass(slots=True)
class OutboundMessage:
    thread_id: str
    text: str
    timestamp: float = field(default_factory=time.monotonic)


class ReceptionE2ERunner(Node):
    def __init__(self) -> None:
        super().__init__('reception_e2e_runner')

        self._events: list[dict[str, Any]] = []
        self._session_states: list[dict[str, Any]] = []
        self._sent_messages: list[OutboundMessage] = []
        self._created_threads: list[str] = []
        self._current_thread_id = 'thread-e2e-0'
        self._condition = threading.Condition()

        self._speech_pub = self.create_publisher(SpeechEvent, '/asr/speech_events', 10)
        self._utterance_pub = self.create_publisher(Utterance, '/asr/utterances', 10)
        self._incoming_pub = self.create_publisher(ChatMessage, '/chat_bridge/incoming', 10)

        self.create_service(AsrGetStatus, '/asr/get_status', self._on_asr_status)
        self.create_service(CreateThread, '/chat_bridge/create_thread', self._on_create_thread)
        self.create_service(SendMessage, '/chat_bridge/send_message', self._on_send_message)

        self.create_subscription(String, '/reception/events', self._on_event, 50)
        self.create_subscription(String, '/reception/session_state', self._on_session_state, 50)

    def _notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _on_event(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'event_type': 'invalid_json', 'raw': msg.data}
        payload['_ts'] = time.monotonic()
        self._events.append(payload)
        self.get_logger().info(f"event {payload.get('event_type')}: {json.dumps(payload, ensure_ascii=False)}")
        self._notify()

    def _on_session_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'session': None, 'raw': msg.data}
        payload['_ts'] = time.monotonic()
        self._session_states.append(payload)
        self._notify()

    def _on_asr_status(self, request: AsrGetStatus.Request, response: AsrGetStatus.Response) -> AsrGetStatus.Response:
        del request
        response.model_source = 'mock-continuous'
        response.revision = 'local'
        response.backend = 'mock'
        response.device = 'cpu'
        response.offline_mode = True
        response.frames_received = 0
        response.frames_dropped = 0
        response.rtf_ema = 0.0
        response.last_error = ''
        return response

    def _on_create_thread(self, request: CreateThread.Request, response: CreateThread.Response) -> CreateThread.Response:
        thread_id = f'thread-e2e-{len(self._created_threads) + 1}'
        self._current_thread_id = thread_id
        self._created_threads.append(thread_id)
        response.success = True
        response.adapter_name = request.parent_target.adapter_name
        response.thread_id = thread_id
        response.channel_id = request.parent_target.target_id
        response.message_id = f'msg-create-{len(self._created_threads)}'
        response.error_message = ''
        self.get_logger().info(f'create_thread thread_id={thread_id} title={request.thread_title!r}')
        self.get_logger().info('initial_text:\n' + request.initial_text)
        self._notify()
        return response

    def _on_send_message(self, request: SendMessage.Request, response: SendMessage.Response) -> SendMessage.Response:
        self._sent_messages.append(OutboundMessage(thread_id=request.target.target_id, text=request.text))
        response.success = True
        response.adapter_name = request.target.adapter_name
        response.thread_id = request.target.target_id
        response.channel_id = request.target.target_id
        response.message_id = f'msg-{len(self._sent_messages)}'
        response.error_message = ''
        self.get_logger().info('send_message:\n' + request.text)
        self._notify()
        return response

    def mark(self, name: str) -> ScenarioContext:
        return ScenarioContext(
            name=name,
            event_index=len(self._events),
            sent_index=len(self._sent_messages),
            thread_index=len(self._created_threads),
        )

    def wait_for(self, description: str, predicate: Callable[[], bool], timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                if predicate():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(self._format_timeout(description))
                self._condition.wait(timeout=min(remaining, 0.5))

    def _format_timeout(self, description: str) -> str:
        recent_events = self._events[-12:]
        recent_messages = [message.text for message in self._sent_messages[-6:]]
        session = self.latest_session()
        return (
            f'timeout waiting for {description}\n'
            f'latest_session={json.dumps(session, ensure_ascii=False, default=str)}\n'
            f'recent_events={json.dumps(recent_events, ensure_ascii=False, default=str)}\n'
            f'recent_sent_messages={json.dumps(recent_messages, ensure_ascii=False, default=str)}'
        )

    def latest_session(self) -> dict[str, Any]:
        if not self._session_states:
            return {'session': None}
        return self._session_states[-1]

    def latest_phase(self) -> str | None:
        session = self.latest_session().get('session')
        if not session:
            return None
        return session.get('phase')

    def session_is_idle(self) -> bool:
        return self.latest_session().get('session') is None

    def events_since(self, context: ScenarioContext, event_type: str | None = None) -> list[dict[str, Any]]:
        events = self._events[context.event_index :]
        if event_type is None:
            return events
        return [event for event in events if event.get('event_type') == event_type]

    def sent_since(self, context: ScenarioContext) -> list[OutboundMessage]:
        return self._sent_messages[context.sent_index :]

    def emit_utterance(self, text: str, *, interrupted_tts: bool = False) -> str:
        utterance_id = uuid.uuid4().hex[:8]
        self.get_logger().info(f'emit_utterance {utterance_id}: {text}')

        started = SpeechEvent()
        started.utterance_id = utterance_id
        started.stamp = self.get_clock().now().to_msg()
        started.event_type = SpeechEvent.STARTED
        started.confidence = 1.0
        self._speech_pub.publish(started)

        time.sleep(0.15)

        utterance = Utterance()
        utterance.utterance_id = utterance_id
        utterance.started_at = self.get_clock().now().to_msg()
        utterance.finalized_at = self.get_clock().now().to_msg()
        utterance.text = text
        utterance.confidence = 1.0
        utterance.interrupted_tts = interrupted_tts
        self._utterance_pub.publish(utterance)

        time.sleep(0.15)

        ended = SpeechEvent()
        ended.utterance_id = utterance_id
        ended.stamp = self.get_clock().now().to_msg()
        ended.event_type = SpeechEvent.ENDED
        ended.confidence = 1.0
        self._speech_pub.publish(ended)
        return utterance_id

    def inject_secretary_reply(self, text: str, *, message_id: str) -> None:
        msg = ChatMessage()
        msg.adapter_name = 'discord'
        msg.thread_id = self._current_thread_id
        msg.channel_id = self._current_thread_id
        msg.message_id = message_id
        msg.user_id = 'secretary-user'
        msg.user_name = 'secretary'
        msg.text = text
        self._incoming_pub.publish(msg)
        self.get_logger().info(f'inject_secretary_reply message_id={message_id} text={text!r}')


def _contains_confirmed(messages: list[OutboundMessage], needle: str | None = None) -> bool:
    for message in messages:
        if '【受付内容確定】' not in message.text:
            continue
        if needle is None or needle in message.text:
            return True
    return False


def _count_confirmed(messages: list[OutboundMessage]) -> int:
    return sum(1 for message in messages if '【受付内容確定】' in message.text)


def _contains_text(messages: list[OutboundMessage], needle: str) -> bool:
    return any(needle in message.text for message in messages)


def run_scenarios(runner: ReceptionE2ERunner) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []

    runner.get_logger().info('waiting for all_backends_ready')
    runner.wait_for(
        'all_backends_ready',
        lambda: any(event.get('event_type') == 'all_backends_ready' for event in runner._events),
        timeout_sec=420.0,
    )
    runner.wait_for('idle session before scenarios', runner.session_is_idle, timeout_sec=10.0)

    scenarios: list[tuple[str, Callable[[ReceptionE2ERunner, ScenarioContext], None]]] = [
        ('happy_path', _scenario_happy_path),
        ('missing_field_followup', _scenario_missing_field_followup),
        ('correction_during_confirmation', _scenario_correction),
        ('post_confirm_memo', _scenario_post_confirm_memo),
        ('duplicate_secretary_reply', _scenario_duplicate_reply),
    ]

    for name, scenario in scenarios:
        context = runner.mark(name)
        try:
            scenario(runner, context)
        except Exception as exc:  # noqa: BLE001
            results.append(ScenarioResult(name=name, success=False, details=str(exc)))
            break
        else:
            results.append(ScenarioResult(name=name, success=True, details='PASS'))
        runner.wait_for('session reset between scenarios', runner.session_is_idle, timeout_sec=20.0)
        time.sleep(0.5)

    return results


def _wait_for_tts(runner: ReceptionE2ERunner, context: ScenarioContext, dialog_act: str, timeout_sec: float = 120.0) -> None:
    runner.wait_for(
        f'tts_completed dialog_act={dialog_act}',
        lambda: any(
            event.get('event_type') == 'tts_completed' and event.get('dialog_act') == dialog_act
            for event in runner.events_since(context)
        ),
        timeout_sec=timeout_sec,
    )


def _scenario_happy_path(runner: ReceptionE2ERunner, context: ScenarioContext) -> None:
    runner.emit_utterance('OpenAIの田中です。山田さんに面会で来ました。')
    _wait_for_tts(runner, context, 'confirm')
    runner.emit_utterance('はい、間違いありません。')
    runner.wait_for(
        'confirmed outbound message',
        lambda: _contains_confirmed(runner.sent_since(context)),
        timeout_sec=120.0,
    )
    runner.inject_secretary_reply('山田は5分ほどで参ります。ロビーでお待ちください。', message_id='msg-secretary-happy')
    runner.wait_for(
        'secretary reply relay',
        lambda: any(event.get('event_type') == 'secretary_reply' for event in runner.events_since(context)),
        timeout_sec=120.0,
    )
    _wait_for_tts(runner, context, 'relay_secretary')


def _scenario_missing_field_followup(runner: ReceptionE2ERunner, context: ScenarioContext) -> None:
    runner.emit_utterance('OpenAIの田中です。')
    _wait_for_tts(runner, context, 'ask_purpose')
    runner.emit_utterance('山田さんに面会で来ました。')
    _wait_for_tts(runner, context, 'confirm')
    runner.emit_utterance('はい、間違いありません。')
    runner.wait_for(
        'confirmed outbound message after follow-up',
        lambda: _contains_confirmed(runner.sent_since(context)),
        timeout_sec=120.0,
    )


def _scenario_correction(runner: ReceptionE2ERunner, context: ScenarioContext) -> None:
    runner.emit_utterance('OpenAIの田中です。山田さんに面会で来ました。')
    _wait_for_tts(runner, context, 'confirm')
    runner.emit_utterance('違います。所属はDeepMindです。')
    runner.wait_for(
        'updated outbound message with DeepMind',
        lambda: _contains_text(runner.sent_since(context), 'DeepMind'),
        timeout_sec=120.0,
    )
    _wait_for_tts(runner, context, 'confirm')
    runner.emit_utterance('はい。')
    runner.wait_for(
        'confirmed outbound message with DeepMind',
        lambda: _contains_confirmed(runner.sent_since(context), 'DeepMind'),
        timeout_sec=120.0,
    )


def _scenario_post_confirm_memo(runner: ReceptionE2ERunner, context: ScenarioContext) -> None:
    runner.emit_utterance('OpenAIの田中です。山田さんに面会で来ました。')
    _wait_for_tts(runner, context, 'confirm')
    runner.emit_utterance('はい、間違いありません。')
    runner.wait_for(
        'initial confirmed message',
        lambda: _count_confirmed(runner.sent_since(context)) >= 1,
        timeout_sec=120.0,
    )
    before = len(runner.sent_since(context))
    runner.emit_utterance('よろしくお願いします。')
    _wait_for_tts(runner, context, 'acknowledge_waiting')
    after = len(runner.sent_since(context))
    if after != before:
        raise AssertionError(
            f'post-confirm memo emitted unexpected outbound message count before={before} after={after}'
        )


def _scenario_duplicate_reply(runner: ReceptionE2ERunner, context: ScenarioContext) -> None:
    runner.emit_utterance('OpenAIの田中です。山田さんに面会で来ました。')
    _wait_for_tts(runner, context, 'confirm')
    runner.emit_utterance('はい。')
    runner.wait_for(
        'confirmed message before duplicate reply',
        lambda: _contains_confirmed(runner.sent_since(context)),
        timeout_sec=120.0,
    )
    runner.inject_secretary_reply('重複確認用です。担当者がロビーに向かいます。', message_id='msg-dup-1')
    time.sleep(0.3)
    runner.inject_secretary_reply('重複確認用です。担当者がロビーに向かいます。', message_id='msg-dup-1')
    runner.wait_for(
        'one secretary_reply event',
        lambda: len(runner.events_since(context, 'secretary_reply')) >= 1,
        timeout_sec=120.0,
    )
    _wait_for_tts(runner, context, 'relay_secretary')
    reply_events = runner.events_since(context, 'secretary_reply')
    if len(reply_events) != 1:
        raise AssertionError(f'duplicate secretary reply was not deduplicated: count={len(reply_events)}')


def main() -> int:
    rclpy.init()
    runner = ReceptionE2ERunner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(runner)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        results = run_scenarios(runner)
    finally:
        executor.shutdown()
        runner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)

    failures = [result for result in results if not result.success]
    for result in results:
        status = 'PASS' if result.success else 'FAIL'
        print(f'{status}: {result.name}')
        if result.details != 'PASS':
            print(result.details)

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())