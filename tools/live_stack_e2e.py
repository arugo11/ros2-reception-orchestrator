#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
import uuid
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from asr_interfaces.msg import SpeechEvent
from asr_interfaces.msg import Utterance


class LiveStackProbe(Node):
    def __init__(self) -> None:
        super().__init__('reception_live_stack_probe')
        self.events: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.create_subscription(String, '/reception/events', self._on_event, 200)
        self.create_subscription(String, '/reception/session_state', self._on_state, 50)
        self.utterance_pub = self.create_publisher(Utterance, '/asr/utterances', 10)
        self.speech_event_pub = self.create_publisher(SpeechEvent, '/asr/speech_events', 10)

    def _on_event(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'raw': msg.data}
        payload['_ts'] = time.monotonic()
        self.events.append(payload)
        with self.condition:
            self.condition.notify_all()

    def _on_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'session': None, 'raw': msg.data}
        payload['_ts'] = time.monotonic()
        self.states.append(payload)
        with self.condition:
            self.condition.notify_all()

    def wait_for(self, description: str, predicate, timeout_sec: float) -> None:  # noqa: ANN001
        deadline = time.monotonic() + timeout_sec
        with self.condition:
            while True:
                if predicate():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(self._format_timeout(description))
                self.condition.wait(timeout=min(remaining, 0.5))

    def latest_session(self) -> dict[str, Any]:
        if not self.states:
            return {}
        latest = self.states[-1]
        session = latest.get('session')
        return session if isinstance(session, dict) else {}

    def latest_event_since(self, index: int, event_type: str) -> dict[str, Any] | None:
        for event in reversed(self.events[index:]):
            if event.get('event_type') == event_type:
                return event
        return None

    def inject_text(self, text: str) -> None:
        utterance_id = f'live-e2e-{uuid.uuid4().hex[:10]}'
        stamp = self.get_clock().now().to_msg()

        started = SpeechEvent()
        started.utterance_id = utterance_id
        started.stamp = stamp
        started.event_type = SpeechEvent.STARTED
        started.confidence = 1.0
        self.speech_event_pub.publish(started)

        utterance = Utterance()
        utterance.utterance_id = utterance_id
        utterance.started_at = stamp
        utterance.finalized_at = stamp
        utterance.text = text
        utterance.confidence = 1.0
        utterance.interrupted_tts = False
        self.utterance_pub.publish(utterance)

        ended = SpeechEvent()
        ended.utterance_id = utterance_id
        ended.stamp = stamp
        ended.event_type = SpeechEvent.ENDED
        ended.confidence = 1.0
        self.speech_event_pub.publish(ended)

    def _format_timeout(self, description: str) -> str:
        return json.dumps(
            {
                'description': description,
                'recent_events': self.events[-15:],
                'recent_states': self.states[-5:],
            },
            ensure_ascii=False,
            indent=2,
        )


def _run_scenario(scenario_path: Path, timeout_sec: float) -> dict[str, Any]:
    scenario = json.loads(scenario_path.read_text(encoding='utf-8'))
    turns = scenario.get('turns', [])
    if not turns:
        raise ValueError('scenario.turns must not be empty')

    rclpy.init()
    probe = LiveStackProbe()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(probe)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    report: dict[str, Any] = {
        'scenario': str(scenario_path),
        'turn_results': [],
    }

    try:
        probe.wait_for(
            'reception topics ready',
            lambda: probe.count_publishers('/reception/events') > 0 and probe.count_publishers('/reception/session_state') > 0,
            timeout_sec,
        )
        time.sleep(1.0)
        event_cursor = len(probe.events)

        for index, turn in enumerate(turns, start=1):
            text = str(turn['text'])
            expected_dialog_act = str(turn['expect_dialog_act'])
            probe.inject_text(text)
            probe.wait_for(
                f'tts_started for turn {index}',
                lambda cursor=event_cursor: any(
                    event.get('event_type') == 'tts_started' for event in probe.events[cursor:]
                ),
                timeout_sec,
            )
            tts_event = probe.latest_event_since(event_cursor, 'tts_started') or {}
            session = probe.latest_session()
            report['turn_results'].append(
                {
                    'turn': index,
                    'input_text': text,
                    'expected_dialog_act': expected_dialog_act,
                    'actual_dialog_act': tts_event.get('dialog_act'),
                    'spoken_text': tts_event.get('text'),
                    'session_phase': session.get('phase'),
                    'visitor_info': session.get('visitor_info'),
                }
            )
            event_cursor = len(probe.events)
            time.sleep(float(scenario.get('gap_sec', 2.0)))

        report['final_session'] = probe.latest_session()
        report['passed'] = True

        expected = scenario.get('expect', {})
        session = report['final_session']
        visitor = session.get('visitor_info', {}) if isinstance(session, dict) else {}
        if expected.get('phase') and session.get('phase') != expected['phase']:
            report['passed'] = False
            report['error'] = f"expected phase={expected['phase']}, got {session.get('phase')}"
        for field in ('name', 'affiliation', 'purpose'):
            if expected.get(field) is not None and visitor.get(field) != expected[field]:
                report['passed'] = False
                report['error'] = f"expected {field}={expected[field]}, got {visitor.get(field)}"
                break
        if expected.get('last_dialog_act'):
            last_turn = report['turn_results'][-1]
            if last_turn.get('actual_dialog_act') != expected['last_dialog_act']:
                report['passed'] = False
                report['error'] = (
                    f"expected last_dialog_act={expected['last_dialog_act']}, got {last_turn.get('actual_dialog_act')}"
                )
    finally:
        executor.shutdown()
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Run E2E scenarios against an already running reception stack.')
    parser.add_argument('scenarios', nargs='+', type=Path)
    parser.add_argument('--timeout-sec', type=float, default=90.0)
    parser.add_argument('--output', type=Path, default=Path('/tmp/reception_live_stack_e2e.json'))
    args = parser.parse_args()

    results = []
    passed = True
    for scenario_path in args.scenarios:
        result = _run_scenario(scenario_path, args.timeout_sec)
        results.append(result)
        passed = passed and bool(result.get('passed'))

    report = {
        'passed': passed,
        'results': results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())