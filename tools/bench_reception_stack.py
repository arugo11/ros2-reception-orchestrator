#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from asr_interfaces.msg import Utterance


@dataclass(slots=True)
class TurnResult:
    utterance_id: str
    utterance: str
    published_monotonic: float
    first_event_monotonic: float | None = None
    tts_started_monotonic: float | None = None
    turn_id: int | None = None


class BenchNode(Node):
    def __init__(self) -> None:
        super().__init__('reception_stack_bench')
        self.publisher = self.create_publisher(Utterance, '/asr/utterances', 10)
        self.subscription = self.create_subscription(String, '/reception/events', self._on_event, 50)
        self.state_subscription = self.create_subscription(
            String,
            '/reception/session_state',
            self._on_state,
            10,
        )
        self.results: list[TurnResult] = []
        self._event_log: list[dict[str, object]] = []
        self._state_log: list[dict[str, object]] = []
        self._latest_state: dict[str, Any] | None = None

    def publish_turn(self, text: str, utterance_id: str) -> None:
        msg = Utterance()
        msg.utterance_id = utterance_id
        msg.text = text
        msg.confidence = 0.99
        now = time.monotonic()
        self.results.append(
            TurnResult(
                utterance_id=utterance_id,
                utterance=text,
                published_monotonic=now,
            )
        )
        self.publisher.publish(msg)

    def _on_event(self, msg: String) -> None:
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'raw': msg.data}
        self._event_log.append({'monotonic': now, 'payload': payload})
        if not self.results:
            return
        event_type = payload.get('event_type')
        utterance_id = payload.get('utterance_id')
        if isinstance(utterance_id, str):
            matched = self._find_result_by_utterance_id(utterance_id)
            if matched is not None and matched.first_event_monotonic is None:
                matched.first_event_monotonic = now
            if event_type == 'llm_dispatched' and matched is not None:
                turn_id = payload.get('turn_id')
                if isinstance(turn_id, int):
                    matched.turn_id = turn_id
        if event_type == 'tts_started':
            turn_id = payload.get('turn_id')
            if isinstance(turn_id, int):
                matched = self._find_result_by_turn_id(turn_id)
                if matched is not None and matched.tts_started_monotonic is None:
                    matched.tts_started_monotonic = now

    def _on_state(self, msg: String) -> None:
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'raw': msg.data}
        self._latest_state = payload
        self._state_log.append({'monotonic': now, 'payload': payload})

    def _find_result_by_utterance_id(self, utterance_id: str) -> TurnResult | None:
        for result in self.results:
            if result.utterance_id == utterance_id:
                return result
        return None

    def _find_result_by_turn_id(self, turn_id: int) -> TurnResult | None:
        for result in self.results:
            if result.turn_id == turn_id:
                return result
        return None

    def summary(self) -> dict[str, object]:
        return {
            'turns': [
                {
                    **asdict(result),
                    'first_event_latency_ms': (
                        round((result.first_event_monotonic - result.published_monotonic) * 1000.0, 1)
                        if result.first_event_monotonic is not None
                        else None
                    ),
                    'tts_started_latency_ms': (
                        round((result.tts_started_monotonic - result.published_monotonic) * 1000.0, 1)
                        if result.tts_started_monotonic is not None
                        else None
                    ),
                }
                for result in self.results
            ],
            'events': self._event_log,
            'last_session_state': self._latest_state,
            'state_updates': self._state_log,
        }

    def saw_event(self, event_name: str) -> bool:
        for item in self._event_log:
            payload = item.get('payload')
            if isinstance(payload, dict) and payload.get('event_type') == event_name:
                return True
        return False


def _extract_session(summary: dict[str, object]) -> dict[str, Any] | None:
    latest = summary.get('last_session_state')
    if not isinstance(latest, dict):
        return None
    session = latest.get('session')
    return session if isinstance(session, dict) else None


def _lookup_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _evaluate_expectations(summary: dict[str, object], expectations: dict[str, Any]) -> dict[str, object]:
    session = _extract_session(summary) or {}
    turn_metrics = summary.get('turns', [])
    failures: list[str] = []

    expected_phase = expectations.get('phase')
    if expected_phase is not None and session.get('phase') != expected_phase:
        failures.append(f"expected phase={expected_phase}, got {session.get('phase')}")

    expected_slots = expectations.get('visitor_info')
    if isinstance(expected_slots, dict):
        visitor = session.get('visitor_info', {})
        for key, expected_value in expected_slots.items():
            actual_value = _lookup_nested({'visitor_info': visitor}, f'visitor_info.{key}')
            if actual_value != expected_value:
                failures.append(f"expected visitor_info.{key}={expected_value}, got {actual_value}")

    max_tts_started_latency_ms = expectations.get('max_tts_started_latency_ms')
    if isinstance(max_tts_started_latency_ms, (int, float)) and isinstance(turn_metrics, list):
        observed_tts_started = False
        for index, turn in enumerate(turn_metrics, start=1):
            if not isinstance(turn, dict):
                continue
            latency = turn.get('tts_started_latency_ms')
            if latency is None:
                continue
            observed_tts_started = True
            if latency > float(max_tts_started_latency_ms):
                failures.append(
                    f'turn {index} tts_started_latency_ms={latency} exceeds {max_tts_started_latency_ms}'
                )
        if not observed_tts_started:
            failures.append('missing any observed tts_started_latency_ms')

    required_events = expectations.get('required_events')
    if isinstance(required_events, list):
        event_names = {
            payload.get('event_type')
            for payload in (
                item.get('payload', {})
                for item in summary.get('events', [])
                if isinstance(item, dict)
            )
            if isinstance(payload, dict)
        }
        for event_name in required_events:
            if event_name not in event_names:
                failures.append(f'missing required event {event_name}')

    required_any_events = expectations.get('required_any_events')
    if isinstance(required_any_events, list):
        event_names = {
            payload.get('event_type')
            for payload in (
                item.get('payload', {})
                for item in summary.get('events', [])
                if isinstance(item, dict)
            )
            if isinstance(payload, dict)
        }
        for options in required_any_events:
            if not isinstance(options, list):
                continue
            if not any(event_name in event_names for event_name in options):
                failures.append(f'missing any required event from {options}')

    return {
        'passed': not failures,
        'failures': failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Replay scripted utterances into a running reception stack.')
    parser.add_argument('--scenario', required=True, help='JSON file containing {"turns":[...]}')
    parser.add_argument('--gap-sec', type=float, default=0.5, help='Gap between turns')
    parser.add_argument('--output', default='', help='Optional JSON output path')
    parser.add_argument('--sample-gpu', action='store_true', help='Capture nvidia-smi before and after run')
    parser.add_argument(
        '--wait-ready-timeout-sec',
        type=float,
        default=90.0,
        help='Wait for all_backends_ready before publishing turns',
    )
    args = parser.parse_args()

    scenario = json.loads(Path(args.scenario).read_text(encoding='utf-8'))
    turns = list(scenario.get('turns', []))
    gap_sec = float(scenario.get('gap_sec', args.gap_sec))
    settle_sec = float(scenario.get('settle_sec', 3.0))
    expectations = scenario.get('expect', {})

    gpu_before = None
    gpu_after = None
    if args.sample_gpu:
        gpu_before = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    rclpy.init()
    node = BenchNode()
    try:
        ready_deadline = time.monotonic() + args.wait_ready_timeout_sec
        while time.monotonic() < ready_deadline and not node.saw_event('all_backends_ready'):
            rclpy.spin_once(node, timeout_sec=0.1)
        # Let subscriptions and publishers settle after the ready event so the
        # first scripted utterance is not lost to startup races.
        settle_after_ready = time.monotonic() + 0.5
        while time.monotonic() < settle_after_ready:
            rclpy.spin_once(node, timeout_sec=0.05)

        for index, turn in enumerate(turns, start=1):
            node.publish_turn(str(turn), f'bench-{index}')
            deadline = time.monotonic() + gap_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        settle_deadline = time.monotonic() + settle_sec
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        result = node.summary()
        if isinstance(expectations, dict) and expectations:
            result['expectation_check'] = _evaluate_expectations(result, expectations)
        if args.sample_gpu:
            gpu_after = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            result['gpu_before'] = gpu_before
            result['gpu_after'] = gpu_after
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding='utf-8')
        else:
            print(text)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
