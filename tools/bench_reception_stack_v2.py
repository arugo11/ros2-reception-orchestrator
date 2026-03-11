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

from asr_interfaces.msg import Utterance
from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent
from reception_interfaces.msg import SessionStateV2


@dataclass(slots=True)
class TurnResult:
    turn_seq: int
    utterance_id: str
    utterance: str
    published_monotonic: float
    first_event_monotonic: float | None = None
    tts_started_monotonic: float | None = None


class BenchNode(Node):
    def __init__(self) -> None:
        super().__init__('reception_stack_bench_v2')
        self.publisher = self.create_publisher(Utterance, '/asr/utterances', 10)
        self.event_sub = self.create_subscription(ExecutionEvent, '/reception/events', self._on_event, 100)
        self.state_sub = self.create_subscription(SessionStateV2, '/reception/session_state', self._on_state, 20)

        self.results: list[TurnResult] = []
        self._event_log: list[dict[str, object]] = []
        self._latest_state: dict[str, Any] | None = None
        self._state_log: list[dict[str, object]] = []

    def publish_turn(self, text: str, utterance_id: str, turn_seq: int) -> None:
        now = time.monotonic()
        msg = Utterance()
        msg.utterance_id = utterance_id
        msg.text = text
        msg.confidence = 0.99
        self.results.append(
            TurnResult(
                turn_seq=turn_seq,
                utterance_id=utterance_id,
                utterance=text,
                published_monotonic=now,
            )
        )
        self.publisher.publish(msg)

    def _on_event(self, msg: ExecutionEvent) -> None:
        now = time.monotonic()
        payload = {
            'event_type': self._map_event_type(msg),
            'command_id': msg.command_id,
            'command_type': int(msg.command_type),
            'turn_seq': int(msg.turn_seq),
            'status': int(msg.status),
            'reason_code': int(msg.reason_code),
            'detail': msg.detail,
        }
        self._event_log.append({'monotonic': now, 'payload': payload})

        matched = self._find_result_by_turn_seq(int(msg.turn_seq))
        if matched is not None and matched.first_event_monotonic is None:
            matched.first_event_monotonic = now

        if (
            int(msg.command_type) == ExecutionCommand.COMMAND_TTS
            and int(msg.status) == ExecutionEvent.STATUS_STARTED
            and matched is not None
            and matched.tts_started_monotonic is None
        ):
            matched.tts_started_monotonic = now

    def _on_state(self, msg: SessionStateV2) -> None:
        now = time.monotonic()
        payload = {
            'session': {
                'session_id': msg.session_id,
                'phase': msg.phase,
                'visitor_info': {
                    'name': msg.visitor_info.name or None,
                    'affiliation': msg.visitor_info.affiliation or None,
                    'purpose': msg.visitor_info.purpose or None,
                },
                'pending_confirmation': {
                    'name': msg.pending_confirmation.name or None,
                    'affiliation': msg.pending_confirmation.affiliation or None,
                    'purpose': msg.pending_confirmation.purpose or None,
                },
                'latest_applied_turn': int(msg.latest_applied_turn),
                'version': int(msg.version),
            }
        }
        self._latest_state = payload
        self._state_log.append({'monotonic': now, 'payload': payload})

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

    @staticmethod
    def _map_event_type(msg: ExecutionEvent) -> str:
        if int(msg.command_type) == ExecutionCommand.COMMAND_TTS and int(msg.status) == ExecutionEvent.STATUS_STARTED:
            return 'tts_started'
        if msg.command_id.startswith('extract-') and int(msg.status) == ExecutionEvent.STATUS_STARTED:
            return 'llm_dispatched'
        if int(msg.reason_code) == ExecutionEvent.REASON_REPLACED:
            return 'pending_response_collapsed'
        return 'pipeline'

    def _find_result_by_turn_seq(self, turn_seq: int) -> TurnResult | None:
        for result in self.results:
            if result.turn_seq == turn_seq:
                return result
        return None


def _lookup_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _evaluate_expectations(summary: dict[str, object], expectations: dict[str, Any], published_turn_count: int) -> dict[str, object]:
    latest = summary.get('last_session_state') or {}
    session = latest.get('session', {}) if isinstance(latest, dict) else {}
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
    event_names = {
        payload.get('event_type')
        for payload in (
            item.get('payload', {})
            for item in summary.get('events', [])
            if isinstance(item, dict)
        )
        if isinstance(payload, dict)
    }
    if isinstance(required_events, list):
        for event_name in required_events:
            if event_name not in event_names:
                failures.append(f'missing required event {event_name}')

    required_any_events = expectations.get('required_any_events')
    if isinstance(required_any_events, list):
        merged_happened = len(turn_metrics) < published_turn_count
        for options in required_any_events:
            if not isinstance(options, list):
                continue
            if not any(event_name in event_names for event_name in options):
                if merged_happened and 'semantic_turn_merged' in options:
                    continue
                failures.append(f'missing any required event from {options}')

    return {'passed': not failures, 'failures': failures}


def main() -> int:
    parser = argparse.ArgumentParser(description='Replay scripted utterances into v2 reception stack.')
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--gap-sec', type=float, default=0.5)
    parser.add_argument('--output', default='')
    parser.add_argument('--sample-gpu', action='store_true')
    parser.add_argument('--wait-ready-timeout-sec', type=float, default=90.0)
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
        start = time.monotonic()
        while time.monotonic() - start < args.wait_ready_timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.1)

        for index, turn in enumerate(turns, start=1):
            node.publish_turn(str(turn), f'bench-{index}', index)
            deadline = time.monotonic() + gap_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)

        settle_deadline = time.monotonic() + settle_sec
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        result = node.summary()
        if isinstance(expectations, dict) and expectations:
            result['expectation_check'] = _evaluate_expectations(result, expectations, len(turns))
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
