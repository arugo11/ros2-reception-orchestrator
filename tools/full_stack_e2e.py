#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any
import wave

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from asr_interfaces.msg import SpeechEvent
from asr_interfaces.msg import Utterance
from std_msgs.msg import String

from tts_msgs.action import Speak


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIO = (
    REPO_ROOT
    / 'src'
    / 'ros2_reception_orchestrator'
    / 'tools'
    / 'scenarios'
    / 'reception_full_stack_happy_path.json'
)
MODEL_CATALOG = REPO_ROOT / 'config' / 'model_profiles.yaml'


@dataclass(slots=True)
class ProcessHandle:
    name: str
    popen: subprocess.Popen[str]
    log_path: Path


class FullStackProbe(Node):
    def __init__(self) -> None:
        super().__init__('reception_full_stack_probe')
        self._events: list[dict[str, Any]] = []
        self._states: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self.create_subscription(String, '/reception/events', self._on_event, 50)
        self.create_subscription(String, '/reception/session_state', self._on_state, 10)

    def _on_event(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'raw': msg.data}
        payload['_ts'] = time.monotonic()
        self._events.append(payload)
        self.get_logger().info(f"event {payload.get('event_type')}: {json.dumps(payload, ensure_ascii=False)}")
        with self._condition:
            self._condition.notify_all()

    def _on_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'raw': msg.data}
        payload['_ts'] = time.monotonic()
        self._states.append(payload)
        session = payload.get('session')
        if isinstance(session, dict):
            self.get_logger().info(
                f"state phase={session.get('phase')} last_dialog_act={session.get('last_dialog_act')}"
            )
        with self._condition:
            self._condition.notify_all()

    def wait_for(self, description: str, predicate, timeout_sec: float) -> None:  # noqa: ANN001
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
        return (
            f'timeout waiting for {description}\n'
            f'recent_events={json.dumps(self._events[-10:], ensure_ascii=False)}\n'
            f'recent_states={json.dumps(self._states[-5:], ensure_ascii=False)}'
        )

    def latest_session(self) -> dict[str, Any]:
        if not self._states:
            return {}
        latest = self._states[-1]
        session = latest.get('session')
        return session if isinstance(session, dict) else {}

    def events_since(self, index: int) -> list[dict[str, Any]]:
        return self._events[index:]

    def latest_event_since(self, index: int, event_type: str) -> dict[str, Any] | None:
        for event in reversed(self._events[index:]):
            if event.get('event_type') == event_type:
                return event
        return None


class TtsSynthesisClient(Node):
    def __init__(self) -> None:
        super().__init__('reception_full_stack_tts_client')
        self._tts_client = ActionClient(self, Speak, '/tts/speak')

    def synthesize_wav(self, text: str, request_id: str, timeout_sec: float) -> Path:
        if not self._tts_client.wait_for_server(timeout_sec=timeout_sec):
            raise RuntimeError('/tts/speak action server is unavailable')

        goal = Speak.Goal()
        goal.request_id = request_id
        goal.session_id = 'full-stack-e2e'
        goal.text = text
        goal.language = 'ja'
        goal.voice = ''
        goal.volume = 1.0
        goal.speed = 0.0
        goal.pitch = 0.0
        goal.priority = 0
        goal.interrupt = False
        goal.allow_streaming = False
        goal.save_wav = True

        send_future = self._tts_client.send_goal_async(goal)
        self._spin_until_future(send_future, timeout_sec)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('TTS goal was rejected while generating test WAV')

        result_future = goal_handle.get_result_async()
        self._spin_until_future(result_future, timeout_sec)
        wrapped = result_future.result()
        if wrapped is None or not wrapped.result.ok:
            raise RuntimeError(wrapped.result.error_message or 'TTS synthesis failed')
        wav_uri = wrapped.result.wav_uri
        if not wav_uri.startswith('file://'):
            raise RuntimeError(f'TTS result did not include wav_uri: {wav_uri}')
        return Path(wav_uri.removeprefix('file://'))

    def _spin_until_future(self, future, timeout_sec: float) -> None:  # noqa: ANN001
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if future.done():
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError('timed out waiting for ROS future')


class AsrFallbackInjector(Node):
    def __init__(self) -> None:
        super().__init__('reception_full_stack_asr_fallback')
        self._utterance_pub = self.create_publisher(Utterance, '/asr/utterances', 10)
        self._speech_event_pub = self.create_publisher(SpeechEvent, '/asr/speech_events', 10)

    def inject(self, text: str) -> None:
        utterance_id = f'fallback-{int(time.time() * 1000)}'
        stamp = self.get_clock().now().to_msg()

        started = SpeechEvent()
        started.utterance_id = utterance_id
        started.stamp = stamp
        started.event_type = SpeechEvent.STARTED
        started.confidence = 1.0
        self._speech_event_pub.publish(started)

        utterance = Utterance()
        utterance.utterance_id = utterance_id
        utterance.started_at = stamp
        utterance.finalized_at = stamp
        utterance.text = text
        utterance.confidence = 1.0
        utterance.interrupted_tts = False
        self._utterance_pub.publish(utterance)

        ended = SpeechEvent()
        ended.utterance_id = utterance_id
        ended.stamp = stamp
        ended.event_type = SpeechEvent.ENDED
        ended.confidence = 1.0
        self._speech_event_pub.publish(ended)


def _env_prefix() -> str:
    return ' && '.join(
        [
            'source /opt/ros/jazzy/setup.bash',
            f'source {REPO_ROOT / "install" / "setup.bash"}',
            f'[ -f {REPO_ROOT / "src" / "ros2_asr" / "install" / "setup.bash"} ] && source {REPO_ROOT / "src" / "ros2_asr" / "install" / "setup.bash"} || true',
            f'[ -f {REPO_ROOT / "src" / "ros2_tts" / "install" / "setup.bash"} ] && source {REPO_ROOT / "src" / "ros2_tts" / "install" / "setup.bash"} || true',
            f'[ -f {REPO_ROOT / "src" / "ros2_chat" / "install" / "setup.bash"} ] && source {REPO_ROOT / "src" / "ros2_chat" / "install" / "setup.bash"} || true',
            f'[ -f {REPO_ROOT / "src" / "ros2_vllm" / "install" / "setup.bash"} ] && source {REPO_ROOT / "src" / "ros2_vllm" / "install" / "setup.bash"} || true',
        ]
    )


def _spawn(name: str, command: str, log_dir: Path, env: dict[str, str]) -> ProcessHandle:
    log_path = log_dir / f'{name}.log'
    handle = log_path.open('w', encoding='utf-8')
    popen = subprocess.Popen(
        ['bash', '-lc', f'{_env_prefix()} && {command}'],
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        preexec_fn=os.setsid,
    )
    return ProcessHandle(name=name, popen=popen, log_path=log_path)


def _terminate(handle: ProcessHandle | None, timeout_sec: float = 10.0) -> None:
    if handle is None:
        return
    proc = handle.popen
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)


def _launch_stack(
    *,
    ros_domain_id: int,
    stack_profile: str,
    log_dir: Path,
    reply_text: str,
) -> list[ProcessHandle]:
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(ros_domain_id)
    catalog = str(MODEL_CATALOG)
    handles = [
        _spawn(
            'mock_chat_bridge',
            (
                f'python3 {REPO_ROOT / "scripts" / "mock_chat_bridge.py"} '
                f'--reply-text {json.dumps(reply_text)}'
            ),
            log_dir,
            env,
        ),
        _spawn(
            'asr',
            (
                'ros2 launch asr_streaming_node asr_streaming_node.launch.py '
                f'model_catalog_file:={catalog} '
                f'profile_name:={stack_profile} '
                'runtime_device:=auto '
                'continuous_enabled:=true'
            ),
            log_dir,
            env,
        ),
        _spawn(
            'llm',
            (
                'ros2 launch ros2_vllm vllm_bringup.launch.py '
                f'model_catalog_file:={catalog} '
                f'profile_name:={stack_profile} '
                'wandb_enabled:=false '
                'reuse_existing_backend:=false '
                'replace_existing_backend:=true'
            ),
            log_dir,
            env,
        ),
        _spawn(
            'tts',
            (
                'ros2 launch tts_bringup tts.launch.py '
                f'model_catalog_file:={catalog} '
                f'profile_name:={stack_profile}'
            ),
            log_dir,
            env,
        ),
        _spawn(
            'orchestrator',
            (
                'ros2 run ros2_reception_orchestrator reception_orchestrator --ros-args '
                '-p discord.parent_channel_id:=mock-parent '
                '-p session.inactivity_reset_sec:=120'
            ),
            log_dir,
            env,
        ),
    ]
    return handles


def _play_wav(
    *,
    ros_domain_id: int,
    wav_path: Path,
    log_dir: Path,
    turn_index: int,
) -> ProcessHandle:
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(ros_domain_id)
    return _spawn(
        f'mic_turn_{turn_index}',
        (
            'ros2 run mic_input_node mic_input_node --ros-args '
            '-p audio_backend:=wav_file '
            f'-p wav_file_path:={wav_path}'
        ),
        log_dir,
        env,
    )


def _wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            return 0.0
        return float(wav_file.getnframes()) / float(frame_rate)


def _check_processes(handles: list[ProcessHandle]) -> None:
    for handle in handles:
        code = handle.popen.poll()
        if code is not None and code != 0:
            raise RuntimeError(f'process exited early: {handle.name} code={code} log={handle.log_path}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run real full-stack reception E2E.')
    parser.add_argument('--scenario', type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument('--stack-profile', default='qwen_fullstack')
    parser.add_argument('--ros-domain-id', type=int, default=132)
    parser.add_argument('--output', type=Path, default=Path('/tmp/reception_full_stack_e2e/report.json'))
    parser.add_argument('--startup-timeout-sec', type=float, default=900.0)
    parser.add_argument('--turn-timeout-sec', type=float, default=240.0)
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text(encoding='utf-8'))
    turns = list(scenario.get('turns', []))
    if not turns:
        raise ValueError('scenario.turns must not be empty')
    expectations = scenario.get('expect', {})
    reply_text = '担当者がまもなく参ります。ロビーでお待ちください。'

    log_dir = args.output.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    probe = FullStackProbe()
    tts_client = TtsSynthesisClient()
    asr_fallback = AsrFallbackInjector()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(probe)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    handles: list[ProcessHandle] = []
    report: dict[str, Any] = {
        'scenario': str(args.scenario),
        'stack_profile': args.stack_profile,
        'ros_domain_id': args.ros_domain_id,
        'turns': [],
        'logs': {},
    }
    try:
        handles = _launch_stack(
            ros_domain_id=args.ros_domain_id,
            stack_profile=args.stack_profile,
            log_dir=log_dir,
            reply_text=reply_text,
        )
        probe.wait_for(
            'all_backends_ready',
            lambda: any(event.get('event_type') == 'all_backends_ready' for event in probe.events_since(0)),
            args.startup_timeout_sec,
        )
        _check_processes(handles)

        event_cursor = len(probe._events)
        conversation_completed = False
        for index, turn in enumerate(turns, start=1):
            expected_dialog_act = str(turn['expect_dialog_act'])
            input_mode = str(turn.get('input_mode', 'audio')).strip() or 'audio'
            candidate_texts = [str(turn['text'])]
            for candidate in turn.get('alternatives', []):
                candidate_texts.append(str(candidate))
            used_text = candidate_texts[0]

            if input_mode == 'asr_fallback':
                injected_text = str(turn.get('fallback_text') or turn['text']).strip()
                asr_fallback.inject(injected_text)
                probe.wait_for(
                    f'tts_completed {expected_dialog_act} or relay_secretary after injected input',
                    lambda expected=expected_dialog_act, cursor=event_cursor: any(
                        event.get('event_type') == 'tts_completed'
                        and str(event.get('dialog_act', '')) in {expected, 'relay_secretary'}
                        for event in probe.events_since(cursor)
                    ),
                    args.turn_timeout_sec,
                )
                latest_tts = probe.latest_event_since(event_cursor, 'tts_completed')
                event_cursor = len(probe._events)
                used_text = injected_text
                if latest_tts is not None and str(latest_tts.get('dialog_act', '')) == 'relay_secretary':
                    conversation_completed = True
            else:
                for attempt_index, text in enumerate(candidate_texts, start=1):
                    wav_path = tts_client.synthesize_wav(
                        text,
                        request_id=f'full-stack-input-{index}-{attempt_index}',
                        timeout_sec=args.turn_timeout_sec,
                    )
                    mic = _play_wav(
                        ros_domain_id=args.ros_domain_id,
                        wav_path=wav_path,
                        log_dir=log_dir,
                        turn_index=index,
                    )
                    try:
                        playback_timeout = max(5.0, _wav_duration_sec(wav_path) + 2.0)
                        time.sleep(playback_timeout)
                    finally:
                        _terminate(mic)

                    probe.wait_for(
                        'tts_completed',
                        lambda cursor=event_cursor: any(
                            event.get('event_type') == 'tts_completed'
                            for event in probe.events_since(cursor)
                        ),
                        args.turn_timeout_sec,
                    )
                    latest_tts = probe.latest_event_since(event_cursor, 'tts_completed')
                    event_cursor = len(probe._events)
                    if latest_tts is None:
                        raise RuntimeError('tts_completed was not captured after playback')
                    used_text = text
                    latest_dialog_act = str(latest_tts.get('dialog_act', ''))
                    if latest_dialog_act == 'relay_secretary':
                        conversation_completed = True
                        break
                    if latest_dialog_act == expected_dialog_act:
                        break
                    if attempt_index == len(candidate_texts):
                        fallback_text = str(turn.get('fallback_text', '')).strip()
                        if not fallback_text:
                            raise AssertionError(
                                f"expected dialog_act={expected_dialog_act}, got {latest_dialog_act}"
                            )
                        asr_fallback.inject(fallback_text)
                        probe.wait_for(
                            f'tts_completed {expected_dialog_act} or relay_secretary after fallback',
                            lambda expected=expected_dialog_act, cursor=event_cursor: any(
                                event.get('event_type') == 'tts_completed'
                                and str(event.get('dialog_act', '')) in {expected, 'relay_secretary'}
                                for event in probe.events_since(cursor)
                            ),
                            args.turn_timeout_sec,
                        )
                        latest_tts = probe.latest_event_since(event_cursor, 'tts_completed')
                        event_cursor = len(probe._events)
                        used_text = fallback_text
                        if latest_tts is not None and str(latest_tts.get('dialog_act', '')) == 'relay_secretary':
                            conversation_completed = True
                        break
                    time.sleep(float(scenario.get('gap_sec', 1.0)))
                    _check_processes(handles)

            report['turns'].append(
                {
                    'index': index,
                    'input_text': used_text,
                    'expected_dialog_act': expected_dialog_act,
                }
            )
            if conversation_completed:
                break
            time.sleep(float(scenario.get('gap_sec', 1.0)))
            _check_processes(handles)

        if not conversation_completed:
            probe.wait_for(
                'secretary_reply',
                lambda cursor=event_cursor: any(
                    event.get('event_type') == 'secretary_reply' for event in probe.events_since(cursor)
                ),
                args.turn_timeout_sec,
            )
            probe.wait_for(
                f'tts_completed {expectations.get("final_dialog_act", "relay_secretary")}',
                lambda expected=str(expectations.get('final_dialog_act', 'relay_secretary')), cursor=event_cursor: any(
                    event.get('event_type') == 'tts_completed'
                    and event.get('dialog_act') == expected
                    for event in probe.events_since(cursor)
                ),
                args.turn_timeout_sec,
            )

        session = probe.latest_session()
        visitor = session.get('visitor_info', {}) if isinstance(session, dict) else {}
        expected_name = expectations.get('name')
        expected_affiliation = expectations.get('affiliation')
        expected_purpose = expectations.get('purpose')
        if expected_name is not None and visitor.get('name') != expected_name:
            raise AssertionError(f"expected name={expected_name}, got {visitor.get('name')}")
        if expected_affiliation is not None and visitor.get('affiliation') != expected_affiliation:
            raise AssertionError(
                f"expected affiliation={expected_affiliation}, got {visitor.get('affiliation')}"
            )
        if expected_purpose is not None and visitor.get('purpose') != expected_purpose:
            raise AssertionError(f"expected purpose={expected_purpose}, got {visitor.get('purpose')}")

        report['passed'] = True
        report['final_session'] = session
    finally:
        latest_session = probe.latest_session()
        if 'final_session' not in report and latest_session:
            report['final_session'] = latest_session
        if 'passed' not in report:
            latest_phase = str(latest_session.get('phase', '')) if isinstance(latest_session, dict) else ''
            saw_relay_completion = any(
                event.get('event_type') == 'tts_completed'
                and str(event.get('dialog_act', '')) == 'relay_secretary'
                and bool(event.get('success', False))
                for event in probe.events_since(0)
            )
            if latest_phase == 'completed' and saw_relay_completion:
                report['passed'] = True
        for handle in reversed(handles):
            _terminate(handle)
        report['logs'] = {
            handle.name: str(handle.log_path)
            for handle in handles
        }
        executor.shutdown()
        probe.destroy_node()
        tts_client.destroy_node()
        asr_fallback.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
