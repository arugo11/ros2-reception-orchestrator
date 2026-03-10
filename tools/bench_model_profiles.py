#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = [
    TOOLS_ROOT / 'scenarios' / 'reception_happy_path.json',
    TOOLS_ROOT / 'scenarios' / 'reception_correction.json',
    TOOLS_ROOT / 'scenarios' / 'reception_overlap.json',
]
READY_MARKER = 'All backends ready: ASR, LLM, TTS, and chat bridge are available'
FAILURE_MARKERS = (
    'vLLM process exited early with code',
    'activate failed',
    'Engine core initialization failed',
    'Unknown gguf model_type',
    'backend error:',
)
PROCESS_PATTERNS = [
    'ros2 launch ros2_reception_orchestrator reception_bringup.launch.py',
    '/install/ros2_reception_orchestrator/lib/ros2_reception_orchestrator/reception_orchestrator',
    '/install/asr_streaming_node/lib/asr_streaming_node/asr_streaming_node',
    '/install/ros2_vllm/lib/ros2_vllm/llm_chat_node',
    '/install/ros2_vllm/lib/ros2_vllm/vllm_server_node',
    '/install/tts_server/lib/tts_server/tts_server',
    '/install/ros2_chat/lib/ros2_chat/chat_bridge_node',
    '/install/mic_input_node/lib/mic_input_node/mic_input_node',
    'vllm serve',
]


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, check=False, **kwargs)


def _cleanup_stack() -> None:
    output = _run(['ps', '-eo', 'pid,args'], capture_output=True).stdout
    me = os.getpid()
    parent = os.getppid()
    terminated: list[int] = []
    for line in output.splitlines()[1:]:
        try:
            pid_s, args = line.strip().split(' ', 1)
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in (me, parent):
            continue
        if any(pattern in args for pattern in PROCESS_PATTERNS):
            try:
                os.kill(pid, signal.SIGTERM)
                terminated.append(pid)
            except ProcessLookupError:
                pass
    if terminated:
        time.sleep(2.0)


def _launch_stack(llm_profile: str, asr_profile: str, tts_profile: str) -> subprocess.Popen[str]:
    cmd = (
        'source /opt/ros/jazzy/setup.bash && '
        f'source {REPO_ROOT / "install" / "setup.bash"} && '
        'ros2 launch ros2_reception_orchestrator reception_bringup.launch.py '
        'discord_parent_channel_id:=discord:1479886028549525678:1479886034840846499 '
        'enable_mic_input:=false '
        'playback_enabled:=false '
        f'llm_profile:={llm_profile} '
        f'asr_profile:={asr_profile} '
        f'tts_profile:={tts_profile}'
    )
    return subprocess.Popen(
        ['bash', '-lc', cmd],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _wait_for_ready(process: subprocess.Popen[str], timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    logs: list[str] = []
    while time.monotonic() < deadline:
        line = process.stdout.readline() if process.stdout is not None else ''
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue
        logs.append(line.rstrip())
        if READY_MARKER in line:
            return {'ready': True, 'logs': logs}
        if any(marker in line for marker in FAILURE_MARKERS):
            return {'ready': False, 'logs': logs, 'failure': line.rstrip()}
    return {
        'ready': False,
        'logs': logs,
        'failure': 'startup timeout waiting for composite ready marker',
    }


def _run_scenario(scenario_path: Path, output_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(TOOLS_ROOT / 'bench_reception_stack.py'),
        '--scenario',
        str(scenario_path),
        '--wait-ready-timeout-sec',
        '0',
        '--sample-gpu',
        '--output',
        str(output_path),
    ]
    completed = _run(cmd, cwd=REPO_ROOT, capture_output=True)
    data: dict[str, Any] = {}
    if output_path.is_file():
        data = json.loads(output_path.read_text(encoding='utf-8'))
    return {
        'returncode': completed.returncode,
        'stdout': completed.stdout,
        'stderr': completed.stderr,
        'result': data,
    }


def _summarize_scenario(data: dict[str, Any]) -> dict[str, Any]:
    result = data.get('result', {})
    expectation = result.get('expectation_check', {})
    turns = result.get('turns', [])
    tts_latencies = [
        turn.get('tts_started_latency_ms')
        for turn in turns
        if isinstance(turn, dict) and turn.get('tts_started_latency_ms') is not None
    ]
    return {
        'passed': expectation.get('passed'),
        'failures': expectation.get('failures', []),
        'turn_count': len(turns) if isinstance(turns, list) else None,
        'last_session_state': result.get('last_session_state'),
        'tts_started_latency_ms': tts_latencies,
        'gpu_before': result.get('gpu_before'),
        'gpu_after': result.get('gpu_after'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Serial benchmark runner for reception model profiles.')
    parser.add_argument('--llm-profile', action='append', required=True)
    parser.add_argument('--asr-profile', default='qwen3_asr_gpu')
    parser.add_argument('--tts-profile', default='qwen3_tts_gpu')
    parser.add_argument('--startup-timeout-sec', type=float, default=120.0)
    parser.add_argument('--scenario', action='append', default=[])
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    scenarios = [Path(item).resolve() for item in (args.scenario or [])] or DEFAULT_SCENARIOS
    run_dir = Path('/tmp/reception_profile_bench')
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        'generated_at_epoch': time.time(),
        'llm_profiles': [],
        'asr_profile': args.asr_profile,
        'tts_profile': args.tts_profile,
        'scenarios': [str(path) for path in scenarios],
    }

    for llm_profile in args.llm_profile:
        _cleanup_stack()
        launch = _launch_stack(llm_profile, args.asr_profile, args.tts_profile)
        ready = _wait_for_ready(launch, args.startup_timeout_sec)
        profile_result: dict[str, Any] = {
            'llm_profile': llm_profile,
            'startup_ready': bool(ready.get('ready')),
            'startup_failure': ready.get('failure'),
            'startup_log_tail': ready.get('logs', [])[-40:],
            'scenarios': [],
        }
        if ready.get('ready'):
            for scenario in scenarios:
                output_path = run_dir / f'{llm_profile}__{scenario.stem}.json'
                scenario_result = _run_scenario(scenario, output_path)
                profile_result['scenarios'].append(
                    {
                        'scenario': scenario.stem,
                        **_summarize_scenario(scenario_result),
                    }
                )
        if launch.poll() is None:
            launch.send_signal(signal.SIGINT)
            try:
                launch.wait(timeout=15)
            except subprocess.TimeoutExpired:
                launch.kill()
        _cleanup_stack()
        report['llm_profiles'].append(profile_result)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
