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


def _launch_stack(
    workspace: Path,
    *,
    profile_name: str,
    llm_provider: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    command = (
        "source /opt/ros/jazzy/setup.bash && "
        f"source {workspace / 'install' / 'setup.bash'} && "
        "ros2 launch ros2_reception_orchestrator reception_bringup.launch.py "
        f"profile_name:={profile_name} "
        f"llm_provider:={llm_provider} "
        "enable_mic_input:=false "
        "playback_enabled:=false "
        "discord_parent_channel_id:=discord:1479886028549525678:1479886034840846499 "
        "gpu_memory_utilization:=0.25"
    )
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        ['bash', '-lc', command],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )


def _run_evaluator(
    workspace: Path,
    *,
    scenario: Path,
    timeout_sec: float,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    command = (
        "source /opt/ros/jazzy/setup.bash && "
        f"source {workspace / 'install' / 'setup.bash'} && "
        f"{sys.executable} {workspace / 'src/ros2_reception_orchestrator/tools/live_stack_e2e.py'} "
        f"{scenario} --timeout-sec {timeout_sec} --output {output_path}"
    )
    return subprocess.run(
        ['bash', '-lc', command],
        cwd=str(workspace),
        text=True,
        capture_output=True,
    )


def _wait_for_ready(process: subprocess.Popen[str], timeout_sec: float) -> list[str]:
    deadline = time.monotonic() + timeout_sec
    lines: list[str] = []
    assert process.stdout is not None
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                raise RuntimeError('stack exited before ready')
            time.sleep(0.1)
            continue
        lines.append(line.rstrip())
        if 'All backends ready: ASR, LLM, TTS, and chat bridge are available' in line:
            return lines
    raise TimeoutError('stack did not become ready in time')


def _stop_stack(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run live_stack_e2e with a clean restart per scenario.')
    parser.add_argument('scenarios', nargs='+', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('/workspaces/ros2-workspace-template'))
    parser.add_argument('--timeout-sec', type=float, default=90.0)
    parser.add_argument('--stack-ready-timeout-sec', type=float, default=120.0)
    parser.add_argument('--output', type=Path, default=Path('/tmp/reception_live_stack_e2e_restart.json'))
    parser.add_argument('--profile-name', type=str, default='qwen_fullstack')
    parser.add_argument('--llm-provider', type=str, default='vllm')
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    overall_passed = True

    for scenario in args.scenarios:
        stack = _launch_stack(
            args.workspace,
            profile_name=args.profile_name,
            llm_provider=args.llm_provider,
        )
        ready_log: list[str] = []
        try:
            ready_log = _wait_for_ready(stack, args.stack_ready_timeout_sec)
            scenario_output = Path(f'/tmp/{scenario.stem}_restart_eval.json')
            eval_proc = _run_evaluator(
                args.workspace,
                scenario=scenario,
                timeout_sec=args.timeout_sec,
                output_path=scenario_output,
            )
            if not scenario_output.exists():
                raise RuntimeError(
                    'evaluator did not produce output file\n'
                    f'stdout:\n{eval_proc.stdout}\n'
                    f'stderr:\n{eval_proc.stderr}'
                )
            result = json.loads(scenario_output.read_text(encoding='utf-8'))
            result['scenario_output'] = str(scenario_output)
            result['evaluator_returncode'] = eval_proc.returncode
            result['stack_ready_log_tail'] = ready_log[-20:]
        finally:
            _stop_stack(stack)
        results.append(result)
        overall_passed = overall_passed and bool(result.get('passed'))

    report = {
        'passed': overall_passed,
        'results': results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
