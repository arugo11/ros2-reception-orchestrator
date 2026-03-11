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
DEFAULT_DISCORD_PARENT_CHANNEL = 'discord:1479886028549525678:1479886034840846499'
READY_MARKER = 'All backends ready: ASR, LLM, TTS, and chat bridge are available'
FAILURE_MARKERS = (
    'vLLM process exited early with code',
    'activate failed',
    'Engine core initialization failed',
    'Unknown gguf model_type',
    'backend error:',
    'Gemini provider requires api_key parameter or GEMINI_API_KEY env var',
    'LLM backend probe failed:',
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


def _load_env_var(key: str) -> str:
    direct = os.environ.get(key, '').strip()
    if direct:
        return direct

    dotenv = REPO_ROOT / 'src' / 'ros2_reception_orchestrator' / '.env'
    if not dotenv.is_file():
        return ''

    for raw_line in dotenv.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        lhs, rhs = line.split('=', 1)
        if lhs.strip() != key:
            continue
        return rhs.strip().strip('"').strip("'")
    return ''


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


def _launch_stack(
    profile_name: str,
    llm_provider: str,
    asr_profile: str,
    tts_profile: str,
    discord_parent_channel_id: str,
) -> subprocess.Popen[str]:
    cmd = (
        'source /opt/ros/jazzy/setup.bash && '
        f'source {REPO_ROOT / "install" / "setup.bash"} && '
        'ros2 launch ros2_reception_orchestrator reception_bringup.launch.py '
        f'discord_parent_channel_id:={discord_parent_channel_id} '
        'enable_mic_input:=false '
        'playback_enabled:=false '
        f'profile_name:={profile_name} '
        f'llm_provider:={llm_provider} '
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
        env=os.environ.copy(),
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
        text = line.rstrip()
        logs.append(text)
        if READY_MARKER in text:
            return {'ready': True, 'logs': logs}
        if any(marker in text for marker in FAILURE_MARKERS):
            return {'ready': False, 'logs': logs, 'failure': text}
    return {
        'ready': False,
        'logs': logs,
        'failure': 'startup timeout waiting for composite ready marker',
    }


def _run_scenario(scenario_path: Path, output_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(TOOLS_ROOT / 'bench_reception_stack_v2.py'),
        '--scenario',
        str(scenario_path),
        '--wait-ready-timeout-sec',
        '0',
        '--sample-gpu',
        '--output',
        str(output_path),
    ]
    completed = _run(cmd, cwd=REPO_ROOT, capture_output=True)
    result_data: dict[str, Any] = {}
    if output_path.is_file():
        result_data = json.loads(output_path.read_text(encoding='utf-8'))
    return {
        'returncode': completed.returncode,
        'stdout': completed.stdout,
        'stderr': completed.stderr,
        'result': result_data,
    }


def _summarize_scenario(scenario_result: dict[str, Any], scenario_name: str, output_path: Path) -> dict[str, Any]:
    result = scenario_result.get('result', {})
    expectation = result.get('expectation_check', {})
    session = result.get('last_session_state', {}).get('session', {})
    turns = result.get('turns', [])
    tts_latencies = [
        turn.get('tts_started_latency_ms')
        for turn in turns
        if isinstance(turn, dict) and turn.get('tts_started_latency_ms') is not None
    ]
    return {
        'scenario': scenario_name,
        'output_path': str(output_path),
        'returncode': scenario_result.get('returncode'),
        'passed': bool(expectation.get('passed', False)),
        'failures': expectation.get('failures', []),
        'phase': session.get('phase'),
        'visitor_info': session.get('visitor_info', {}),
        'tts_started_latency_ms': tts_latencies,
        'gpu_before': result.get('gpu_before'),
        'gpu_after': result.get('gpu_after'),
    }


def _collect_stats(report: dict[str, Any]) -> dict[str, Any]:
    providers = {'vllm': {}, 'gemini': {}}
    runs = report.get('runs', [])

    for provider in providers:
        provider_runs = [run for run in runs if run.get('provider') == provider]
        startup_ok = sum(1 for run in provider_runs if run.get('startup_ready'))
        scenario_rows = [
            row
            for run in provider_runs
            for row in run.get('scenarios', [])
            if isinstance(row, dict)
        ]
        scenario_pass = sum(1 for row in scenario_rows if row.get('passed'))
        scenario_failures = [
            failure
            for row in scenario_rows
            for failure in row.get('failures', [])
            if isinstance(failure, str)
        ]
        providers[provider] = {
            'run_count': len(provider_runs),
            'startup_ready_count': startup_ok,
            'startup_ready_rate': _ratio(startup_ok, len(provider_runs)),
            'scenario_total': len(scenario_rows),
            'scenario_pass_count': scenario_pass,
            'scenario_pass_rate': _ratio(scenario_pass, len(scenario_rows)),
            'scenario_failure_counts': _count_items(scenario_failures),
        }

    pairwise = _pairwise_outcomes(runs)
    classification = _classify_failures(runs, pairwise)

    return {
        'providers': providers,
        'pairwise': pairwise,
        'classification': classification,
    }


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(float(num) / float(den), 4)


def _count_items(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _pairwise_outcomes(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for run in runs:
        indexed[(int(run.get('repeat', -1)), str(run.get('provider')))] = run

    outcomes: list[dict[str, Any]] = []
    repeats = sorted({int(run.get('repeat', -1)) for run in runs})
    for repeat in repeats:
        vllm = indexed.get((repeat, 'vllm'))
        gemini = indexed.get((repeat, 'gemini'))
        if not vllm or not gemini:
            continue

        scenario_names = sorted(
            {
                row.get('scenario')
                for row in vllm.get('scenarios', []) + gemini.get('scenarios', [])
                if isinstance(row, dict) and row.get('scenario')
            }
        )
        for scenario in scenario_names:
            vrow = _find_scenario(vllm, scenario)
            grow = _find_scenario(gemini, scenario)
            if not vrow or not grow:
                continue
            outcomes.append(
                {
                    'repeat': repeat,
                    'scenario': scenario,
                    'vllm_passed': bool(vrow.get('passed')),
                    'gemini_passed': bool(grow.get('passed')),
                    'vllm_failures': list(vrow.get('failures', [])),
                    'gemini_failures': list(grow.get('failures', [])),
                }
            )
    return outcomes


def _find_scenario(run: dict[str, Any], scenario: str) -> dict[str, Any] | None:
    for row in run.get('scenarios', []):
        if isinstance(row, dict) and row.get('scenario') == scenario:
            return row
    return None


def _classify_failures(runs: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> dict[str, Any]:
    design_findings: list[str] = []
    model_findings: list[str] = []
    config_findings: list[str] = []

    for row in pairwise:
        if row['vllm_passed'] and row['gemini_passed']:
            continue
        if (not row['vllm_passed']) and (not row['gemini_passed']):
            design_findings.append(
                f"repeat={row['repeat']} scenario={row['scenario']} both_failed "
                f"vllm={row['vllm_failures']} gemini={row['gemini_failures']}"
            )
        elif (not row['vllm_passed']) != (not row['gemini_passed']):
            model_findings.append(
                f"repeat={row['repeat']} scenario={row['scenario']} one_side_failed "
                f"vllm_passed={row['vllm_passed']} gemini_passed={row['gemini_passed']}"
            )

    for run in runs:
        if run.get('provider') != 'gemini' or run.get('startup_ready'):
            continue
        msg = str(run.get('startup_failure') or '')
        lowered = msg.lower()
        if 'api_key' in lowered or 'gemini_api_key' in lowered or 'provider' in lowered:
            config_findings.append(
                f"repeat={run.get('repeat')} startup_failure={msg}"
            )

    return {
        'design_or_orchestration_failures': design_findings,
        'model_or_provider_failures': model_findings,
        'configuration_failures': config_findings,
    }


def _write_markdown(report: dict[str, Any], summary: dict[str, Any], path: Path) -> None:
    providers = summary.get('providers', {})
    classification = summary.get('classification', {})

    lines: list[str] = []
    lines.append('# vLLM vs Gemini Benchmark Summary')
    lines.append('')
    lines.append('## Provider Metrics')
    lines.append('')
    lines.append('| provider | runs | startup_ready | startup_rate | scenario_pass | scenario_total | scenario_pass_rate |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')

    for provider in ('vllm', 'gemini'):
        metric = providers.get(provider, {})
        lines.append(
            f"| {provider} | {metric.get('run_count', 0)} | {metric.get('startup_ready_count', 0)} "
            f"| {metric.get('startup_ready_rate', 0.0):.4f} | {metric.get('scenario_pass_count', 0)} "
            f"| {metric.get('scenario_total', 0)} | {metric.get('scenario_pass_rate', 0.0):.4f} |"
        )

    lines.append('')
    lines.append('## Failure Classification')
    lines.append('')

    for title, key in (
        ('Design/Orchestration candidates', 'design_or_orchestration_failures'),
        ('Model/Provider candidates', 'model_or_provider_failures'),
        ('Configuration candidates', 'configuration_failures'),
    ):
        lines.append(f'### {title}')
        findings = classification.get(key, [])
        if findings:
            for finding in findings:
                lines.append(f'- {finding}')
        else:
            lines.append('- none')
        lines.append('')

    lines.append('## Artifacts')
    lines.append('')
    lines.append(f"- report_json: `{report.get('output_path', '')}`")
    lines.append('- per_run_outputs: `/tmp/reception_profile_bench/<provider>/run_<n>/*.json`')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _validate_ros_setup() -> None:
    check = _run(
        ['bash', '-lc', 'source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 --help >/dev/null'],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if check.returncode != 0:
        raise RuntimeError('ROS setup check failed: source /opt/ros/jazzy/setup.bash && source install/setup.bash')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run repeated side-by-side benchmark for vLLM vs Gemini.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--startup-timeout-sec', type=float, default=180.0)
    parser.add_argument('--asr-profile', default='qwen3_asr_gpu')
    parser.add_argument('--tts-profile', default='qwen3_tts_gpu')
    parser.add_argument('--vllm-profile', default='qwen_fullstack')
    parser.add_argument('--gemini-profile', default='gemini_fullstack_experiment')
    parser.add_argument('--discord-parent-channel-id', default=DEFAULT_DISCORD_PARENT_CHANNEL)
    parser.add_argument('--scenario', action='append', default=[])
    parser.add_argument('--output', default='')
    parser.add_argument('--summary-md', default='')
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError('--repeats must be >= 1')

    _validate_ros_setup()

    api_key = _load_env_var('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY not found in environment or src/ros2_reception_orchestrator/.env')
    os.environ['GEMINI_API_KEY'] = api_key

    scenarios = [Path(item).resolve() for item in (args.scenario or [])] or DEFAULT_SCENARIOS
    for scenario in scenarios:
        if not scenario.is_file():
            raise FileNotFoundError(f'scenario not found: {scenario}')

    ts = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    output_path = Path(args.output).resolve() if args.output else Path(f'/tmp/reception_profile_bench/vllm_vs_gemini_{ts}.json')
    summary_md_path = Path(args.summary_md).resolve() if args.summary_md else output_path.with_suffix('.md')

    report: dict[str, Any] = {
        'generated_at_epoch': time.time(),
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'repeats': args.repeats,
        'scenarios': [str(path) for path in scenarios],
        'asr_profile': args.asr_profile,
        'tts_profile': args.tts_profile,
        'providers': {
            'vllm': {
                'profile_name': args.vllm_profile,
                'llm_provider': 'vllm',
            },
            'gemini': {
                'profile_name': args.gemini_profile,
                'llm_provider': 'gemini',
            },
        },
        'runs': [],
    }

    provider_order = [
        ('vllm', args.vllm_profile, 'vllm'),
        ('gemini', args.gemini_profile, 'gemini'),
    ]

    for repeat in range(1, args.repeats + 1):
        for provider_key, profile_name, llm_provider in provider_order:
            _cleanup_stack()

            run_root = Path(f'/tmp/reception_profile_bench/{provider_key}/run_{repeat}')
            run_root.mkdir(parents=True, exist_ok=True)

            launch = _launch_stack(
                profile_name=profile_name,
                llm_provider=llm_provider,
                asr_profile=args.asr_profile,
                tts_profile=args.tts_profile,
                discord_parent_channel_id=args.discord_parent_channel_id,
            )
            ready = _wait_for_ready(launch, args.startup_timeout_sec)
            run_record: dict[str, Any] = {
                'repeat': repeat,
                'provider': provider_key,
                'profile_name': profile_name,
                'llm_provider': llm_provider,
                'startup_ready': bool(ready.get('ready')),
                'startup_failure': ready.get('failure'),
                'startup_log_tail': ready.get('logs', [])[-80:],
                'scenarios': [],
            }

            if ready.get('ready'):
                for scenario in scenarios:
                    output_file = run_root / f'{scenario.stem}.json'
                    scenario_result = _run_scenario(scenario, output_file)
                    run_record['scenarios'].append(
                        _summarize_scenario(scenario_result, scenario.stem, output_file)
                    )

            if launch.poll() is None:
                launch.send_signal(signal.SIGINT)
                try:
                    launch.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    launch.kill()

            _cleanup_stack()
            report['runs'].append(run_record)

    summary = _collect_stats(report)
    report['summary'] = summary
    report['output_path'] = str(output_path)
    report['summary_md_path'] = str(summary_md_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_markdown(report, summary, summary_md_path)

    print(json.dumps({'output': str(output_path), 'summary_md': str(summary_md_path)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
