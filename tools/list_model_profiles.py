#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description='List available reception model profiles.')
    parser.add_argument(
        '--package-share',
        default=str(
            Path(__file__).resolve().parents[1]
        ),
        help='Path to ros2_reception_orchestrator package root/share-compatible tree',
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.package_share).resolve()))
    from ros2_reception_orchestrator.model_profiles import list_profiles  # noqa: WPS433

    root = Path(args.package_share).resolve()
    result = {
        task: [
            {
                'profile_name': profile.profile_name,
                'repo_id': profile.repo_id,
                'runner': profile.runner,
                'device': profile.data.get('device', ''),
                'dtype': profile.data.get('dtype', ''),
                'backend': profile.data.get('backend', profile.data.get('backend_class', '')),
                'latency_target_ms': profile.data.get('latency_target_ms'),
                'experimental': profile.data.get('experimental', False),
                'quantization': profile.data.get('quantization', ''),
                'base_model_id': profile.data.get('base_model_id', ''),
                'quality_notes': profile.data.get('quality_notes', ''),
            }
            for profile in list_profiles(root, task)
        ]
        for task in ('llm', 'asr', 'tts')
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
