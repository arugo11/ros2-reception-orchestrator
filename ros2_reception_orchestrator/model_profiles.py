from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_TASKS = {'llm', 'asr', 'tts'}
_ALLOWED_RUNNERS = {'vllm', 'transformers', 'custom_python'}


@dataclass(slots=True)
class ModelProfile:
    task: str
    profile_name: str
    repo_id: str
    runner: str
    data: dict[str, Any]

    def require_string(self, key: str, default: str = '') -> str:
        value = self.data.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"Profile '{self.profile_name}' key '{key}' must be a string")
        return value.strip()

    def require_bool(self, key: str, default: bool = False) -> bool:
        value = self.data.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"Profile '{self.profile_name}' key '{key}' must be a bool")
        return value

    def require_list_of_strings(self, key: str) -> list[str]:
        value = self.data.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(
                f"Profile '{self.profile_name}' key '{key}' must be a list of strings"
            )
        return [item.strip() for item in value]

    def optional_string(self, key: str, default: str = '') -> str:
        value = self.data.get(key, default)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ValueError(f"Profile '{self.profile_name}' key '{key}' must be a string")
        return value.strip()

    def optional_bool(self, key: str, default: bool = False) -> bool:
        value = self.data.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"Profile '{self.profile_name}' key '{key}' must be a bool")
        return value


def profile_root(package_share: str | Path) -> Path:
    return Path(package_share) / 'config' / 'model_profiles'


def list_profiles(package_share: str | Path, task: str) -> list[ModelProfile]:
    root = profile_root(package_share) / task
    profiles: list[ModelProfile] = []
    for path in sorted(root.glob('*.yaml')):
        profiles.append(load_profile(package_share, task, path.stem))
    return profiles


def load_profile(package_share: str | Path, task: str, profile_name: str) -> ModelProfile:
    path = profile_root(package_share) / task / f'{profile_name}.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'Profile not found: {path}')
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Profile must be a mapping: {path}')
    resolved_task = str(data.get('task', task)).strip()
    if resolved_task not in _ALLOWED_TASKS:
        raise ValueError(f'Profile task must be one of {_ALLOWED_TASKS}: {path}')
    if resolved_task != task:
        raise ValueError(f'Profile task mismatch for {path}: expected {task}, got {resolved_task}')
    profile = ModelProfile(
        task=resolved_task,
        profile_name=str(data.get('profile_name', profile_name)).strip() or profile_name,
        repo_id=str(data.get('repo_id', '')).strip(),
        runner=str(data.get('runner', '')).strip(),
        data=data,
    )
    _validate_profile(profile, path)
    return profile


def _validate_profile(profile: ModelProfile, path: Path) -> None:
    if not profile.repo_id:
        raise ValueError(f'Profile repo_id is required: {path}')
    if profile.runner not in _ALLOWED_RUNNERS:
        raise ValueError(f'Profile runner must be one of {_ALLOWED_RUNNERS}: {path}')
    if 'startup_args' in profile.data:
        profile.require_list_of_strings('startup_args')
    if 'expected_languages' in profile.data:
        profile.require_list_of_strings('expected_languages')
