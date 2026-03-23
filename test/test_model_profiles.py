from __future__ import annotations

from pathlib import Path

import pytest

from ros2_reception_orchestrator.model_profiles import list_profiles
from ros2_reception_orchestrator.model_profiles import load_profile


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_default_llm_profile():
    profile = load_profile(_package_root(), 'llm', 'reception_default')

    assert profile.profile_name == 'reception_default'
    assert profile.repo_id == 'Qwen/Qwen2.5-1.5B-Instruct'
    assert profile.runner == 'vllm'
    assert profile.require_list_of_strings('startup_args') == [
        '--enforce-eager',
        '--generation-config',
        'vllm',
        '--max-num-seqs',
        '1',
    ]


def test_load_gguf_profile_metadata():
    profile = load_profile(_package_root(), 'llm', 'qwen35_2b_gguf_q5km')

    assert profile.profile_name == 'qwen35_2b_gguf_q5km'
    assert profile.repo_id == 'unsloth/Qwen3.5-2B-GGUF:Q5_K_M'
    assert profile.optional_bool('experimental', False) is True
    assert profile.optional_string('quantization') == 'Q5_K_M'
    assert profile.optional_string('base_model_id') == 'Qwen/Qwen3.5-2B'


def test_list_profiles_contains_supported_candidates():
    llm_profiles = {profile.profile_name for profile in list_profiles(_package_root(), 'llm')}
    asr_profiles = {profile.profile_name for profile in list_profiles(_package_root(), 'asr')}
    tts_profiles = {profile.profile_name for profile in list_profiles(_package_root(), 'tts')}

    assert 'qwen35_2b_gguf_q5km' in llm_profiles
    assert 'qwen35_4b_text' in llm_profiles
    assert 'nemotron_nano_9b_japanese' in llm_profiles
    assert 'qwen3_asr_gpu' in asr_profiles
    assert 'qwen3_asr_0_6b_cpu' in asr_profiles
    assert 'qwen3_asr_1_7b_gpu' in asr_profiles
    assert 'kotoba_whisper_gpu' in asr_profiles
    assert 'qwen3_tts_gpu' in tts_profiles
    assert 'speecht5_gpu' in tts_profiles


def test_invalid_profile_validation(tmp_path: Path):
    profile_dir = tmp_path / 'config' / 'model_profiles' / 'llm'
    profile_dir.mkdir(parents=True)
    profile_dir.joinpath('broken.yaml').write_text(
        '\n'.join(
            [
                'profile_name: broken',
                'task: llm',
                'runner: not_a_runner',
                'repo_id: ""',
            ]
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError):
        load_profile(tmp_path, 'llm', 'broken')
