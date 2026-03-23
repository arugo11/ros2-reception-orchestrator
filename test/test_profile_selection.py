from ros2_reception_orchestrator.profile_selection import DEFAULT_ASR_PROFILE
from ros2_reception_orchestrator.profile_selection import DEFAULT_LLM_PROFILE
from ros2_reception_orchestrator.profile_selection import DEFAULT_SHARED_PROFILE
from ros2_reception_orchestrator.profile_selection import DEFAULT_TTS_PROFILE
from ros2_reception_orchestrator.profile_selection import resolve_component_profiles


def test_component_profile_defaults_match_single_gpu_safe_stack() -> None:
    resolved = resolve_component_profiles()

    assert DEFAULT_SHARED_PROFILE == 'qwen_fullstack'
    assert DEFAULT_ASR_PROFILE == 'qwen3_asr_0_6b_cpu'
    assert DEFAULT_LLM_PROFILE == 'qwen35_4b_text'
    assert DEFAULT_TTS_PROFILE == 'qwen3_tts_gpu'
    assert resolved.shared_profile == 'qwen_fullstack'
    assert resolved.asr_profile == 'qwen3_asr_0_6b_cpu'
    assert resolved.llm_profile == 'qwen35_4b_text'
    assert resolved.tts_profile == 'qwen3_tts_gpu'


def test_explicit_component_profiles_override_shared_profile() -> None:
    resolved = resolve_component_profiles(
        profile_name='qwen_fullstack',
        asr_profile='qwen3_asr_1_7b_gpu',
        llm_profile='nemotron_nano_9b_japanese',
        tts_profile='qwen3_tts_gpu',
    )

    assert resolved.shared_profile == 'qwen_fullstack'
    assert resolved.asr_profile == 'qwen3_asr_1_7b_gpu'
    assert resolved.llm_profile == 'nemotron_nano_9b_japanese'
    assert resolved.tts_profile == 'qwen3_tts_gpu'
