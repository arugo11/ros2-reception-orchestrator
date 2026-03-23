from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SHARED_PROFILE = 'qwen_fullstack'
DEFAULT_ASR_PROFILE = 'qwen3_asr_0_6b_cpu'
DEFAULT_LLM_PROFILE = 'qwen35_4b_text'
DEFAULT_TTS_PROFILE = 'qwen3_tts_gpu'


@dataclass(frozen=True, slots=True)
class ComponentProfiles:
    shared_profile: str
    asr_profile: str
    llm_profile: str
    tts_profile: str


def resolve_component_profiles(
    *,
    profile_name: str = '',
    asr_profile: str = '',
    llm_profile: str = '',
    tts_profile: str = '',
) -> ComponentProfiles:
    shared = (profile_name or '').strip()
    return ComponentProfiles(
        shared_profile=shared or DEFAULT_SHARED_PROFILE,
        asr_profile=(asr_profile or '').strip() or shared or DEFAULT_ASR_PROFILE,
        llm_profile=(llm_profile or '').strip() or shared or DEFAULT_LLM_PROFILE,
        tts_profile=(tts_profile or '').strip() or shared or DEFAULT_TTS_PROFILE,
    )
