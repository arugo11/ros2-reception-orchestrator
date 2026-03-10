import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ros2_reception_orchestrator.model_profiles import load_profile


_LOCK_PATH = Path('/tmp/ros2_reception_bringup.lock')


def _join_env_paths(*parts: str) -> str:
    return ':'.join(part for part in parts if part)


def _acquire_singleton_lock(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    del context, args, kwargs
    if _LOCK_PATH.exists():
        try:
            pid = int(_LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, OSError):
            _LOCK_PATH.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f'reception_bringup is already running (pid={pid}). Stop the existing stack first.'
            )
    _LOCK_PATH.write_text(str(os.getpid()), encoding='utf-8')
    return []


def _release_singleton_lock(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    del context, args, kwargs
    _LOCK_PATH.unlink(missing_ok=True)
    return []


def _build_runtime(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    del args, kwargs
    package_share = get_package_share_directory('ros2_reception_orchestrator')
    llm_profile = load_profile(package_share, 'llm', LaunchConfiguration('llm_profile').perform(context))
    asr_profile = load_profile(package_share, 'asr', LaunchConfiguration('asr_profile').perform(context))
    tts_profile = load_profile(package_share, 'tts', LaunchConfiguration('tts_profile').perform(context))

    reception_params_file = LaunchConfiguration('reception_params_file').perform(context)
    discord_params_file = LaunchConfiguration('discord_params_file').perform(context)
    vllm_params_file = LaunchConfiguration('vllm_params_file').perform(context)
    asr_config_file = LaunchConfiguration('asr_config_file').perform(context)
    tts_config_file = LaunchConfiguration('tts_config_file').perform(context)
    discord_parent_channel_id = LaunchConfiguration('discord_parent_channel_id').perform(context)
    session_inactivity_reset_sec = LaunchConfiguration('session_inactivity_reset_sec')
    audio_backend = LaunchConfiguration('audio_backend').perform(context)
    wav_file_path = LaunchConfiguration('wav_file_path').perform(context)
    alsa_device = LaunchConfiguration('alsa_device').perform(context)
    pulse_source = LaunchConfiguration('pulse_source').perform(context)
    pulse_server = LaunchConfiguration('pulse_server').perform(context)
    enable_mic_input = LaunchConfiguration('enable_mic_input')
    playback_enabled = LaunchConfiguration('playback_enabled')
    playback_device = LaunchConfiguration('playback_device')
    playback_sample_rate_hz = LaunchConfiguration('playback_sample_rate_hz')
    resolved_playback_sample_rate_hz = str(
        tts_profile.data.get('playback_sample_rate_hz', 24000)
    )
    if str(playback_sample_rate_hz.perform(context)).strip():
        resolved_playback_sample_rate_hz = str(playback_sample_rate_hz.perform(context))

    actions = [
        GroupAction(
            condition=IfCondition(enable_mic_input),
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution(
                            [FindPackageShare('mic_input_node'), 'launch', 'mic_input_node.launch.py']
                        )
                    ),
                    launch_arguments={
                        'audio_backend': audio_backend,
                        'wav_file_path': wav_file_path,
                        'alsa_device': alsa_device,
                        'pulse_source': pulse_source,
                        'pulse_server': pulse_server,
                    }.items(),
                ),
            ],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare('asr_streaming_node'), 'launch', 'asr_streaming_node.launch.py']
                )
            ),
            launch_arguments={
                'config_file': asr_config_file,
                'model_source': asr_profile.repo_id,
                'model_backend': asr_profile.require_string('backend', 'whisper'),
                'runtime_device': str(asr_profile.data.get('device', 'cuda:0')),
                'runtime_compute_type': str(asr_profile.data.get('dtype', 'auto')),
                'runtime_require_gpu': str(asr_profile.require_bool('require_gpu', True)).lower(),
                'continuous_enabled': 'true',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare('ros2_vllm'), 'launch', 'vllm_bringup.launch.py']
                )
            ),
            launch_arguments={
                'params_file': vllm_params_file,
                'model_name': llm_profile.repo_id,
                'endpoint_prefix': 'llm',
                'vllm_port': '8000',
                'max_model_len': str(llm_profile.data.get('context_len', 1024)),
                'dtype': llm_profile.require_string('dtype', 'bfloat16'),
                'gpu_memory_utilization': str(
                    llm_profile.data.get('gpu_memory_utilization', 0.35)
                ),
                'trust_remote_code': str(llm_profile.require_bool('trust_remote_code', True)).lower(),
                'vllm_extra_args_json': json.dumps(llm_profile.require_list_of_strings('startup_args')),
                'wandb_enabled': 'false',
                'reuse_existing_backend': 'false',
                'replace_existing_backend': 'true',
                'server_node_name': 'vllm_server_node',
                'chat_node_name': 'llm_chat_node',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare('ros2_chat'), 'launch', 'discord_bridge.launch.py']
                )
            ),
            launch_arguments={'params_file': discord_params_file}.items(),
        ),
        Node(
            package='tts_server',
            executable='tts_server',
            name='tts_server',
            output='screen',
            additional_env={
                'PYTHONPATH': _join_env_paths(
                    os.environ.get('QWEN_TTS_PYTHONPATH', ''),
                    os.environ.get('QWEN_RUNTIME_PYTHONPATH', ''),
                    os.environ.get('PYTHONPATH', ''),
                ),
                'PATH': _join_env_paths(
                    os.environ.get('QWEN_SOX_BIN_DIR', ''),
                    os.environ.get('PATH', ''),
                ),
                'LD_LIBRARY_PATH': _join_env_paths(
                    os.environ.get('QWEN_SOX_LIB_DIR', ''),
                    os.environ.get('LD_LIBRARY_PATH', ''),
                ),
            },
            parameters=[
                tts_config_file,
                {
                    'backend.class': tts_profile.require_string(
                        'backend_class',
                        'tts_server.backends.qwen_tts:QwenTtsBackend',
                    ),
                    'hf.model_id': tts_profile.repo_id,
                    'runtime.device': str(tts_profile.data.get('device', 'cuda:0')),
                    'runtime.torch_dtype': str(tts_profile.data.get('dtype', 'bfloat16')),
                    'qwen.voice_instruction': tts_profile.require_string(
                        'voice_instruction',
                        '落ち着いた丁寧な日本語の大学受付音声',
                    ),
                    'warmup.enabled': bool(tts_profile.data.get('warmup_enabled', False)),
                    'warmup.text': str(tts_profile.data.get('warmup_text', '受付を開始します。')),
                    'playback.enabled': ParameterValue(playback_enabled, value_type=bool),
                    'playback.device': playback_device,
                    'playback.sample_rate_hz': ParameterValue(
                        int(resolved_playback_sample_rate_hz), value_type=int
                    ),
                },
            ],
        ),
        Node(
            package='ros2_reception_orchestrator',
            executable='reception_orchestrator',
            name='reception_orchestrator',
            output='screen',
            parameters=[
                reception_params_file,
                {
                    'discord.parent_channel_id': discord_parent_channel_id,
                    'session.inactivity_reset_sec': ParameterValue(
                        session_inactivity_reset_sec, value_type=int
                    ),
                },
            ],
        ),
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            OpaqueFunction(function=_acquire_singleton_lock),
            RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=_release_singleton_lock)])),
            DeclareLaunchArgument(
                'reception_params_file',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('ros2_reception_orchestrator'), 'config', 'params.yaml']
                ),
            ),
            DeclareLaunchArgument(
                'discord_params_file',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('ros2_chat'), 'config', 'params.yaml']
                ),
            ),
            DeclareLaunchArgument(
                'vllm_params_file',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('ros2_vllm'), 'config', 'params.yaml']
                ),
            ),
            DeclareLaunchArgument(
                'asr_config_file',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('asr_streaming_node'), 'config', 'config.yaml']
                ),
            ),
            DeclareLaunchArgument(
                'tts_config_file',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('tts_server'), 'config', 'tts_server.yaml']
                ),
            ),
            DeclareLaunchArgument('discord_parent_channel_id', default_value=''),
            DeclareLaunchArgument('session_inactivity_reset_sec', default_value='30'),
            DeclareLaunchArgument('audio_backend', default_value='auto'),
            DeclareLaunchArgument('wav_file_path', default_value=''),
            DeclareLaunchArgument('alsa_device', default_value='default'),
            DeclareLaunchArgument('pulse_source', default_value='RDPSource'),
            DeclareLaunchArgument('pulse_server', default_value='unix:/mnt/wslg/PulseServer'),
            DeclareLaunchArgument('enable_mic_input', default_value='true'),
            DeclareLaunchArgument('playback_enabled', default_value='false'),
            DeclareLaunchArgument('playback_device', default_value=''),
            DeclareLaunchArgument('playback_sample_rate_hz', default_value=''),
            DeclareLaunchArgument('llm_profile', default_value='reception_default'),
            DeclareLaunchArgument('asr_profile', default_value='qwen3_asr_gpu'),
            DeclareLaunchArgument('tts_profile', default_value='qwen3_tts_gpu'),
            OpaqueFunction(function=_build_runtime),
        ]
    )
