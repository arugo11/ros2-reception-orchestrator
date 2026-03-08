import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

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


def generate_launch_description() -> LaunchDescription:
    reception_params_file = LaunchConfiguration('reception_params_file')
    discord_params_file = LaunchConfiguration('discord_params_file')
    vllm_params_file = LaunchConfiguration('vllm_params_file')
    asr_config_file = LaunchConfiguration('asr_config_file')
    tts_config_file = LaunchConfiguration('tts_config_file')
    discord_parent_channel_id = LaunchConfiguration('discord_parent_channel_id')
    session_inactivity_reset_sec = LaunchConfiguration('session_inactivity_reset_sec')
    audio_backend = LaunchConfiguration('audio_backend')
    wav_file_path = LaunchConfiguration('wav_file_path')
    alsa_device = LaunchConfiguration('alsa_device')
    pulse_source = LaunchConfiguration('pulse_source')
    pulse_server = LaunchConfiguration('pulse_server')
    playback_enabled = LaunchConfiguration('playback_enabled')
    playback_device = LaunchConfiguration('playback_device')
    playback_sample_rate_hz = LaunchConfiguration('playback_sample_rate_hz')
    supervisor_model_name = LaunchConfiguration('supervisor_model_name')
    dialog_model_name = LaunchConfiguration('dialog_model_name')
    asr_runtime_device = LaunchConfiguration('asr_runtime_device')
    tts_runtime_device = LaunchConfiguration('tts_runtime_device')

    return LaunchDescription(
        [
            OpaqueFunction(function=_acquire_singleton_lock),
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[OpaqueFunction(function=_release_singleton_lock)]
                )
            ),
            DeclareLaunchArgument(
                'reception_params_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('ros2_reception_orchestrator'),
                        'config',
                        'params.yaml',
                    ]
                ),
                description='Path to reception orchestrator parameter file',
            ),
            DeclareLaunchArgument(
                'discord_params_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('ros2_chat'),
                        'config',
                        'params.yaml',
                    ]
                ),
                description='Path to ros2_chat parameter file',
            ),
            DeclareLaunchArgument(
                'vllm_params_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('ros2_vllm'),
                        'config',
                        'params.yaml',
                    ]
                ),
                description='Path to ros2_vllm parameter file',
            ),
            DeclareLaunchArgument(
                'asr_config_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('asr_streaming_node'),
                        'config',
                        'config.yaml',
                    ]
                ),
                description='Path to ASR config YAML',
            ),
            DeclareLaunchArgument(
                'tts_config_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('tts_server'),
                        'config',
                        'tts_server.yaml',
                    ]
                ),
                description='Path to TTS config YAML',
            ),
            DeclareLaunchArgument(
                'discord_parent_channel_id',
                default_value='',
                description='Discord parent channel target for reception threads',
            ),
            DeclareLaunchArgument(
                'session_inactivity_reset_sec',
                default_value='30',
                description='Idle timeout before resetting the reception session',
            ),
            DeclareLaunchArgument(
                'audio_backend',
                default_value='auto',
                description='Mic input backend',
            ),
            DeclareLaunchArgument(
                'wav_file_path',
                default_value='',
                description='WAV file path used when audio_backend=wav_file',
            ),
            DeclareLaunchArgument(
                'alsa_device',
                default_value='default',
                description='ALSA device name when audio_backend=alsa_arecord',
            ),
            DeclareLaunchArgument(
                'pulse_source',
                default_value='RDPSource',
                description='PulseAudio source name when audio_backend=pulse_parec',
            ),
            DeclareLaunchArgument(
                'pulse_server',
                default_value='unix:/mnt/wslg/PulseServer',
                description='PulseAudio server path when audio_backend=pulse_parec',
            ),
            DeclareLaunchArgument(
                'playback_enabled',
                default_value='false',
                description='Enable local TTS playback',
            ),
            DeclareLaunchArgument(
                'playback_device',
                default_value='',
                description='Playback device passed to tts_server',
            ),
            DeclareLaunchArgument(
                'playback_sample_rate_hz',
                default_value='16000',
                description='Playback sample rate override passed to tts_server',
            ),
            DeclareLaunchArgument(
                'supervisor_model_name',
                default_value='Qwen/Qwen3.5-0.8B',
                description='Supervisor LLM model name',
            ),
            DeclareLaunchArgument(
                'dialog_model_name',
                default_value='Qwen/Qwen2.5-0.5B-Instruct',
                description='Dialog LLM model name',
            ),
            DeclareLaunchArgument(
                'asr_runtime_device',
                default_value='cuda:0',
                description='ASR runtime device override.',
            ),
            DeclareLaunchArgument(
                'tts_runtime_device',
                default_value='cuda:0',
                description='TTS runtime device override.',
            ),
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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare('asr_streaming_node'),
                            'launch',
                            'asr_streaming_node.launch.py',
                        ]
                    )
                ),
                launch_arguments={
                    'config_file': asr_config_file,
                    'runtime_device': asr_runtime_device,
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
                    'model_name': supervisor_model_name,
                    'endpoint_prefix': 'supervisor_llm',
                    'vllm_port': '8001',
                    'max_model_len': '512',
                    'gpu_memory_utilization': '0.28',
                    'wandb_enabled': 'false',
                    'reuse_existing_backend': 'false',
                    'replace_existing_backend': 'true',
                    'server_node_name': 'supervisor_vllm_server_node',
                    'chat_node_name': 'supervisor_llm_chat_node',
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
                    'model_name': dialog_model_name,
                    'endpoint_prefix': 'dialog_llm',
                    'vllm_port': '8002',
                    'max_model_len': '256',
                    'gpu_memory_utilization': '0.22',
                    'wandb_enabled': 'false',
                    'reuse_existing_backend': 'false',
                    'replace_existing_backend': 'true',
                    'server_node_name': 'dialog_vllm_server_node',
                    'chat_node_name': 'dialog_llm_chat_node',
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
                        'runtime.device': tts_runtime_device,
                        'playback.enabled': ParameterValue(playback_enabled, value_type=bool),
                        'playback.device': playback_device,
                        'playback.sample_rate_hz': ParameterValue(
                            playback_sample_rate_hz, value_type=int
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
    )
