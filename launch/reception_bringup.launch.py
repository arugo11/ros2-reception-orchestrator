import os
from pathlib import Path
import signal
import subprocess
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_LOCK_PATH = Path('/tmp/ros2_reception_bringup.lock')


def _default_model_catalog_path() -> str:
    workspace_candidate = Path.cwd() / 'config' / 'model_profiles.yaml'
    if workspace_candidate.is_file():
        return str(workspace_candidate)
    return ''


def _load_dotenv_value(name: str) -> str:
    direct = os.environ.get(name, '').strip()
    if direct:
        return direct
    candidates = [
        Path.cwd() / '.env',
        Path.cwd() / 'src' / 'ros2_chat' / '.env',
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ''


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


def _terminate_existing_stack_processes(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    del context, args, kwargs
    completed = subprocess.run(
        ['ps', '-eo', 'pid=,args='],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []

    current_pid = os.getpid()
    workspace_root = str(Path.cwd())
    command_fragments = (
        'reception_orchestrator',
        'semantic_extractor_server',
        'response_planner_server',
        'asr_streaming_node',
        'llm_chat_node',
        'vllm_server_node',
        'tts_server',
        'chat_bridge_node',
        'mic_input_node',
        'visitor_detection_node',
        'ros2_reception_orchestrator.visitor_detection',
        'camera_image_publisher',
        'vllm serve',
    )
    candidate_pids: list[int] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        command = parts[1]
        if workspace_root not in command and 'vllm serve' not in command:
            continue
        if not any(fragment in command for fragment in command_fragments):
            continue
        candidate_pids.append(pid)

    if not candidate_pids:
        return []

    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    time.sleep(1.0)
    for pid in candidate_pids:
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    return []


def generate_launch_description() -> LaunchDescription:
    reception_params_default = PathJoinSubstitution(
        [FindPackageShare('ros2_reception_orchestrator'), 'config', 'params.yaml']
    )
    discord_params_default = PathJoinSubstitution(
        [FindPackageShare('ros2_chat'), 'config', 'params.yaml']
    )
    vllm_params_default = PathJoinSubstitution(
        [FindPackageShare('ros2_vllm'), 'config', 'params.yaml']
    )
    asr_config_default = PathJoinSubstitution(
        [FindPackageShare('asr_streaming_node'), 'config', 'config.yaml']
    )
    tts_config_default = PathJoinSubstitution(
        [FindPackageShare('tts_server'), 'config', 'tts_server.yaml']
    )

    model_catalog_file = LaunchConfiguration('model_catalog_file')
    profile_name = LaunchConfiguration('profile_name')
    asr_profile = LaunchConfiguration('asr_profile')
    llm_profile = LaunchConfiguration('llm_profile')
    tts_profile = LaunchConfiguration('tts_profile')
    llm_provider = LaunchConfiguration('llm_provider')
    llm_api_base_url = LaunchConfiguration('llm_api_base_url')
    enable_mic_input = LaunchConfiguration('enable_mic_input')
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
    enable_demo_gui = LaunchConfiguration('enable_demo_gui')
    demo_gui_host = LaunchConfiguration('demo_gui_host')
    demo_gui_port = LaunchConfiguration('demo_gui_port')
    enable_visitor_detection = LaunchConfiguration('enable_visitor_detection')
    enable_camera_publisher = LaunchConfiguration('enable_camera_publisher')
    visitor_detector_model_path = LaunchConfiguration('visitor_detector_model_path')
    camera_device = LaunchConfiguration('camera_device')
    camera_image_topic = LaunchConfiguration('camera_image_topic')
    camera_width = LaunchConfiguration('camera_width')
    camera_height = LaunchConfiguration('camera_height')
    camera_fps = LaunchConfiguration('camera_fps')

    resolved_asr_profile = PythonExpression([
        "'", asr_profile, "' if '", asr_profile, "' else ('", profile_name, "' if '", profile_name, "' else 'qwen3_asr_0_6b_cpu')"
    ])
    resolved_llm_profile = PythonExpression([
        "'", llm_profile, "' if '", llm_profile, "' else ('", profile_name, "' if '", profile_name, "' else 'qwen35_4b_text')"
    ])
    resolved_tts_profile = PythonExpression([
        "'", tts_profile, "' if '", tts_profile, "' else ('", profile_name, "' if '", profile_name, "' else 'qwen3_tts_gpu')"
    ])

    mic_input = GroupAction(
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
    )

    asr_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('asr_streaming_node'), 'launch', 'asr_streaming_node.launch.py']
            )
        ),
        launch_arguments={
            'config_file': asr_config_file,
            'model_catalog_file': model_catalog_file,
            'profile_name': resolved_asr_profile,
            'continuous_enabled': 'true',
        }.items(),
    )

    llm_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros2_vllm'), 'launch', 'vllm_bringup.launch.py']
            )
        ),
        launch_arguments={
            'params_file': vllm_params_file,
            'llm_provider': llm_provider,
            'api_base_url': llm_api_base_url,
            'model_catalog_file': model_catalog_file,
            'profile_name': resolved_llm_profile,
            'endpoint_prefix': 'llm',
            'vllm_port': '8000',
            'wandb_enabled': 'false',
            'reuse_existing_backend': 'false',
            'replace_existing_backend': 'true',
            'server_node_name': 'vllm_server_node',
            'chat_node_name': 'llm_chat_node',
        }.items(),
    )

    tts_stack = Node(
        package='tts_server',
        executable='tts_server',
        name='tts_server',
        output='screen',
        parameters=[
            tts_config_file,
            {
                'model_catalog_file': model_catalog_file,
                'profile_name': resolved_tts_profile,
                'playback.enabled': ParameterValue(playback_enabled, value_type=bool),
                'playback.device': playback_device,
                'playback.sample_rate_hz': ParameterValue(
                    playback_sample_rate_hz, value_type=int
                ),
            },
        ],
    )

    def _build_visitor_detection_action(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        del args, kwargs
        if context.perform_substitution(enable_visitor_detection).strip().lower() not in {'1', 'true', 'yes', 'on'}:
            return []
        detector_model_path = context.perform_substitution(visitor_detector_model_path).strip()
        if not detector_model_path:
            for candidate in (
                '/tmp/visitor_models/face_detection_yunet_2023mar.onnx',
                str(Path.cwd() / 'models' / 'face_detection_yunet_2023mar.onnx'),
            ):
                if Path(candidate).is_file():
                    detector_model_path = candidate
                    break
        reception_params_file_value = context.perform_substitution(reception_params_file)
        command: list[str] = [
            str(Path.cwd() / '.venv' / 'bin' / 'python'),
            '-m',
            'ros2_reception_orchestrator.visitor_detection',
            '--ros-args',
            '--params-file',
            reception_params_file_value,
            '-p',
            'detector_backend:=opencv_haar_upperbody',
        ]
        if detector_model_path:
            command.extend(['-p', f'detector_model_path:={detector_model_path}'])
        return [
            ExecuteProcess(
                cmd=command,
                name='visitor_detection_node',
                output='screen',
            )
        ]

    def _build_camera_publisher_action(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        del args, kwargs
        if context.perform_substitution(enable_visitor_detection).strip().lower() not in {'1', 'true', 'yes', 'on'}:
            return []
        if context.perform_substitution(enable_camera_publisher).strip().lower() not in {'1', 'true', 'yes', 'on'}:
            return []
        return [
            Node(
                package='ros2_reception_orchestrator',
                executable='camera_image_publisher',
                name='camera_image_publisher',
                output='screen',
                parameters=[
                    {
                        'camera_device': context.perform_substitution(camera_device).strip() or '/dev/video0',
                        'image_topic': context.perform_substitution(camera_image_topic).strip() or '/camera/image_raw',
                        'width': int(context.perform_substitution(camera_width).strip() or '640'),
                        'height': int(context.perform_substitution(camera_height).strip() or '480'),
                        'fps': float(context.perform_substitution(camera_fps).strip() or '30.0'),
                    }
                ],
            )
        ]

    discord_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros2_chat'), 'launch', 'discord_bridge.launch.py']
            )
        ),
        launch_arguments={'params_file': discord_params_file}.items(),
    )

    reception_orchestrator = Node(
        package='ros2_reception_orchestrator',
        executable='reception_orchestrator',
        name='reception_orchestrator',
        output='screen',
        parameters=[
            reception_params_file,
            {
                'discord.parent_channel_id': ParameterValue(
                    discord_parent_channel_id, value_type=str
                ),
                'session.inactivity_reset_sec': ParameterValue(
                    session_inactivity_reset_sec, value_type=int
                ),
            },
        ],
    )
    semantic_extractor = Node(
        package='ros2_reception_orchestrator',
        executable='semantic_extractor_server',
        name='semantic_extractor_server',
        output='screen',
        parameters=[reception_params_file],
    )

    response_planner = Node(
        package='ros2_reception_orchestrator',
        executable='response_planner_server',
        name='response_planner_server',
        output='screen',
        parameters=[reception_params_file],
    )

    demo_dashboard = Node(
        condition=IfCondition(enable_demo_gui),
        package='ros2_reception_orchestrator',
        executable='reception_demo_dashboard',
        name='reception_demo_dashboard',
        output='screen',
        parameters=[
            {
                'host': demo_gui_host,
                'port': ParameterValue(demo_gui_port, value_type=int),
                'profile_name': profile_name,
                'asr_profile': resolved_asr_profile,
                'llm_profile': resolved_llm_profile,
                'tts_profile': resolved_tts_profile,
                'llm_provider': llm_provider,
                'audio_backend': audio_backend,
                'alsa_device': alsa_device,
                'playback_enabled': ParameterValue(playback_enabled, value_type=bool),
                'playback_device': playback_device,
                'tts.action_name': '/tts/speak',
                'session.state_topic': '/reception/session_state',
                'execution.event_topic': '/reception/events',
                'conversation.trace_topic': '/reception/conversation_trace',
                'llm.status_topic': '/llm/status',
                'chat.status_topic': '/chat_bridge/status',
                'visitor_detection.state_topic': '/visitor_detection/state',
                'visitor_detection.event_topic': '/visitor_detection/events',
            },
        ],
    )

    return LaunchDescription(
        [
            OpaqueFunction(function=_terminate_existing_stack_processes),
            OpaqueFunction(function=_acquire_singleton_lock),
            RegisterEventHandler(
                OnShutdown(on_shutdown=[OpaqueFunction(function=_release_singleton_lock)])
            ),
            DeclareLaunchArgument(
                'model_catalog_file',
                default_value=_default_model_catalog_path(),
                description='Shared model catalog YAML path.',
            ),
            DeclareLaunchArgument(
                'profile_name',
                default_value='',
                description='Compatibility profile applied to all components when asr_profile/llm_profile/tts_profile are not set.',
            ),
            DeclareLaunchArgument(
                'asr_profile',
                default_value='',
                description='ASR model profile name. If empty, falls back to profile_name and then the package default.',
            ),
            DeclareLaunchArgument(
                'llm_profile',
                default_value='',
                description='LLM model profile name. If empty, falls back to profile_name and then the package default.',
            ),
            DeclareLaunchArgument(
                'tts_profile',
                default_value='',
                description='TTS model profile name. If empty, falls back to profile_name and then the package default.',
            ),
            DeclareLaunchArgument(
                'llm_provider',
                default_value='vllm',
                description='LLM provider to use for /llm/chat, e.g. vllm or gemini.',
            ),
            DeclareLaunchArgument(
                'llm_api_base_url',
                default_value='',
                description='OpenAI-compatible API base URL for direct LLM providers such as Gemini.',
            ),
            DeclareLaunchArgument(
                'reception_params_file',
                default_value=reception_params_default,
                description='Path to reception orchestrator parameter file.',
            ),
            DeclareLaunchArgument(
                'discord_params_file',
                default_value=discord_params_default,
                description='Path to ros2_chat parameter file.',
            ),
            DeclareLaunchArgument(
                'vllm_params_file',
                default_value=vllm_params_default,
                description='Path to ros2_vllm parameter file.',
            ),
            DeclareLaunchArgument(
                'asr_config_file',
                default_value=asr_config_default,
                description='Path to ASR runtime config YAML.',
            ),
            DeclareLaunchArgument(
                'tts_config_file',
                default_value=tts_config_default,
                description='Path to TTS runtime config YAML.',
            ),
            DeclareLaunchArgument(
                'discord_parent_channel_id',
                default_value=_load_dotenv_value('DISCORD_CHANNEL_ID'),
                description='Optional Discord parent channel override.',
            ),
            DeclareLaunchArgument(
                'session_inactivity_reset_sec',
                default_value='30',
                description='Seconds before the reception session auto resets.',
            ),
            DeclareLaunchArgument(
                'enable_mic_input',
                default_value='true',
                description='Launch the microphone input node.',
            ),
            DeclareLaunchArgument(
                'audio_backend',
                default_value='auto',
                description='Mic input backend: auto, wav_file, pyaudio, alsa_arecord, pulse_parec.',
            ),
            DeclareLaunchArgument(
                'wav_file_path',
                default_value='',
                description='Optional WAV file path when audio_backend=wav_file.',
            ),
            DeclareLaunchArgument(
                'alsa_device',
                default_value='default',
                description='ALSA capture device when audio_backend=alsa_arecord.',
            ),
            DeclareLaunchArgument(
                'pulse_source',
                default_value='RDPSource',
                description='PulseAudio source name when audio_backend=pulse_parec.',
            ),
            DeclareLaunchArgument(
                'pulse_server',
                default_value='unix:/mnt/wslg/PulseServer',
                description='PulseAudio server path when audio_backend=pulse_parec.',
            ),
            DeclareLaunchArgument(
                'playback_enabled',
                default_value='false',
                description='Enable local TTS playback on the host machine.',
            ),
            DeclareLaunchArgument(
                'playback_device',
                default_value='',
                description='Optional playback device override for the TTS server.',
            ),
            DeclareLaunchArgument(
                'playback_sample_rate_hz',
                default_value='24000',
                description='Playback sample rate for the TTS server.',
            ),
            DeclareLaunchArgument(
                'enable_visitor_detection',
                default_value='true',
                description='Launch the visitor detection node.',
            ),
            DeclareLaunchArgument(
                'enable_camera_publisher',
                default_value='true',
                description='Launch a simple OpenCV camera publisher.',
            ),
            DeclareLaunchArgument(
                'visitor_detector_model_path',
                default_value='',
                description='Path to the face detector ONNX model.',
            ),
            DeclareLaunchArgument(
                'camera_device',
                default_value='/dev/video0',
                description='Camera device path for the built-in image publisher.',
            ),
            DeclareLaunchArgument(
                'camera_image_topic',
                default_value='/camera/image_raw',
                description='Image topic for the built-in camera publisher.',
            ),
            DeclareLaunchArgument(
                'camera_width',
                default_value='640',
                description='Camera width for the built-in camera publisher.',
            ),
            DeclareLaunchArgument(
                'camera_height',
                default_value='480',
                description='Camera height for the built-in camera publisher.',
            ),
            DeclareLaunchArgument(
                'camera_fps',
                default_value='30.0',
                description='Camera FPS for the built-in camera publisher.',
            ),
            DeclareLaunchArgument(
                'enable_demo_gui',
                default_value='false',
                description='Launch the localhost reception demo dashboard.',
            ),
            DeclareLaunchArgument(
                'demo_gui_host',
                default_value='127.0.0.1',
                description='Host interface for the localhost reception demo dashboard.',
            ),
            DeclareLaunchArgument(
                'demo_gui_port',
                default_value='8090',
                description='Port for the localhost reception demo dashboard.',
            ),
            mic_input,
            asr_stack,
            llm_stack,
            tts_stack,
            discord_bridge,
            OpaqueFunction(function=_build_camera_publisher_action),
            OpaqueFunction(function=_build_visitor_detection_action),
            semantic_extractor,
            response_planner,
            reception_orchestrator,
            demo_dashboard,
        ]
    )
