from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('ros2_reception_orchestrator'),
                        'config',
                        'params.yaml',
                    ]
                ),
                description='Path to reception orchestrator parameter file',
            ),
            Node(
                package='ros2_reception_orchestrator',
                executable='reception_orchestrator',
                name='reception_orchestrator',
                output='screen',
                parameters=[params_file],
            ),
        ]
    )
