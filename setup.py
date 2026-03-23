from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ros2_reception_orchestrator'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['*.py']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', [path for path in glob('config/*') if os.path.isfile(path)]),
        (
            'share/' + package_name + '/config/model_profiles/llm',
            glob('config/model_profiles/llm/*.yaml'),
        ),
        (
            'share/' + package_name + '/config/model_profiles/asr',
            glob('config/model_profiles/asr/*.yaml'),
        ),
        (
            'share/' + package_name + '/config/model_profiles/tts',
            glob('config/model_profiles/tts/*.yaml'),
        ),
    ],
    install_requires=['setuptools', 'PyYAML>=6.0.1'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description='ROS 2 reception orchestrator package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reception_orchestrator = ros2_reception_orchestrator.node_v2:main',
            'semantic_extractor_server = ros2_reception_orchestrator.semantic_extractor_server:main',
            'response_planner_server = ros2_reception_orchestrator.response_planner_server:main',
            'reception_demo_dashboard = ros2_reception_orchestrator.demo_dashboard:main',
        ],
    },
)
