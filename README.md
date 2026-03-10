# ros2_reception_orchestrator

ROS 2 package for coordinating a voice-based reception flow across ASR, dual LLM backends, TTS, and Discord notifications.

## Package Scope

This repository owns the reception orchestration package itself.
It depends on other ROS 2 packages that provide:

- ASR interfaces and streaming nodes
- Discord bridge integration
- vLLM chat backends
- TTS services and actions

## Repository Layout

- `ros2_reception_orchestrator/`: Python package implementation
- `launch/`: package launch files
- `config/`: runtime parameters
- `test/`: package tests
- `docs/`: design and validation notes

## Runtime Notes

The bringup launch reads optional environment variables for isolated Qwen runtime dependencies:

- `QWEN_TTS_PYTHONPATH`
- `QWEN_RUNTIME_PYTHONPATH`
- `QWEN_SOX_BIN_DIR`
- `QWEN_SOX_LIB_DIR`

## Build

From a ROS 2 workspace containing this package and its dependent packages:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select ros2_reception_orchestrator
```