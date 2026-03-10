# Qwen3.5 Profile Bench 2026-03-10

Profiles tested against the scripted reception scenarios:

- `qwen35_2b_gguf_q5km`
- `qwen35_0_8b`
- `reception_default`

Result summary:

- `qwen35_2b_gguf_q5km`
  - Startup failed in `vLLM`
  - Root cause: `RuntimeError: Unknown gguf model_type: qwen3_5`
- `qwen35_0_8b`
  - Startup succeeded
  - Failed all scripted scenarios
  - No valid slot extraction or TTS event coverage in the benchmark
- `reception_default`
  - Startup succeeded
  - Semantically closer than `qwen35_0_8b`
  - Still failed all scripted scenarios due incorrect name/affiliation extraction and incomplete TTS coverage

Artifacts:

- JSON report: [qwen35_profile_bench_2026-03-10.json](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/docs/benchmarks/qwen35_profile_bench_2026-03-10.json)
