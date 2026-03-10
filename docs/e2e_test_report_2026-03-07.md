# Reception Orchestrator E2E Test Report (2026-03-07)

## Summary

- Scope:
  - Fix two blockers found in initial E2E
  - Run scenario-based E2E with real `ros2_vllm` + real `tts_server`
  - Run audio robustness tests with TTS-generated WAV + real `asr_streaming_node`
  - Run one staged real-ASR orchestrator flow
- Result:
  - Core FSM E2E with mock ASR/chat passed across 5 major scenarios.
  - Real Discord bridge bringup, channel send, thread creation, and thread posting are validated with production credentials from `src/ros2_chat/.env`.
  - Real Discord outbound + mock ASR + real vLLM + real TTS orchestrator scenarios passed across happy-path, missing-field, correction, and post-confirm memo cases.
  - Real ASR can drive the orchestrator, but transcription quality is still unstable for short phrases and organization names.

## Code Fixes Applied

1. Orchestrator deadlock / timeout fix
   - File: `ros2_reception_orchestrator/node.py`
   - Change:
     - moved ROS entities into a `ReentrantCallbackGroup`
     - changed `_on_tick()` to acquire the control lock in non-blocking mode
   - Effect:
     - removed the 180-second stall where the first TTS action result never reached the orchestrator

2. Natural affirmative detection
   - File: `ros2_reception_orchestrator/info_extractor.py`
   - Change:
     - accept phrases such as `はい、間違いありません。`, `はい、お願いします。`, `問題ありません。`, `その通りです。`
     - tightened correction detection so `間違いありません` is not treated as a correction signal

3. Better purpose extraction
   - File: `ros2_reception_orchestrator/info_extractor.py`
   - Change:
     - added `で来ました` handling so `山田さんに面会で来ました` is normalized to `山田さんに面会`

4. Safer secretary reply relay
   - File: `ros2_reception_orchestrator/dialog_adapter.py`
   - Change:
     - short human-authored secretary replies are now spoken as-is
     - LLM formatting keeps a fallback to the original text if the rewrite drifts too much

5. Missing-field direct-answer inference
   - Files:
     - `ros2_reception_orchestrator/session_manager.py`
     - `ros2_reception_orchestrator/info_extractor.py`
   - Change:
     - when exactly one required field is still missing, short direct answers such as `東京大学です` can fill that field

6. ros2_chat sidecar runtime fallback
   - File: `ros2_chat/chat_bridge_node.py`
   - Change:
     - if system `node` is unavailable, the bridge now falls back to the bundled runtime in `src/ros2_chat/.tools/node-*/bin/node`
   - Effect:
     - real Discord bridge can start in this workspace without requiring global Node.js on `PATH`

7. ros2_chat launch autostart stabilization
   - File: `ros2_chat/launch/discord_bridge.launch.py`
   - Change:
     - replaced immediate `OnProcessStart -> configure` with a short delayed configure action
   - Effect:
     - `ros2 launch ros2_chat discord_bridge.launch.py` now reaches `active/READY` reliably in this environment

8. No retry prompt on silent ASR result while waiting
   - File: `ros2_reception_orchestrator/session_manager.py`
   - Change:
     - empty ASR results only trigger `もう一度お願いいたします` during `collecting` / `confirming`
     - silent results in `notified_waiting` are ignored
   - Effect:
     - fixed an E2E regression where waiting for the secretary caused unnecessary retry prompts

## Regression Tests

- `src/ros2_reception_orchestrator/test/test_reception_core.py`
- Result: `14 passed`

Added/updated coverage:
- natural affirmative phrases
- no false correction on `間違いありません`
- purpose cleanup for `〜で来ました`
- secretary reply fallback behavior
- missing-field direct-answer inference

## Scenario E2E: Mock ASR + Real vLLM + Real TTS + Mock Chat

Test environment:
- LLM backend: `Qwen/Qwen2.5-1.5B-Instruct`
- TTS backend: `esnya/japanese_speecht5_tts`
- chat bridge: local mock service/publisher
- inactivity reset: accelerated to 5 seconds for scenario testing unless stated otherwise

### Scenario 1: Happy Path

- Goal:
  - `full info -> confirm -> notify -> secretary reply -> reset`
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `はい、間違いありません。`
- Secretary reply:
  - `山田は5分ほどで参ります。ロビーでお待ちください。`
- Result: PASS
- Observed:
  - thread created on first final utterance
  - confirmation spoken
  - confirmed Discord post sent
  - secretary reply relayed
  - session reset after inactivity

### Scenario 2: Missing Field Follow-Up

- Goal:
  - confirm collection across multiple visitor turns
- Input utterances:
  - `OpenAIの田中です。`
  - `山田さんに面会で来ました。`
  - `はい、間違いありません。`
- Secretary reply:
  - `山田は応接室でお待ちしています。ご案内します。`
- Result: PASS
- Observed:
  - after first utterance, TTS asked for the missing purpose
  - second utterance filled purpose and moved to confirmation
  - confirmed post sent and secretary reply spoken

### Scenario 3: Correction During Confirmation

- Goal:
  - overwrite structured state only when explicit correction is given
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `違います。所属はDeepMindです。`
  - `はい。`
- Secretary reply:
  - `DeepMindの担当者がすぐに参ります。ロビーでお待ちください。`
- Result: PASS
- Observed:
  - second utterance changed affiliation from `OpenAI` to `DeepMind`
  - updated Discord post sent before re-confirmation
  - reconfirmation used corrected affiliation

### Scenario 4: Additional Memo After Confirmation

- Goal:
  - verify post-confirm visitor speech is stored as memo unless it is a real correction
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `はい。`
  - `よろしくお願いします。`
- Secretary reply:
  - none
- Result: PASS
- Observed:
  - post-confirm utterance did not overwrite state
  - TTS response was the waiting acknowledgement
  - session reset after timeout without chat reply

### Scenario 5: Duplicate Secretary Reply

- Goal:
  - ensure duplicate incoming Discord messages with the same `message_id` are relayed only once
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `はい。`
- Secretary reply:
  - `重複確認用です。担当者がロビーに向かいます。`
- Extra behavior:
  - same `message_id` published twice
- Result: PASS
- Observed:
  - TTS spoke the secretary reply exactly once
  - duplicate incoming message was ignored

### Scenario 6: No Reply Timeout

- Goal:
  - verify session can reset from `notified_waiting` without any secretary response
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `はい。`
- Secretary reply:
  - none
- Result: PASS
- Observed:
  - confirmed post sent
  - no relay occurred
  - session reset after timeout

## Discord-Inclusive E2E

Environment:
- bridge secrets loaded from `src/ros2_chat/.env`
- bridge runtime: real `ros2_chat` sidecar + real Discord Gateway
- outbound path: real Discord
- inbound reply path: hybrid
  - real bridge topic `/chat_bridge/incoming`
  - secretary reply injected as a human-authored `ChatMessage`
  - reason: no second human/operator account was automated in this session

### Discord Bridge Smoke: Launch + Direct Send

- Goal:
  - confirm launch autostart, Gateway connection, and direct channel send using real credentials
- Result: PASS
- Observed:
  - `ros2 launch ros2_chat discord_bridge.launch.py` reached `active`
  - `/chat_bridge/status` reported `READY`, `gateway_connected=true`
  - `/chat_bridge/send_message` to the configured parent channel returned `success=true`

### Discord Bridge Smoke: Real Thread Creation + Self-Message Ignore

- Goal:
  - confirm `/chat_bridge/create_thread` on the real channel and verify bot-authored messages are not echoed back
- Result: PASS
- Observed:
  - `/chat_bridge/create_thread` returned a real normalized Discord `thread_id`
  - sending a follow-up message to that thread returned `success=true`
  - `/chat_bridge/incoming` produced no event for the bridge's own messages during the observation window

### Discord Scenario 1: Happy Path with Hybrid Secretary Reply

- Goal:
  - `collect -> confirm -> notify -> secretary relay -> reset`
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `はい、間違いありません。`
- Secretary reply:
  - injected to `/chat_bridge/incoming` as:
    - `田中様、山田が3分ほどでロビーに参ります。少々お待ちください。`
- Result: PASS
- Observed:
  - real Discord thread created:
    - `discord:1479886028549525678:1479886034840846499:1479956036889743461`
  - confirmation post and confirmed post were sent to that thread
  - secretary reply was spoken verbatim by TTS
  - session reset after timeout

### Discord Scenario 2: Missing Field Follow-Up

- Goal:
  - verify multi-turn collection with real Discord outbound updates
- Input utterances:
  - `OpenAIの田中です。`
  - `山田さんに面会で来ました。`
  - `はい、間違いありません。`
- Result: PASS
- Observed:
  - first turn created a real Discord thread with partial info
  - assistant asked for the missing purpose
  - second turn triggered a real thread update post
  - third turn sent the confirmed post
  - reset occurred cleanly without extra retry prompts

### Discord Scenario 3: Correction During Confirmation

- Goal:
  - verify explicit correction overwrites structured state and updates the real thread
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `違います。所属はDeepMindです。`
  - `はい。`
- Result: PASS
- Observed:
  - initial thread created with `OpenAI`
  - correction utterance changed affiliation to `DeepMind`
  - corrected confirmation was spoken
  - updated thread post and final confirmed post were both sent to Discord

### Discord Scenario 4: Post-Confirm Memo

- Goal:
  - verify additional memo does not overwrite structured state or cause an unnecessary Discord update
- Input utterances:
  - `OpenAIの田中です。山田さんに面会で来ました。`
  - `はい、間違いありません。`
  - `よろしくお願いします。`
- Result: PASS
- Observed:
  - thread creation and confirmed post succeeded
  - post-confirm utterance triggered only the waiting acknowledgement
  - no additional Discord thread post was emitted for the memo utterance
  - session reset after timeout

## Audio Matrix: Real ASR with TTS-Generated WAV

Test path:
- `tts_server` action with `save_wav=true`
- WAV preprocessing: trim leading silence + gain boost
- `mic_input_node(audio_backend:=wav_file)`
- real `asr_streaming_node` (`vad.threshold:=0.001`)

### Batch A: Mixed English / General Phrases

| Case | Input | Transcript |
|---|---|---|
| `full_info_1` | `OpenAIの田中です。山田さんに面会で来ました。` | `AIの田中です山田さんに面会できました` |
| `full_info_2` | `ABC株式会社の佐藤です。打ち合わせで参りました。` | `株式会社の佐藤です打ち合わせでまいりました` |
| `name_only` | `山田です。` | `山田です` |
| `affiliation_only` | `所属はOpenAIです。` | `ENAIです` |
| `purpose_only` | `学長との面会で参りました。` | `面会で参りました` |
| `correction` | `違います。所属はDeepMindです。` | `ではD.PMMDです` |
| `affirm_ok` | `はい、間違いありません。` | `違いありません。` |
| `affirm_yes` | `はい、お願いします。` | `お願いします` |
| `affirm_true` | `その通りです。` | `おりです` |
| `memo` | `よろしくお願いします。` | `お願いします` |

Observations:
- English organization names degrade heavily.
- `お願いします` is relatively robust.
- short affirmatives other than `お願いします` remain unstable.

### Batch B: Japanese-Heavy Sentences

| Case | Input | Transcript |
|---|---|---|
| `jp_full_1` | `東京大学の山田です。学長との面会で参りました。` | `山田です学長との面会で参りました` |
| `jp_full_2` | `青葉株式会社の田中です。打ち合わせで参りました。` | `貴重の田中です` |
| `jp_full_3` | `市役所の鈴木です。書類の件で来ました。` | `書類の件できました` |
| `jp_affirm` | `お願いします。` | `お願いします` |
| `jp_correction` | `違います。所属は東京大学です。` | `東京大学です` |

Observations:
- Japanese institution names are better than English names, but still often dropped.
- correction replies can preserve the key affiliation token (`東京大学です`).

### Batch C: Short Field-Wise Answers

| Case | Input | Transcript |
|---|---|---|
| `field_name` | `山田です。` | `ごちそう` |
| `field_affiliation` | `所属は東京大学です。` | `東京大学です` |
| `field_purpose` | `学長との面会で参りました。` | `ことの面会で参りました` |
| `field_affirm` | `お願いします。` | `します` |

Observations:
- very short name-only answers are fragile
- affiliation with Japanese proper nouns is usable
- purpose phrases are partially degraded but still contain useful intent words

### Batch D: Short-Phrase Preprocessing Sweep

- Attempt:
  - reduced `trim_sec` to `0.20` for short phrases
- Result:
  - at least one case timed out waiting for ASR finalization
- Conclusion:
  - the current preprocessing is still not robust for very short synthetic utterances

## Staged Real-ASR Orchestrator E2E

Path:
- real `asr_streaming_node`
- `mic_input_node(audio_backend:=wav_file)` for each turn
- real `ros2_reception_orchestrator`
- real `tts_server`
- real `ros2_vllm`
- mock chat bridge

Attempted turn sequence:
1. `東京大学の山田です。学長との面会で参りました。`
2. `お願いします。`
3. `はい、お願いします。` (retry after first short confirm input was not enough)

Result: PARTIAL PASS

Observed behavior:
- Orchestrator reached `confirming` on the first real-ASR turn.
- Extracted content was inaccurate:
  - name: `山田`
  - affiliation: `同郷大学`
  - purpose: `同郷大学の山田です学長との面会`
- First short confirm attempt failed and triggered retry TTS:
  - `恐れ入ります。もう一度お願いいたします。`
- Second confirm attempt (`はい、お願いします。`) succeeded.
- Confirmed post was sent.
- Secretary reply was spoken.
- Session reset after inactivity.

Interpretation:
- Full pipeline viability is confirmed.
- Real-ASR quality is not yet high enough for unattended production use with synthetic-source Japanese speech and current VAD/preprocessing settings.

## Hardware Readiness

### Microphone Device Access

- Command path:
  - `mic_input_node(audio_backend:=alsa_arecord, alsa_device:=plughw:1,0)`
- Result: PASS
- Evidence:
  - `/audio/frames` was received with `sample_rate: 16000`, `channels: 1`, `sample_format: S16LE`

### Local TTS Playback

- Previously validated in this session:
  - `tts_server` playback enabled
  - `playback.sample_rate_hz:=16000`
  - `playback.device:=plughw:1,0`
- Result: PASS

## Remaining Issues

1. Real ASR accuracy is still the main production blocker.
   - short names and English organization names are weak
   - synthetic short confirm phrases are brittle

2. Secretary reply inbound was only hybrid-tested.
   - real Discord outbound is validated
   - human-authored inbound was injected on `/chat_bridge/incoming` instead of being produced by a second real Discord user

3. Replacing the ASR server while the orchestrator is running can strand an in-flight `/asr/listen` goal.
   - observed when stopping one `mock_asr_server` instance and starting another
   - practical impact for production is limited if ASR is started before orchestrator and kept alive, but the client should eventually recover from server restarts

4. Discord REST readback with the current bot permissions returned HTTP 403.
   - outbound verification relied on ROS service success responses, real Gateway health, and orchestrator-side thread/message logs
   - live message fetch/list APIs may require additional Discord permissions beyond current send/create capabilities

5. Some package shutdown paths still throw `rcl_shutdown already called` on `Ctrl-C`.
   - observed in `tts_server` and `ros2_vllm`
   - this did not block functional testing, but should be cleaned up

6. `mock_asr_server` can throw an action shutdown exception if interrupted during result publication.
   - observed once during scenario switching
   - this does not affect the production ASR node, but it is noise in automated test orchestration

## Recommendation Before Final Live Mic / Earphone Acceptance

- Use the current fixed orchestrator build.
- Run with:
  - `ros2 launch ros2_chat discord_bridge.launch.py`
  - real `asr_streaming_node`
  - `mic_input_node(audio_backend:=alsa_arecord, alsa_device:=plughw:1,0)`
  - `tts_server` with playback enabled on the operator's earphone device
  - `ros2_vllm` on `Qwen/Qwen2.5-1.5B-Instruct`
- Expectation:
  - pipeline should function
  - recognition quality must still be judged by a human speaker in a live environment
