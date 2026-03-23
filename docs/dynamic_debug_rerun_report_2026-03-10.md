# Dynamic Debug Rerun Report 2026-03-10

## Summary

`dynamic_debug_report_2026-03-10.md` で報告した不具合は、この rerun 時点では未解消。

特に重大なのは `purpose-first` シナリオで、clean restart 後の isolated run でも再現した。これは一時的な session 汚染ではなく、現在の semantic / commit policy の不具合が残っていることを示す。

## Scope

再評価対象:

- `reception_dynamic_purpose_first_live.json`
- `reception_dynamic_filler_recovery_live.json`
- `reception_dynamic_affiliation_correction_live.json`

評価方法:

- live stack を起動
- synthetic ASR publisher ベースの `live_stack_e2e.py` で会話を投入
- semantic 評価の安定化のため `playback_enabled:=false`

使用コマンドの要点:

```bash
ros2 launch ros2_reception_orchestrator reception_bringup.launch.py \
  profile_name:=qwen_fullstack \
  enable_mic_input:=false \
  playback_enabled:=false \
  discord_parent_channel_id:=discord:1479886028549525678:1479886034840846499 \
  gpu_memory_utilization:=0.25
```

## Artifacts

Raw outputs:

- `/tmp/reception_dynamic_live_eval.json`
- `/tmp/reception_dynamic_purpose_first_isolated.json`

Scenario files:

- [reception_dynamic_purpose_first_live.json](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/tools/scenarios/reception_dynamic_purpose_first_live.json)
- [reception_dynamic_filler_recovery_live.json](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/tools/scenarios/reception_dynamic_filler_recovery_live.json)
- [reception_dynamic_affiliation_correction_live.json](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/tools/scenarios/reception_dynamic_affiliation_correction_live.json)

## Result

### Overall

- Combined rerun: fail
- Isolated `purpose-first`: fail

### Important caveat

`live_stack_e2e.py` は scenario 間で session reset を強制しないため、combined run の後半 2 シナリオには前シナリオの state 汚染が入っている。

したがって、combined run 全体は「不具合が残っている」証拠にはなるが、厳密な根拠としては `purpose-first` の isolated run を優先して採用する。

## Isolated Scenario A: Purpose-First

対象:

- `学長に会いに来ました。`
- `島中です。`
- `菅谷研究室です。`
- `はい。`

期待:

- `name=島中`
- `affiliation=菅谷研究室`
- `purpose=学長に会いに来ました`
- final phase: `notified_waiting`

実際:

- `name=学長`
- `affiliation=菅谷研究室`
- `purpose=null`
- final phase: `collecting`

### Turn-by-turn failures

1. `学長に会いに来ました。`
   - expected: `ask_name`
   - actual: `ask_affiliation`
   - committed state: `name=学長`

2. `島中です。`
   - expected: `ask_affiliation`
   - actual: `ask_purpose`
   - committed state: `affiliation=島中`

3. `菅谷研究室です。`
   - expected: `confirm`
   - actual: `ask_purpose`

4. `はい。`
   - expected: `notify_waiting`
   - actual: `ask_purpose`

### Conclusion for Scenario A

以下の不具合が残っている。

- purpose-first utterance から meeting target を visitor `name` として commit してしまう
- `島中です。` を `affiliation` に誤 commit してしまう
- `purpose` が最後まで確定せず、`confirm` に進めない
- `はい。` でも `affirm` ではなく `ask_purpose` 継続になる

## Combined Rerun Notes

combined run では 3 シナリオすべて fail だった。

### Scenario B: Filler Recovery

汚染込みの最終 state:

- `name=学長`
- `affiliation=菅谷研究室`
- `purpose=情報工学科`

観測された問題:

- filler の後も state が clean に戻らない
- `情報工学科です。` が `purpose` に commit される
- `書類を届けに来ました。` が correction / overwrite ではなく waiting 遷移側へ吸われる

### Scenario C: Affiliation Correction

combined run でも fail。

観測された問題:

- correction を phase-aware に扱えていない
- `confirming` 中の新情報が `affirm` や waiting 側に誤吸収されうる

## Root Cause Assessment

今回の rerun から、以下は未解消。

- `slot candidate extraction` と `slot commit` の分離は不十分
- phase-aware commit policy が弱く、主項目以外の slot を誤 commit する
- purpose-first utterance の解釈が破綻し、person target を visitor name に倒す
- confirming 中の新情報を correction/overwrite として扱うポリシーが不足

## Secondary Issues

semantic 主問題ではないが、以下は継続している。

- `chat_bridge` sidecar reconnect warning
  - non-blocking
- `playback_device` placeholder misuse による ALSA error
  - semantic 評価では `playback_enabled:=false` で回避可能

## Final Verdict

前回レポートの不具合は、この rerun 時点では未解消。

特に `purpose-first` の isolated run が clean restart 後にも失敗しているため、現状を「改善済み」とは判定できない。
