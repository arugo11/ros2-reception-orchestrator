# Dynamic Debug Report 2026-03-10

## Summary

Synthetic ASR を別プロセスから publish する形で、既存 test suite に入っていない会話シナリオを live stack に対して実行した。

このターンでは AGENTS に `ros2-ai-native-debugging` skill が列挙されていなかったため、既存の local skill directory にある publisher script を fallback として使用した。

- publisher:
  [`publish_utterance.py`](/workspaces/ros2-workspace-template/.codex/skills/ros2-ai-native-debugging/scripts/publish_utterance.py)
- stack:
  `reception_bringup.launch.py`
- profile:
  `qwen_fullstack`
- runtime:
  `enable_mic_input:=false`
  `playback_enabled:=false`

評価結果は、既存 happy-path より難しいシナリオではまだ不安定で、特に `purpose` の誤補完と訂正 handling に大きな弱さがある、というものだった。

## Method

起動コマンド:

```bash
source /opt/ros/jazzy/setup.bash
source /workspaces/ros2-workspace-template/install/setup.bash
source /workspaces/ros2-workspace-template/src/ros2_tts/install/setup.bash

ros2 launch ros2_reception_orchestrator reception_bringup.launch.py \
  profile_name:=qwen_fullstack \
  enable_mic_input:=false \
  playback_enabled:=false \
  discord_parent_channel_id:=discord:1479886028549525678:1479886034840846499 \
  gpu_memory_utilization:=0.25
```

utterance publish は以下で行った:

```bash
python3 /workspaces/ros2-workspace-template/.codex/skills/ros2-ai-native-debugging/scripts/publish_utterance.py --text "..."
```

観測対象:

- `reception_orchestrator` log
- `Pipeline: semantic`
- `Pipeline: state_after_extraction`
- `TTS speak`

## Existing Coverage Excluded

既存 test / scenario は主に以下をカバーしている。

- happy path
- overlap
- correction
- waiting question
- live multislot

今回の追加評価では、そこに無い以下の会話パターンを優先した。

1. 用件先行の順不同入力
2. 雑談 / つなぎ発話からの回復
3. 所属訂正

## Scenario 1: Purpose First, Then Name, Then Affiliation

### Input

1. `学長に会いに来ました。`
2. `島中です。`
3. `菅谷研究室です。`
4. `はい。`

### Observed Behavior

初手の `学長に会いに来ました。` で、system は `purpose` を first-class に扱うのではなく、誤って以下を生成した。

- `name=学長`
- `affiliation=大学`
- `purpose=面会`

その後 reducer は `affiliation` と `purpose` を reject したが、`name=学長` が残った。

結果として conversation は以下へ崩れた。

- `学長さん、ご所属を教えていただけますか。`
- `島中です。` に対して `affiliation=島中`
- `菅谷研究室です。` に対して state は
  - `name=学長`
  - `affiliation=島中`
  - `purpose` missing
- `はい。` に対して `purpose=目的が不明`

最終 confirm:

```text
お名前は学長、ご所属は島中、ご用件は目的が不明でお間違いないでしょうか。
```

### Assessment

- 結果: `fail`
- 主因:
  - purpose-first utterance を `name` に誤投影
  - 後続 turn で誤 state が伝播
  - `はい` が missing-purpose を埋める trigger として誤利用された

## Scenario 2: Filler / Polite Chit-Chat Then Standard Answers

### Input

1. `えっと、よろしくお願いします。`
2. `田中です。`
3. `情報工学科です。`
4. `書類を届けに来ました。`
5. `はい。`

### Observed Behavior

初手の filler utterance で本来は slot 更新なしになるべきだが、以下が起きた。

- primary hallucination:
  - `未知の先生`
  - `未知の研究所`
  - `来訪理由不明`
- reject 後の unusable rescue で
  - `purpose=よろしくお願いします`
  が回収されてしまった

その後:

- `田中です。` で `name=田中`
- `情報工学科です。` で `affiliation=情報工学科`
- `書類を届けに来ました。` は confirm turn の中で扱われたが、
  final `purpose` は更新されず、古い `よろしくお願いします` が残った
- `はい。` で notify_waiting へ進んだ

最終 notify 前 confirm:

```text
お名前は田中、ご所属は情報工学科、ご用件はよろしくお願いしますでお間違いないでしょうか。
```

### Assessment

- 結果: `fail`
- 主因:
  - filler utterance を `ignore_input` とせず、purpose へ誤 commit
  - confirm phase 中の新しい purpose utterance を correction / overwrite として扱えない

## Scenario 3: Affiliation Correction

### Input

1. `山田です。`
2. `総務部です。`
3. `違います。広報部です。`
4. `書類提出です。`
5. `はい。`

### Observed Behavior

ここでは初期収集は比較的よく動いた。

- `山田です。` -> `name=山田`
- `総務部です。` -> `affiliation=総務部`
- `purpose=面会` hallucination は reject された

ただし訂正 turn `違います。広報部です。` で崩れた。

最終的に confirm 文は以下になった。

```text
お名前は山田、ご所属は総務部、ご用件は広報でお間違いないでしょうか。
```

つまり `広報部` という affiliation correction を、system は `purpose=広報` と誤解した。

さらに、その後の `書類提出です。` でも final `purpose` は `書類提出` に更新されず、notify waiting へ進む時点で still `purpose=広報` だった。

### Assessment

- 結果: `fail`
- 主因:
  - `違います。Xです。` の correction scope が affiliation に結び付かない
  - confirm 前に入った新 purpose utterance を final overwrite に使えていない

## Cross-Scenario Findings

### What Worked

- `name` 単独回答の基本回収は以前より安定している
- `affiliation` 単独回答で無条件に `purpose` を入れる頻度は下がっている
- ask 系 TTS fallback は user-facing 文としては比較的自然

### What Failed Repeatedly

- `purpose` の誤補完
  - filler
  - 所属回答
  - confirm 中の相槌
- correction scope の誤解
  - `違います。広報部です。` を affiliation 更新ではなく purpose へ投影
- turn role の誤解
  - `書類を届けに来ました。` のような新情報を confirm 中の overwrite として扱えない
- initial non-informational utterance の取り扱い
  - polite filler を `ignore_input` にできない

## Design Implications

この結果から、現時点で妥当な方針は次の通り。

- system が聞くのは 1 turn 1 slot を維持する
- ただし user が複数 slot を自発的に言った場合だけ multi-slot commit を許す
- `purpose` は最も保守的に commit する
- `ignore_input` 判定をもっと強くする必要がある
- correction は `target slot` を現在の phase と last_dialog_act からもっと強く束縛する必要がある
- confirm phase での新情報 utterance は
  - affirmation
  - correction
  - fresh overwrite
  の区別を強化する必要がある

## Recommended Next Fixes

1. `ignore_input` を強化し、filler / あいさつ / polite noise を purpose に入れない
2. `purpose` commit の閾値をさらに厳しくする
3. `違います。Xです。` の correction target を current asked slot により強く結び付ける
4. `confirming` phase での新情報 utterance を correction candidate として再解釈できるようにする

## Overall Verdict

dynamic debug の方法自体は有効だった。live stack に対して synthetic ASR を順次 publish することで、既存 scripted tests では見えにくい failure mode を再現できた。

一方で現時点の conversational robustness は不十分で、特に以下は未達である。

- purpose-first utterance の吸収
- filler utterance の無害化
- affiliation correction の正確な反映

このため、現状は `happy-path はある程度成立するが、順不同・訂正・雑談混じりにはまだ弱い` と評価する。
