# Reception Model Selection Report (2026-03-10)

## Executive Summary

現段階で最も妥当で、かつ実験上の根拠が最も強い受付スタック構成は次です。

- LLM: `reception_default`
  - 実体: `Qwen/Qwen2.5-1.5B-Instruct`
- ASR: `qwen3_asr_gpu`
  - 実体: `Qwen/Qwen3-ASR-0.6B`
- TTS: `qwen3_tts_gpu`
  - 実体: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

この組み合わせは、少なくとも scripted benchmark の 3 シナリオで最終的に pass している唯一の構成である。

- `happy_path`: pass
- `correction`: pass
- `overlap`: pass

根拠 artifact:

- [/tmp/reception_default_qwen3tts_all_v2.json](/tmp/reception_default_qwen3tts_all_v2.json)
- [qwen35_profile_bench_2026-03-10.md](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/docs/benchmarks/qwen35_profile_bench_2026-03-10.md)
- [e2e_test_report_2026-03-07.md](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/docs/e2e_test_report_2026-03-07.md)

## Previous Drift

このレポート作成時点では、repo に過去の drift が残っていた。

当時、特に以下は best-known configuration ではなかった。

- [reception_bringup.launch.py](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/launch/reception_bringup.launch.py)
  - default: `profile_name='baseline_whisper_mms'`
- [vllm_bringup.launch.py](/workspaces/ros2-workspace-template/src/ros2_vllm/launch/vllm_bringup.launch.py)
  - default: `profile_name='baseline_whisper_mms'`
- [llm_only.launch.py](/workspaces/ros2-workspace-template/src/ros2_vllm/launch/llm_only.launch.py)
  - default: `profile_name='baseline_whisper_mms'`
- [asr_streaming_node.launch.py](/workspaces/ros2-workspace-template/src/ros2_asr/asr_streaming_node/launch/asr_streaming_node.launch.py)
  - default: `profile_name='baseline_whisper_mms'`
- [tts_server.yaml](/workspaces/ros2-workspace-template/src/ros2_tts/tts_server/config/tts_server.yaml)
  - default: `profile_name='baseline_whisper_mms'`
- [config.yaml](/workspaces/ros2-workspace-template/src/ros2_asr/asr_streaming_node/config/config.yaml)
  - default: `catalog.profile_name: "baseline_whisper_mms"`

そして、その `baseline_whisper_mms` の中身は [model_profiles.yaml](/workspaces/ros2-workspace-template/config/model_profiles.yaml) で次になっている。

- LLM: `microsoft/Phi-4-mini-instruct`
- ASR: `Systran/faster-whisper-small`
- TTS: `esnya/japanese_speecht5_tts`

これは「当時の最適構成」ではなかった。少なくとも、この組み合わせを best と判断した benchmark / E2E の証跡は本 workspace には残っていなかった。

## Recommended Configuration

### Best-Known Runtime

検証済みの最適構成は、分離 profile の次の 3 つである。

#### LLM

File:
- [reception_default.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/reception_default.yaml)

Key settings:
- `repo_id: Qwen/Qwen2.5-1.5B-Instruct`
- `runner: vllm`
- `device: cuda:0`
- `dtype: bfloat16`
- `context_len: 1024`
- `gpu_memory_utilization: 0.35`
- `trust_remote_code: true`
- `startup_args: --enforce-eager --generation-config vllm --max-num-seqs 1`

Reason:
- 16GB 単 GPU 機で安定して起動
- 日本語の slot extraction と短い受付対話のバランスが最も良かった
- `Qwen3.5-0.8B` より semantic quality が良い
- `Qwen3.5-2B GGUF` より実運用性が高い

#### ASR

File:
- [qwen3_asr_gpu.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/asr/qwen3_asr_gpu.yaml)

Key settings:
- `repo_id: Qwen/Qwen3-ASR-0.6B`
- `backend: qwen_asr`
- `device: cuda:0`
- `dtype: auto`
- `memory_budget_mb: 4096`
- `require_gpu: true`

Reason:
- scripted benchmark 時点で GPU 上で安定
- `Qwen2.5-1.5B-Instruct` + GPU TTS と共存しても OOM を起こさなかった
- 実際の benchmark artifact に `device=cuda:0` の記録がある

#### TTS

File:
- [qwen3_tts_gpu.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/tts/qwen3_tts_gpu.yaml)

Key settings:
- `repo_id: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `backend_class: tts_server.backends.qwen_tts:QwenTtsBackend`
- `device: cuda:0`
- `dtype: bfloat16`
- `voice_instruction: 落ち着いた丁寧な日本語の大学受付音声`
- `warmup_enabled: true`
- `playback_sample_rate_hz: 24000`

Reason:
- 以前の CPU fallback TTS より bench 上の成立性が高かった
- `reception_default + qwen3_asr_gpu + qwen3_tts_gpu` の組み合わせで scripted pass を達成

## Best-Known Command Line

もし launcher が分離 profile を受ける実装に戻っているなら、使うべき引数はこれである。

```bash
ros2 launch ros2_reception_orchestrator reception_bringup.launch.py \
  llm_profile:=reception_default \
  asr_profile:=qwen3_asr_gpu \
  tts_profile:=qwen3_tts_gpu
```

もし現在のように `profile_name` 1 本に潰されている実装を使うなら、最も近い shared catalog entry は [model_profiles.yaml](/workspaces/ros2-workspace-template/config/model_profiles.yaml) の `qwen_fullstack` である。

ただしこれは厳密には benchmark で pass した profile 定義そのものではない。理由は以下。

- profile metadata の管理場所が異なる
- `gpu_memory_utilization` などの LLM runtime tuning が `reception_default` と一致していない
- benchmark は per-task profile で回されている

したがって、**最適構成そのものは `qwen_fullstack` ではなく `reception_default + qwen3_asr_gpu + qwen3_tts_gpu`** と扱うべきである。

## Experimental History

以下は、現在の結論に至るまでの主要な探索と失敗理由である。

### Phase 1: 初期動作確認と単体接続

初期段階では、各 backend の実起動と orchestrator 接続の成立性を優先した。

確認されたこと:
- `ros2_vllm` 単体で `Qwen/Qwen2.5-1.5B-Instruct` は安定起動
- `tts_server` は local playback まで動作
- `asr_streaming_node` は `kotoba-whisper` 系で動作
- Discord bridge は real credentials で channel send / thread create まで確認

この段階の問題:
- ASR は CPU / final-only で遅い
- TTS callback や orchestrator 側の action wait に詰まりがあった
- Discord thread API が不足していた

このフェーズの成果は [e2e_test_report_2026-03-07.md](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/docs/e2e_test_report_2026-03-07.md) にまとまっている。

### Phase 2: モデル探索前の設計整理

この段階で明確になったのは、16GB 単 GPU 機では「大きい LLM を強引に載せる」より「1 回の推論を安定に返す」ほうが重要だという点である。

理由:
- ASR / TTS / LLM を同居させる必要がある
- 受付用途では長文生成より、短い意味理解と短い応答が重要
- context overflow や VRAM 枯渇の方が、わずかな言語性能向上より痛い

この時点での暫定ベストは `Qwen/Qwen2.5-1.5B-Instruct` だった。

### Phase 3: Qwen3.5 系の比較

候補として以下を比較対象に入れた。

- [qwen35_0_8b.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/qwen35_0_8b.yaml)
- [qwen35_2b.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/qwen35_2b.yaml)
- [qwen35_2b_gguf_q5km.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/qwen35_2b_gguf_q5km.yaml)
- [qwen35_4b.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/qwen35_4b.yaml)

#### Qwen3.5 0.8B

結果:
- 起動は成功
- scripted scenarios は全落ち

主な理由:
- slot extraction が弱い
- correction も弱い
- TTS event coverage も悪く、benchmark 上の成立性がなかった

根拠:
- [qwen35_profile_bench_2026-03-10.md](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/docs/benchmarks/qwen35_profile_bench_2026-03-10.md)
- [qwen35_profile_bench_2026-03-10.json](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/docs/benchmarks/qwen35_profile_bench_2026-03-10.json)

判断:
- 0.8B はこの受付タスクには小さすぎた

#### Qwen3.5 2B 通常重み

結果:
- 16GB GPU では `vLLM` EngineCore 初期化が厳しかった

判断:
- 単独 LLM としては候補でも、ASR / TTS 同居構成では余裕がない

#### Qwen3.5 2B GGUF (`Q5_K_M`)

結果:
- 起動失敗
- エラー: `Unknown gguf model_type: qwen3_5`

原因:
- local `vLLM` runtime が Qwen3.5 GGUF をそのまま扱えなかった

判断:
- モデル品質以前に runtime 非対応
- experimental candidate ではあるが、本番候補ではない

### Phase 4: 他 lightweight LLM 候補

候補:
- [rakuten_mini.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/rakuten_mini.yaml)
- [phi4_mini.yaml](/workspaces/ros2-workspace-template/src/ros2_reception_orchestrator/config/model_profiles/llm/phi4_mini.yaml)

評価:
- repo には profile はある
- ただし本 workspace に残っている pass artifact は `reception_default` に対してのみ存在する
- `baseline_whisper_mms` が `Phi-4-mini` を使っているが、これを best とする evidence は残っていない

判断:
- いま best を名乗れるだけの証跡はない

### Phase 5: ASR / TTS の再選定

ASR は、Whisper 系 CPU fallback から `Qwen3-ASR-0.6B` GPU に寄せたことで、LLM/TTS と同居した scripted run の成立性が上がった。

TTS は、CPU fallback の `SpeechT5/MMS-VITS` 系では遅延が大きく、レイテンシ要件が厳しかった。`Qwen3-TTS` GPU backend に変えたことで scripted benchmark の pass まで持ち込めた。

判断:
- この環境では、ASR/TTS も GPU 化した Qwen 系に揃えるほうが一貫していた

### Phase 6: 最終 scripted benchmark

最終的に pass した artifact は [/tmp/reception_default_qwen3tts_all_v2.json](/tmp/reception_default_qwen3tts_all_v2.json) である。

要点:
- `llm_profile`: `reception_default`
- `asr_profile`: `qwen3_asr_gpu`
- `tts_profile`: `qwen3_tts_gpu`
- `happy_path`: pass
- `correction`: pass
- `overlap`: pass
- GPU usage:
  - おおむね `11.7-11.9 GiB / 16.3 GiB`
  - OOM なし

これが現時点で最も強い証拠である。

## Why This Combination Won

### 1. LLM は 1.5B が最もバランスが良かった

`Qwen/Qwen2.5-1.5B-Instruct` は次の点で最も妥当だった。

- 0.8B より semantic extraction が安定
- 2B 以上より VRAM 安全性が高い
- 16GB 機で ASR/TTS と共存できる
- 日本語の短い受付応答に十分

### 2. ASR は Qwen3-ASR GPU が最も全体最適だった

- CPU Whisper 系より遅延と共存性がよい
- scripted artifact で実際に `cuda:0` 稼働が確認済み
- 受付タスクに必要な短文・日本語中心の発話で一貫していた

### 3. TTS は Qwen3-TTS GPU が benchmark を通した

- CPU fallback 系ではレイテンシが厳しかった
- `Qwen3-TTS` により first response と全体 pass 率が改善した
- warmup 後の安定性が高かった

### 4. VRAM 全体収支が成立した

この構成は quality だけでなく、**16GB 単 GPU で OOM しない**ことが大きい。

逆に落ちた案は、次のどちらかで失敗した。

- 品質不足
- runtime / VRAM 不成立

## Rejected Configurations

### `baseline_whisper_mms`

実体:
- LLM: `Phi-4-mini-instruct`
- ASR: `Systran/faster-whisper-small`
- TTS: `esnya/japanese_speecht5_tts`

理由:
- best と判断した benchmark artifact がない
- CPU TTS / Whisper 系は、今回の最終方針と逆行している
- これが今の default になっているのは「上書き」であり、最適化の結論ではない

### `qwen35_0_8b`

理由:
- 全 scripted scenario fail
- semantic quality 不足

### `qwen35_2b_gguf_q5km`

理由:
- `vLLM` が `qwen3_5` GGUF を読めず起動不可

### `qwen35_2b` / `qwen35_4b`

理由:
- 16GB 環境では VRAM 安全性が悪い
- この workspace に「full stack pass」の証跡がない

## Operational Recommendation

### Best Known Production-Like Setting

これを基準値とするべきである。

- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- ASR: `Qwen/Qwen3-ASR-0.6B`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

### What Should Be Reverted

以下は best-known setting から外れているので、戻す対象である。

- shared default `profile_name=baseline_whisper_mms`
- `Phi-4-mini + faster-whisper-small + SpeechT5` を既定とみなす運用

### What Should Stay Experimental

- `Qwen3.5-2B-GGUF:Q5_K_M`
- `Qwen3.5-0.8B`
- `RakutenAI-2.0-mini`
- `Phi-4-mini`

これらは「比較候補」であって、現段階で default に昇格させる理由はない。

## Final Conclusion

**現段階の最適モデル設定は、`reception_default + qwen3_asr_gpu + qwen3_tts_gpu` である。**

具体的なモデル名に直すと次である。

- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- ASR: `Qwen/Qwen3-ASR-0.6B`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

この結論は、

- 実際に残っている scripted benchmark pass artifact
- これ以前の Qwen3.5 系比較失敗
- 16GB 単 GPU での VRAM 制約
- orchestrator 全体の E2E 成立性

を総合したものである。

なお、このレポート作成後に default は `qwen_fullstack` 系へ更新した。今後また `baseline_whisper_mms` 系へ戻された場合は、それは **最適化の結論ではなく drift** と判断してよい。
