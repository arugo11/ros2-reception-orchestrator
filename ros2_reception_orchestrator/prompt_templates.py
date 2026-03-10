from __future__ import annotations

import json

from .state_models import SessionSnapshot


RECEPTION_SYSTEM_PROMPT = (
    'あなたは大学受付AIです。'
    '最新発話を意味で理解し、来訪者に返す自然な短い日本語を1つ作ってください。'
    '出力は説明なしの1行JSONのみです。Markdown、コードフェンス、箇条書きは禁止です。'
    '必須キーは speech_act, slot_updates, correction, confirmation, ignore_input, confidence, spoken_response です。'
    'speech_act は inform/affirm/deny/correction/question/complaint/greeting/unknown のいずれかです。'
    'slot_updates は name, affiliation, purpose の object で、最新発話に明示された値だけを入れ、未言及は null にしてください。'
    'correction は target と overwrite を持つ object です。target は none/name/affiliation/purpose/all です。'
    'confirmation は ready と accepted を持つ object です。'
    '推測は禁止です。最新発話にない情報を補ってはいけません。'
    '既に埋まっている slot は、最新発話で明示的に訂正された場合だけ slot_updates で上書きできます。'
    '所属や用件を話している発話から name を作り直してはいけません。'
    'unknown, none, null, "-", 不明, 未取得 を slot 値として使ってはいけません。'
    'purpose は「学長に会いに来ました」のように自己完結した句で返してください。'
    'spoken_response は来訪者にそのまま話す本文です。1〜2文、丁寧で自然、今必要なことだけを伝えてください。'
    '複数質問、役割説明、内部語は禁止です。'
    '例: latest_utterance=私の名前は島中です -> slot_updates.name=島中。'
    '例: latest_utterance=菅谷研究室に所属しており、学長に会いに来ました -> affiliation=菅谷研究室, purpose=学長に会いに来ました。'
    '例: latest_utterance=名前が違います。島中です -> correction.target=name, overwrite=true, slot_updates.name=島中。'
    '例: latest_utterance=こんにちは -> slot_updates はすべて null。purpose を推測してはいけません。'
)

RECEPTION_REPAIR_SYSTEM_PROMPT = (
    '大学受付AIです。必ず1行JSONのみ返してください。説明とMarkdownは禁止です。'
)

RECEPTION_CONFIRMATION_RESCUE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "speech_act": {
                "type": "string",
                "enum": ["affirm", "deny", "correction", "unknown"],
            },
            "correction": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["none", "name", "affiliation", "purpose", "all"],
                    },
                    "overwrite": {"type": "boolean"},
                },
                "required": ["target", "overwrite"],
                "additionalProperties": False,
            },
            "confirmation": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean"},
                    "accepted": {"type": "boolean"},
                },
                "required": ["ready", "accepted"],
                "additionalProperties": False,
            },
        },
        "required": ["speech_act", "correction", "confirmation"],
        "additionalProperties": False,
    },
    ensure_ascii=False,
)


def build_reception_user_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    currently_speaking: bool,
    captured_during_tts: bool,
) -> str:
    info = snapshot.visitor_info
    pending = snapshot.pending_confirmation or info
    last_act = snapshot.last_dialog_act or 'null'
    last_spoken = snapshot.last_spoken_text or 'null'
    return '\n'.join(
        [
            'input_json=',
            '{',
            f'  "phase": "{snapshot.phase}",',
            f'  "latest_utterance": {json_string(latest_utterance)},',
            f'  "current_name": {json_string(info.name)},',
            f'  "current_affiliation": {json_string(info.affiliation)},',
            f'  "current_purpose": {json_string(info.purpose)},',
            f'  "pending_name": {json_string(pending.name)},',
            f'  "pending_affiliation": {json_string(pending.affiliation)},',
            f'  "pending_purpose": {json_string(pending.purpose)},',
            f'  "last_system_act": {json_string(last_act)},',
            f'  "last_spoken_text": {json_string(last_spoken)},',
            f'  "currently_speaking": {"true" if currently_speaking else "false"},',
            f'  "captured_during_tts": {"true" if captured_during_tts else "false"}',
            '}',
            'output_json_example=',
            (
                '{"speech_act":"inform","slot_updates":{"name":null,"affiliation":null,"purpose":null},'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},'
                '"ignore_input":false,"confidence":0.75,'
                '"spoken_response":"恐れ入りますが、お名前を伺ってもよろしいでしょうか。"}'
            ),
        ]
    )


def build_reception_repair_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    bad_response: str,
    *,
    currently_speaking: bool,
    captured_during_tts: bool,
) -> str:
    info = snapshot.visitor_info
    return '\n'.join(
        [
            '次の bad_response を正しい1行JSONに修正してください。',
            f'phase={snapshot.phase}',
            f'latest_utterance={latest_utterance}',
            f'current_name={info.name or "null"}',
            f'current_affiliation={info.affiliation or "null"}',
            f'current_purpose={info.purpose or "null"}',
            f'last_system_act={snapshot.last_dialog_act or "null"}',
            f'currently_speaking={"true" if currently_speaking else "false"}',
            f'captured_during_tts={"true" if captured_during_tts else "false"}',
            f'bad_response={bad_response}',
            (
                '必須キー: speech_act, slot_updates, correction, confirmation, '
                'ignore_input, confidence, spoken_response'
            ),
        ]
    )


def build_reception_correction_rescue_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    target_field: str,
) -> str:
    del snapshot
    return '\n'.join(
        [
            '次の発話は既存情報の訂正です。',
            'target_field に対して、latest_utterance に明示された新しい値だけを抽出してください。',
            '以前の値や現在の state を推測に使ってはいけません。latest_utterance だけを見てください。',
            'latest_utterance に新しい値が明示されていない場合だけ null を返してください。',
            '推測は禁止です。訂正されていない他の slot は必ず null にしてください。',
            f'target_field={target_field}',
            f'latest_utterance={latest_utterance}',
            '例1: target_field=name, latest_utterance=名前が違います。島中です -> slot_updates.name=島中',
            '例2: target_field=affiliation, latest_utterance=所属が違います。菅谷研究室です -> slot_updates.affiliation=菅谷研究室',
            'return={"speech_act":"correction","slot_updates":{"name":null,"affiliation":null,"purpose":null},"correction":{"target":"'
            + target_field
            + '","overwrite":true},"confirmation":{"ready":false,"accepted":false},"ignore_input":false,"confidence":0.0,"spoken_response":"承知しました。"}',
        ]
    )


def build_reception_confirmation_rescue_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
) -> str:
    pending = snapshot.pending_confirmation or snapshot.visitor_info
    return '\n'.join(
        [
            '現在は確認フェーズです。latest_utterance が確認内容への受諾か、訂正か、否定か、判別不能かを JSON のみで返してください。',
            '確認対象は current_name/current_affiliation/current_purpose です。',
            '受諾なら speech_act=affirm, confirmation.accepted=true を返してください。',
            '訂正なら speech_act=correction または deny とし、correction.target を返してください。',
            '判別不能なら speech_act=unknown にしてください。',
            '推測禁止。slot 値は返しません。',
            f'current_name={pending.name or "null"}',
            f'current_affiliation={pending.affiliation or "null"}',
            f'current_purpose={pending.purpose or "null"}',
            f'latest_utterance={latest_utterance}',
            '例1: latest_utterance=はい -> {"speech_act":"affirm","correction":{"target":"none","overwrite":false},"confirmation":{"ready":true,"accepted":true}}',
            '例2: latest_utterance=名前が違います。島中です -> {"speech_act":"correction","correction":{"target":"name","overwrite":true},"confirmation":{"ready":false,"accepted":false}}',
            '例3: latest_utterance=違います -> {"speech_act":"deny","correction":{"target":"all","overwrite":false},"confirmation":{"ready":false,"accepted":false}}',
        ]
    )


def build_reception_slot_extract_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    target_fields: list[str],
) -> str:
    info = snapshot.visitor_info
    targets = ','.join(target_fields)
    return '\n'.join(
        [
            '最新発話から target_fields に含まれる slot だけを抽出してください。',
            '推測は禁止です。最新発話に明示された値だけを返してください。',
            'target_fields に含まれない slot は必ず null にしてください。',
            '既に current_name/current_affiliation/current_purpose に値があっても、'
            'target_fields に含まれない限り再出力してはいけません。',
            '名前訂正の発話なら、新しい名前だけを返してください。',
            f'target_fields={targets}',
            f'latest_utterance={latest_utterance}',
            f'current_name={info.name or "null"}',
            f'current_affiliation={info.affiliation or "null"}',
            f'current_purpose={info.purpose or "null"}',
            '返答は1行JSONのみ。',
            '例1: target_fields=name latest_utterance=名前が違います。島中です -> {"name":"島中","affiliation":null,"purpose":null}',
            '例2: target_fields=affiliation,purpose latest_utterance=菅谷研究室に所属しており、学長に会いに来ました -> {"name":null,"affiliation":"菅谷研究室","purpose":"学長に会いに来ました"}',
        ]
    )


RECEPTION_RESPONSE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "speech_act": {
                "type": "string",
                "enum": [
                    "inform",
                    "affirm",
                    "deny",
                    "correction",
                    "question",
                    "complaint",
                    "greeting",
                    "unknown",
                ],
            },
            "slot_updates": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "affiliation": {"type": ["string", "null"]},
                    "purpose": {"type": ["string", "null"]},
                },
                "required": ["name", "affiliation", "purpose"],
                "additionalProperties": False,
            },
            "correction": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["none", "name", "affiliation", "purpose", "all"],
                    },
                    "overwrite": {"type": "boolean"},
                },
                "required": ["target", "overwrite"],
                "additionalProperties": False,
            },
            "confirmation": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean"},
                    "accepted": {"type": "boolean"},
                },
                "required": ["ready", "accepted"],
                "additionalProperties": False,
            },
            "ignore_input": {"type": "boolean"},
            "confidence": {"type": "number"},
            "spoken_response": {"type": "string"},
        },
        "required": [
            "speech_act",
            "slot_updates",
            "correction",
            "confirmation",
            "ignore_input",
            "confidence",
            "spoken_response",
        ],
        "additionalProperties": False,
    },
    ensure_ascii=False,
)


RECEPTION_SLOT_EXTRACT_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"]},
            "affiliation": {"type": ["string", "null"]},
            "purpose": {"type": ["string", "null"]},
        },
        "required": ["name", "affiliation", "purpose"],
        "additionalProperties": False,
    },
    ensure_ascii=False,
)


def json_string(value: str | None) -> str:
    if value is None:
        return 'null'
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'
