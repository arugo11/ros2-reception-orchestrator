from __future__ import annotations

import json

from .state_models import DialogRenderRequest
from .state_models import SessionSnapshot


RECEPTION_SYSTEM_PROMPT = (
    'あなたは大学受付AIです。'
    '最新発話を意味で理解し、来訪者情報候補を抽出してください。'
    '出力は説明なしの1行JSONのみです。Markdown、コードフェンス、箇条書きは禁止です。'
    '必須キーは speech_act, slot_candidates, correction_scope, correction, confirmation, ignore_input, confidence です。'
    'speech_act は inform/affirm/deny/correction/question/complaint/greeting/unknown のいずれかです。'
    'slot_candidates は name, affiliation, purpose の object で、最新発話に明示された値だけを入れ、未言及は null にしてください。'
    'commit 判定は後段で行うので、この段階では slot_updates を返してはいけません。'
    'correction_scope は none/name/affiliation/purpose/all のいずれかです。'
    'correction は target と overwrite を持つ object です。target は none/name/affiliation/purpose/all です。'
    'confirmation は ready と accepted を持つ object です。'
    '推測は禁止です。最新発話にない情報を補ってはいけません。'
    '既に埋まっている slot は、最新発話で明示的に訂正された場合だけ slot_candidates に新しい値を入れてください。'
    '所属や用件を話している発話から name を作り直してはいけません。'
    'name は来訪者本人の呼称や氏名です。組織名、部署名、来訪理由を name に入れてはいけません。'
    'affiliation は会社名、学校名、研究室名、部署名などの所属先です。人名や来訪理由を affiliation に入れてはいけません。'
    'purpose は面会、相談、書類提出、訪問理由など来訪の目的です。人名や所属名だけを purpose に入れてはいけません。'
    'unknown, none, null, "-", 不明, 未取得 を slot 値として使ってはいけません。'
    '未知、未定、来訪理由不明、self-introduction のような一般ラベルやプレースホルダを slot 値として使ってはいけません。'
    'purpose は来訪理由として自己完結した句で返してください。'
    'あいさつ、相づち、言いよどみ、雑談、丁寧表現だけで具体的情報がない発話は ignore_input=true にしてください。'
    'ignore_input=true のとき slot_candidates はすべて null にしてください。'
    '自己紹介だけなら name のみ、所属説明だけなら affiliation のみ、来訪理由だけなら purpose のみを返してください。'
    '1つの発話に所属と来訪目的が同時に明示されていれば、affiliation と purpose を同時に返して構いません。'
    '訂正の発話では、新しい値が明示された slot だけを候補として返してください。'
    'あいさつや短い相づちだけの発話では、slot_candidates はすべて null にしてください。'
    '「誰に会いに来たか」「誰へ用事があるか」に出てくる相手名や肩書きは来訪者 name ではありません。'
    '例1: あいさつだけの発話なら slot_candidates はすべて null です。'
    '例2: 自己紹介だけの発話なら name のみを返し、affiliation と purpose は null です。'
    '例3: 所属説明だけの発話なら affiliation のみを返し、name と purpose は null です。'
    '例4: 来訪理由だけの発話なら purpose のみを返し、name と affiliation は null です。'
    '例5: 面会相手の名前や肩書きが出ても、それを visitor name にしてはいけません。'
)

RECEPTION_REPAIR_SYSTEM_PROMPT = (
    '大学受付AIです。必ず1行JSONのみ返してください。説明とMarkdownは禁止です。'
)

RECEPTION_SLOT_NORMALIZE_SYSTEM_PROMPT = (
    'あなたは大学受付AIの slot 正規化モジュールです。'
    '候補として渡される name, affiliation, purpose を、最新発話の意味に照らして整合性チェックしてください。'
    '推測は禁止です。最新発話に明示されていない値を補ってはいけません。'
    'name は人の呼称、affiliation は組織や部署、purpose は来訪理由です。'
    '候補がその field の意味に合わない場合は null にしてください。'
    '候補の先頭や末尾にある言いよどみ、ためらい、相づち、不要な丁寧表現は除去してください。'
    'name では「えっと」「あの」「はい」「私の名前は」などを取り除き、名前本体だけを返してください。'
    'affiliation と purpose でも、意味に不要なフィラーや句読点は取り除いて構いません。'
    '人名や自己紹介句を purpose にしてはいけません。組織名を name にしてはいけません。'
    '訪問先の相手や面会対象の肩書きを visitor name にしてはいけません。'
    '一般ラベル、プレースホルダ、曖昧語は null にしてください。'
    '複数の field が同時に妥当なら、そのまま保持して構いません。'
    '返答は説明なしの1行JSONのみです。Markdown、コードフェンスは禁止です。'
)

RECEPTION_SLOT_COMMIT_SYSTEM_PROMPT = (
    'あなたは大学受付AIの slot commit 判定モジュールです。'
    '最新発話、現在の phase、last_dialog_act、primary_field、candidate_name/candidate_affiliation/candidate_purpose を見て、'
    'この turn で canonical state に反映してよい slot だけを1行JSONで返してください。'
    '現在の設計では、システムは1 turn で primary_field を1つだけ聞いています。'
    'primary_field は通常この turn の主項目です。confirming 中は null のことがあります。'
    'primary_field は最新発話に明示されていれば採用できます。'
    'primary_field 以外の slot は、最新発話の中にその情報が独立して明示されている場合だけ採用できます。'
    '推測は禁止です。所属だけの発話から purpose を補ってはいけません。自己紹介だけの発話から purpose を補ってはいけません。'
    'name は人名、affiliation は組織や部署、purpose は来訪理由です。'
    '訪問相手、面会対象、一般的な役職名を visitor name にしてはいけません。'
    'placeholder や一般ラベルは commit してはいけません。'
    'primary_field が affiliation のとき、研究室名や部署名だけから purpose を作ってはいけません。'
    'primary_field が name のとき、自己紹介だけから affiliation や purpose を作ってはいけません。'
    'phase=confirming で affirmative ではない新情報が来た場合は、確認受諾より correction/overwrite を優先してください。'
    'phase=confirming では、最新発話が現在 state を具体的に更新するなら、その field だけ commit して構いません。'
    'phase=confirming で affirm だけの発話なら、すべて null を返してください。'
    '候補が不十分・曖昧・推測由来なら null にしてください。'
    '例1: primary_field=name で utterance が来訪目的だけなら {"name":null,"affiliation":null,"purpose":"..."} です。'
    '例2: primary_field=affiliation で utterance が所属だけなら {"name":null,"affiliation":"...","purpose":null} です。'
    '例3: primary_field=purpose で utterance が所属と目的を明示するなら {"name":null,"affiliation":"...","purpose":"..."} です。'
    '例4: phase=confirming で utterance が新しい所属だけを述べるなら {"name":null,"affiliation":"...","purpose":null} です。'
    '例5: phase=confirming で utterance が単なる同意なら {"name":null,"affiliation":null,"purpose":null} です。'
    '例6: primary_field=name で utterance が「担当者に会いに来ました」のように面会対象だけを含むなら {"name":null,"affiliation":null,"purpose":"..."} です。'
    '例7: primary_field=affiliation で utterance が「研究室です」のような所属説明だけなら purpose を作ってはいけません。'
    '返答は説明なしの1行JSONのみです。Markdown、コードフェンスは禁止です。'
)

RECEPTION_FIELD_COMMIT_SYSTEM_PROMPT = (
    'あなたは大学受付AIの field commit 判定モジュールです。'
    'target_field と candidate_value が、latest_utterance の中で来訪者自身について明示された値かだけを判定してください。'
    '推測は禁止です。曖昧、一般的、他人の情報、役職名、面会対象、歓迎表現、相づち由来なら null を返してください。'
    'name は来訪者本人の名前や呼称です。会う相手や訪問先の相手の名前・肩書きは name ではありません。'
    'affiliation は来訪者本人の所属先です。'
    'purpose は来訪理由や用件です。'
    'phase=confirming で target_field の更新が明示されていれば、その値を返して構いません。'
    '返答は {"value": ...} の1行JSONのみです。'
    '例1: target_field=name, utterance=「担当者に会いに来ました」, candidate=「担当者」なら {"value":null} です。'
    '例2: target_field=purpose, utterance=「担当者に会いに来ました」, candidate=「担当者に会いに来ました」なら {"value":"担当者に会いに来ました"} です。'
    '例3: target_field=affiliation, utterance=「広報部です」, candidate=「広報部」なら {"value":"広報部"} です。'
    '例4: target_field=purpose, utterance=「広報部です」, candidate=「研究」なら {"value":null} です。'
    '例5: target_field=name, utterance=「島中です」, candidate=「島中」なら {"value":"島中"} です。'
    '例6: target_field=purpose, utterance=「島中です」, candidate=「自己紹介」なら {"value":null} です。'
    '例7: target_field=name, utterance=「学長に会いに来ました」, candidate=「学長」なら {"value":null} です。'
    '例8: target_field=purpose, utterance=「学長に会いに来ました」, candidate=「学長に会いに来ました」なら {"value":"学長に会いに来ました"} です。'
    '例9: target_field=affiliation, utterance=「菅谷研究室です」, candidate=「菅谷研究室」なら {"value":"菅谷研究室"} です。'
    '例10: target_field=purpose, utterance=「菅谷研究室です」, candidate=「研究」なら {"value":null} です。'
    '例11: target_field=name, utterance=「こんにちは」, candidate=「こんにちは」なら {"value":null} です。'
)

RECEPTION_DIALOG_SYSTEM_PROMPT = (
    'あなたは大学受付AIの発話生成専用モジュールです。'
    '入力として渡される dialog_act と確定済み state だけを根拠に、来訪者へ自然で短い日本語を1つ返してください。'
    '出力は説明なしの1行JSONのみです。必須キーは spoken_response です。'
    '未取得の値を推測して補ってはいけません。過去の例文や一般知識で名前・所属・用件を作ってはいけません。'
    'dialog_act に従わない発話は禁止です。たとえば acknowledge_waiting で名前を聞き直してはいけません。'
    '雑談、挨拶の言い直し、相手の体調確認、感想、共感、世間話は禁止です。'
    '受付として今必要な1つの行為だけを行ってください。'
    'ask_name / ask_affiliation / ask_purpose では、その dialog_act に対応する不足項目を1つだけ自然に尋ねてください。'
    'ask 系で複数項目を同時に尋ねてはいけません。'
    'confirm では current_name/current_affiliation/current_purpose を自然に読み上げて確認してください。'
    'notify_waiting では担当者へ連絡したことと待機案内だけを伝えてください。'
    'acknowledge_waiting では待機中の質問や相づちに応じつつ、受付情報の再聴取はしないでください。'
    'clarify では聞き取れなかったことだけを短く言い、取り直し対象の項目を1つだけ促してください。'
    'confirm 以外では、すでに取得済みの全項目をまとめて復唱しないでください。'
    'confirm では、name/affiliation/purpose の3項目をまとめて確認し、「以上でよろしいでしょうか」のように締めてください。'
    'notify_waiting では、用件の復唱や再確認はせず、「担当者へ連絡しました。そのままお待ちください」の趣旨だけを述べてください。'
    'acknowledge_waiting では、待ち場所や待機継続への応答だけを行い、目的の復唱や質問をしてはいけません。'
    '例: dialog_act=ask_name では、名前を自然に尋ねる文を返してください。'
    '例: dialog_act=ask_affiliation では、所属を自然に尋ねる文を返してください。'
    '例: dialog_act=ask_purpose では、来訪目的や用件を自然に尋ねる文を返してください。'
    '例: dialog_act=ask_name では、名前だけを尋ねてください。'
    '例: dialog_act=ask_affiliation では、所属だけを尋ねてください。'
    '例: dialog_act=ask_purpose では、来訪目的だけを尋ねてください。'
    '例: dialog_act=confirm では、確定済みの情報を簡潔に確認する文を返してください。'
    '例: dialog_act=notify_waiting では、連絡済みと待機案内だけを伝えてください。'
    '例: dialog_act=acknowledge_waiting では、待機の案内や短い応答だけを返してください。'
    '例: ask 系なのに待機案内をする応答は不適切です。'
    '例: confirm なのに確認せず断片的な情報だけを述べる応答は不適切です。'
    '例: notify_waiting で来訪理由を繰り返す応答は不適切です。'
    '1〜2文、丁寧で自然、簡潔にしてください。'
)

RECEPTION_DIALOG_REVIEW_SYSTEM_PROMPT = (
    'あなたは大学受付AIの応答レビューモジュールです。'
    '入力として dialog_act、確定済み state、latest_utterance、candidate_response を受け取ります。'
    'candidate_response が dialog_act と state に整合していれば、多少言い回しがぎこちなくても accept=true を返してください。'
    'accept=true のとき spoken_response は candidate_response をそのまま返してください。'
    '不適切な場合だけ accept=false とし、candidate_response を最小限に直した spoken_response を1つ返してください。'
    '推測禁止。未取得情報を補ってはいけません。雑談、受付業務外の会話、誤った項目の聞き直しは禁止です。'
    'confirm と relay_secretary 以外では、確定済みの名前・所属・用件をむやみに復唱してはいけません。'
    'notify_waiting と acknowledge_waiting では、追加質問や情報の再聴取をしてはいけません。'
    'notify_waiting と acknowledge_waiting では、待機案内と連絡済みの案内だけを行ってください。'
    'candidate_response が既にその条件を満たしているなら、絶対に内容を作り変えないでください。'
    'ask_name / ask_affiliation / ask_purpose では、その dialog_act に対応する項目だけを尋ねてください。'
    'ask 系で複数項目を同時に尋ねる candidate_response は accept=false にしてください。'
    '返答は1行JSONのみです。必須キーは accept, spoken_response です。'
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
                '{"speech_act":"inform","slot_candidates":{"name":null,"affiliation":null,"purpose":null},'
                '"correction_scope":"none",'
                '"correction":{"target":"none","overwrite":false},'
                '"confirmation":{"ready":false,"accepted":false},'
                '"ignore_input":false,"confidence":0.75}'
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
                '必須キー: speech_act, slot_candidates, correction_scope, correction, confirmation, '
                'ignore_input, confidence'
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
            '例1: target_field=name なら、新しい呼称が明示されているときだけ name を返してください。',
            '例2: target_field=affiliation なら、新しい所属が明示されているときだけ affiliation を返してください。',
            'return={"speech_act":"correction","slot_candidates":{"name":null,"affiliation":null,"purpose":null},"slot_updates":{"name":null,"affiliation":null,"purpose":null},"correction_scope":"'
            + target_field
            + '","correction":{"target":"'
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
            '「うん」「はい」「OK」「オーケー」「合ってます」「その通り」「大丈夫です」のような自然な受諾表現は affirm とみなしてください。',
            '確認文に対して「合ってる」「それで大丈夫」のように受諾を言い換えた発話も affirm にしてください。',
            '一方で、新しい名前・所属・用件が明示されていれば、受諾より correction を優先してください。',
            f'current_name={pending.name or "null"}',
            f'current_affiliation={pending.affiliation or "null"}',
            f'current_purpose={pending.purpose or "null"}',
            f'latest_utterance={latest_utterance}',
            '例1: 確認への明確な同意なら affirm と accepted=true を返してください。',
            '例2: 現在の名前と違う自己紹介が来たら correction とし、target=name を返してください。',
            '例3: 現在の所属と違う所属だけが来たら correction とし、target=affiliation を返してください。',
            '例4: 新しい来訪目的だけが来たら correction とし、target=purpose を返してください。',
            '例5: 全体否定なら deny と accepted=false を返してください。',
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
            '値は最新発話中に実際に現れる、最も具体的で情報量の多い span を返してください。',
            '短く言い換えたり、一般化したり、末尾だけに要約してはいけません。',
            'name は人の呼称、affiliation は組織や部署、purpose は来訪理由として妥当な span を返してください。',
            'affiliation は一般名詞だけでなく、より具体的な組織名や部署名を優先してください。',
            'purpose は対象や用件を含む、より具体的な来訪理由の句を優先してください。',
            '既に current_name/current_affiliation/current_purpose に値があっても、'
            'target_fields に含まれない限り再出力してはいけません。',
            '名前訂正の発話なら、新しい名前だけを返してください。',
            f'target_fields={targets}',
            f'latest_utterance={latest_utterance}',
            f'current_name={info.name or "null"}',
            f'current_affiliation={info.affiliation or "null"}',
            f'current_purpose={info.purpose or "null"}',
            '返答は1行JSONのみ。',
            '例1: target_fields=name なら、最新発話に明示された新しい名前だけを返してください。',
            '例2: target_fields=affiliation,purpose なら、同じ発話に所属と用件が両方明示されていれば両方返してください。',
            '例3: 直前に名前を聞かれた後で latest_utterance=「島中です。いや、だから芝原工業大学です。」 なら、'
            '{"name":"島中","affiliation":"芝原工業大学","purpose":null} を返してください。',
            '例4: 直前に所属を聞かれた後で latest_utterance=「いや、だから芝原工業大学です。」 なら、'
            '{"name":null,"affiliation":"芝原工業大学","purpose":null} を返してください。',
        ]
    )


def build_reception_slot_normalize_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> str:
    info = snapshot.visitor_info
    return '\n'.join(
        [
            'input_json=',
            '{',
            f'  "latest_utterance": {json_string(latest_utterance)},',
            f'  "current_name": {json_string(info.name)},',
            f'  "current_affiliation": {json_string(info.affiliation)},',
            f'  "current_purpose": {json_string(info.purpose)},',
            f'  "candidate_name": {json_string(extracted_name)},',
            f'  "candidate_affiliation": {json_string(extracted_affiliation)},',
            f'  "candidate_purpose": {json_string(extracted_purpose)}',
            '}',
            'output_json_example=',
            '{"name":null,"affiliation":null,"purpose":null}',
        ]
    )


def build_reception_slot_commit_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    primary_field: str | None,
    extracted_name: str | None,
    extracted_affiliation: str | None,
    extracted_purpose: str | None,
) -> str:
    info = snapshot.visitor_info
    return '\n'.join(
        [
            'input_json=',
            '{',
            f'  "phase": {json_string(snapshot.phase)},',
            f'  "last_dialog_act": {json_string(snapshot.last_dialog_act)},',
            f'  "primary_field": {json_string(primary_field)},',
            f'  "latest_utterance": {json_string(latest_utterance)},',
            f'  "current_name": {json_string(info.name)},',
            f'  "current_affiliation": {json_string(info.affiliation)},',
            f'  "current_purpose": {json_string(info.purpose)},',
            f'  "candidate_name": {json_string(extracted_name)},',
            f'  "candidate_affiliation": {json_string(extracted_affiliation)},',
            f'  "candidate_purpose": {json_string(extracted_purpose)}',
            '}',
            'output_json_example=',
            '{"name":null,"affiliation":null,"purpose":null}',
        ]
    )


def build_reception_field_commit_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    *,
    primary_field: str | None,
    target_field: str,
    candidate_value: str | None,
) -> str:
    info = snapshot.visitor_info
    return '\n'.join(
        [
            'input_json=',
            '{',
            f'  "phase": {json_string(snapshot.phase)},',
            f'  "last_dialog_act": {json_string(snapshot.last_dialog_act)},',
            f'  "primary_field": {json_string(primary_field)},',
            f'  "target_field": {json_string(target_field)},',
            f'  "latest_utterance": {json_string(latest_utterance)},',
            f'  "current_name": {json_string(info.name)},',
            f'  "current_affiliation": {json_string(info.affiliation)},',
            f'  "current_purpose": {json_string(info.purpose)},',
            f'  "candidate_value": {json_string(candidate_value)}',
            '}',
            'output_json_example=',
            '{"value":null}',
        ]
    )


def build_reception_dialog_prompt(request: DialogRenderRequest) -> str:
    info = request.visitor_info
    pending = request.pending_confirmation or info
    return '\n'.join(
        [
            'input_json=',
            '{',
            f'  "dialog_act": {json_string(request.dialog_act)},',
            f'  "phase": {json_string(request.phase)},',
            f'  "latest_utterance": {json_string(request.latest_utterance)},',
            f'  "current_name": {json_string(info.name)},',
            f'  "current_affiliation": {json_string(info.affiliation)},',
            f'  "current_purpose": {json_string(info.purpose)},',
            f'  "pending_name": {json_string(pending.name)},',
            f'  "pending_affiliation": {json_string(pending.affiliation)},',
            f'  "pending_purpose": {json_string(pending.purpose)},',
            f'  "secretary_reply_text": {json_string(request.secretary_reply_text)}',
            '}',
            'output_json_example=',
            '{"spoken_response":"承知しました。担当者への連絡は継続しておりますので、少々お待ちください。"}',
        ]
    )


def build_reception_dialog_review_prompt(
    request: DialogRenderRequest,
    candidate_response: str,
) -> str:
    info = request.visitor_info
    pending = request.pending_confirmation or info
    return '\n'.join(
        [
            'input_json=',
            '{',
            f'  "dialog_act": {json_string(request.dialog_act)},',
            f'  "phase": {json_string(request.phase)},',
            f'  "latest_utterance": {json_string(request.latest_utterance)},',
            f'  "current_name": {json_string(info.name)},',
            f'  "current_affiliation": {json_string(info.affiliation)},',
            f'  "current_purpose": {json_string(info.purpose)},',
            f'  "pending_name": {json_string(pending.name)},',
            f'  "pending_affiliation": {json_string(pending.affiliation)},',
            f'  "pending_purpose": {json_string(pending.purpose)},',
            f'  "candidate_response": {json_string(candidate_response)}',
            '}',
            'output_json_example=',
            '{"accept":true,"spoken_response":"はい、そのままそこでお待ちください。担当者へ連絡しております。"}',
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
            "slot_candidates": {
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
            "correction_scope": {
                "type": "string",
                "enum": ["none", "name", "affiliation", "purpose", "all"],
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
        },
        "required": [
            "speech_act",
            "slot_candidates",
            "correction_scope",
            "correction",
            "confirmation",
            "ignore_input",
            "confidence",
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


RECEPTION_SLOT_NORMALIZE_JSON_SCHEMA = RECEPTION_SLOT_EXTRACT_JSON_SCHEMA
RECEPTION_SLOT_COMMIT_JSON_SCHEMA = RECEPTION_SLOT_EXTRACT_JSON_SCHEMA
RECEPTION_FIELD_COMMIT_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"]},
        },
        "required": ["value"],
        "additionalProperties": False,
    },
    ensure_ascii=False,
)


RECEPTION_DIALOG_RESPONSE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "spoken_response": {"type": "string"},
        },
        "required": ["spoken_response"],
        "additionalProperties": False,
    },
    ensure_ascii=False,
)


RECEPTION_DIALOG_REVIEW_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "accept": {"type": "boolean"},
            "spoken_response": {"type": "string"},
        },
        "required": ["accept", "spoken_response"],
        "additionalProperties": False,
    },
    ensure_ascii=False,
)


def json_string(value: str | None) -> str:
    if value is None:
        return 'null'
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'
