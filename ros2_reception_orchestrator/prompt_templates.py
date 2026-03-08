from __future__ import annotations

from .state_models import DialogRenderRequest
from .state_models import SessionSnapshot


DIALOG_SYSTEM_PROMPT = (
    'あなたは大学の受付担当です。'
    'dialog_act と state だけを使い、来訪者に向けた丁寧で自然な日本語を1文だけ返してください。'
    '内部語、説明、箇条書き、JSON、引用符、復唱しすぎ、複数質問は禁止です。'
    'confirm 以外では未取得の項目を1つだけ尋ねてください。'
    '受付として簡潔に答え、余計な前置きは入れないでください。'
)

SUPERVISOR_SYSTEM_PROMPT = (
    '大学受付の理解担当。'
    '返答は1行JSONのみ。説明、Markdown、コードフェンス禁止。'
    '出力キーは speech_act, extracted_name, extracted_affiliation, extracted_purpose, next_dialog_act, should_confirm, correction_target, discord_update_kind, ignore_input。'
    'speech_act は inform/affirm/deny/correction/question/complaint/greeting/unknown。'
    'next_dialog_act は ask_name/ask_affiliation/ask_purpose/confirm/notify_waiting/acknowledge_waiting/clarify/retry/relay_secretary。'
    'correction_target は none/name/affiliation/purpose/all。'
    'discord_update_kind は initial/update/confirmed/none。'
    '今回の発話で明示された name/affiliation/purpose だけを入れる。推測禁止。未言及は null。'
    'greeting や noise は ignore_input=true。'
    'unknown や none を extracted_* に入れない。'
    '「名前が違います。島中です」なら speech_act=correction, extracted_name=\"島中\", correction_target=\"name\"。'
    '「菅谷研究室に所属しており、学長に会いに来ました」なら affiliation と purpose を両方入れる。'
)

SUPERVISOR_REPAIR_SYSTEM_PROMPT = (
    '大学受付の理解担当。必ず1行JSONのみ返す。説明とMarkdownは禁止。'
)

SLOT_EXTRACTION_SYSTEM_PROMPT = (
    '大学受付の補助抽出担当。返答は1行JSONのみ。'
    'target_fields に含まれる項目だけ今回の発話から抽出する。その他は null。'
    '推測禁止。未言及は null。unknown や none を入れない。'
    '訂正発話では訂正された項目だけ返す。'
    '「島中です」なら name=\"島中\"。'
    '「菅谷研究室に所属しており、学長に会いに来ました」なら affiliation=\"菅谷研究室\", purpose=\"学長に会いに来ました\"。'
)


def build_supervisor_user_prompt(snapshot: SessionSnapshot, latest_utterance: str) -> str:
    info = snapshot.visitor_info
    pending = snapshot.pending_confirmation or info
    return (
        f'ph={snapshot.phase} '
        f'utt={latest_utterance} '
        f'name={info.name or "-"} '
        f'aff={info.affiliation or "-"} '
        f'pur={info.purpose or "-"} '
        f'pname={pending.name or "-"} '
        f'paff={pending.affiliation or "-"} '
        f'ppur={pending.purpose or "-"} '
        f'last={snapshot.last_dialog_act or "-"}'
    )


def build_supervisor_repair_prompt(snapshot: SessionSnapshot, latest_utterance: str, bad_response: str) -> str:
    info = snapshot.visitor_info
    return (
        f'utt="{latest_utterance}" '
        f'ph="{snapshot.phase}" '
        f'name="{info.name or ""}" '
        f'aff="{info.affiliation or ""}" '
        f'pur="{info.purpose or ""}" '
        f'bad="{bad_response}" '
        'JSON only with keys speech_act,extracted_name,extracted_affiliation,extracted_purpose,'
        'next_dialog_act,should_confirm,correction_target,discord_update_kind,ignore_input'
    )


def build_slot_extraction_prompt(
    snapshot: SessionSnapshot,
    latest_utterance: str,
    target_fields: list[str],
) -> str:
    info = snapshot.visitor_info
    return (
        f'target={",".join(target_fields)} '
        f'last={snapshot.last_dialog_act or "-"} '
        f'utt={latest_utterance} '
        f'name={info.name or "-"} '
        f'aff={info.affiliation or "-"} '
        f'pur={info.purpose or "-"} '
        'return={"name":null,"affiliation":null,"purpose":null}'
    )


def build_dialog_user_prompt(request: DialogRenderRequest) -> str:
    info = request.visitor_info
    pending = request.pending_confirmation or info
    prompt = (
        f'dialog_act={request.dialog_act} '
        f'phase={request.phase} '
        f'name={info.name or "-"} '
        f'affiliation={info.affiliation or "-"} '
        f'purpose={info.purpose or "-"} '
        f'confirm_name={pending.name or "-"} '
        f'confirm_affiliation={pending.affiliation or "-"} '
        f'confirm_purpose={pending.purpose or "-"} '
    )
    if request.dialog_act == 'ask_name':
        prompt += 'task=受付として名前だけを丁寧に尋ねる。所属や目的には触れない。 '
    elif request.dialog_act == 'ask_affiliation':
        prompt += 'task=受付として所属だけを丁寧に尋ねる。名前や目的には触れない。相手の名前確認や訂正確認をしない。 '
    elif request.dialog_act == 'ask_purpose':
        prompt += 'task=受付として来訪目的だけを丁寧に尋ねる。名前や所属の確認はしない。会いたい相手や用件を聞く。1つの質問だけにする。 '
    elif request.dialog_act == 'confirm':
        prompt += 'task=現在の名前・所属・目的を自然に復唱して確認する。 '
    elif request.dialog_act == 'notify_waiting':
        prompt += 'task=担当者へ連絡したので少し待つよう案内する。 '
    elif request.dialog_act == 'acknowledge_waiting':
        prompt += 'task=待機継続を短く丁寧に案内する。 '
    elif request.dialog_act == 'clarify':
        prompt += 'task=聞き取りづらかったので短く言い直しをお願いする。 '
    if request.secretary_reply_text:
        prompt += f'secretary_reply="{request.secretary_reply_text}" '
    prompt += '1文だけ返すこと。'
    return prompt


def build_dialog_repair_prompt(request: DialogRenderRequest, bad_response: str) -> str:
    return '\n'.join(
        [
            f'dialog_act={request.dialog_act}',
            f'bad_response={bad_response}',
            '次の条件で1文に修正してください:',
            '- 短く自然な受付の日本語',
            '- 内部語なし',
            '- 繰り返しなし',
            '- 複数質問なし',
        ]
    )
