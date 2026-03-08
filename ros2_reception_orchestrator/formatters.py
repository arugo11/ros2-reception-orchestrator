from __future__ import annotations

from .state_models import DialogAct
from .state_models import SessionState
from .state_models import VisitorInfo


def _display(value: str | None) -> str:
    return value.strip() if value and value.strip() else '未取得'


def format_initial_post(session: SessionState) -> str:
    info = session.visitor_info
    return '\n'.join(
        [
            '【新規来訪】',
            f'セッションID: {session.session_id}',
            f'状態: {session.phase}',
            f'名前: {_display(info.name)}',
            f'所属: {_display(info.affiliation)}',
            f'来訪目的: {_display(info.purpose)}',
            'メモ: 来訪者が受付で会話を開始',
        ]
    )


def format_update_post(session: SessionState) -> str:
    info = session.visitor_info
    missing = ', '.join(info.missing_fields()) or 'なし'
    return '\n'.join(
        [
            '【情報更新】',
            f'セッションID: {session.session_id}',
            f'状態: {session.phase}',
            f'名前: {_display(info.name)}',
            f'所属: {_display(info.affiliation)}',
            f'来訪目的: {_display(info.purpose)}',
            f'不足項目: {missing}',
        ]
    )


def format_confirmed_post(session: SessionState) -> str:
    info = session.visitor_info
    return '\n'.join(
        [
            '【受付内容確定】',
            f'セッションID: {session.session_id}',
            f'名前: {_display(info.name)}',
            f'所属: {_display(info.affiliation)}',
            f'来訪目的: {_display(info.purpose)}',
            '受付AI: 来訪者への確認済み',
        ]
    )


def fallback_dialog_text(dialog_act: DialogAct, info: VisitorInfo) -> str:
    if dialog_act == 'ask_name':
        return '恐れ入ります。お名前を伺ってもよろしいでしょうか。'
    if dialog_act == 'ask_affiliation':
        if info.name:
            return f'{info.name}さん、ご所属を教えていただけますか。'
        return 'ご所属を教えていただけますか。'
    if dialog_act == 'ask_purpose':
        if info.name:
            return f'{info.name}さん、本日のご用件を教えていただけますか。'
        return '本日のご用件を教えていただけますか。'
    if dialog_act == 'confirm':
        return (
            f'お名前は{_display(info.name)}、ご所属は{_display(info.affiliation)}、'
            f'ご用件は{_display(info.purpose)}でお間違いないでしょうか。'
        )
    if dialog_act == 'notify_waiting':
        return '担当者へ連絡しました。少々お待ちください。'
    if dialog_act == 'acknowledge_waiting':
        return '承知しました。担当者への連絡は継続しておりますので、少々お待ちください。'
    if dialog_act == 'relay_secretary':
        return '担当者からの返信が届きました。'
    if dialog_act == 'clarify':
        return '恐れ入ります。もう一度、短く教えていただけますか。'
    return '恐れ入ります。もう一度お願いいたします。'
