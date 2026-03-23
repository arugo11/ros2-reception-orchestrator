from __future__ import annotations

import json
from urllib.request import urlopen

from builtin_interfaces.msg import Time
from reception_interfaces.msg import ConversationTrace
from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent
from reception_interfaces.msg import SessionStateV2
from ros2_chat_interfaces.msg import ChatBridgeStatus
from ros2_vllm_interfaces.msg import LlmStatus

from ros2_reception_orchestrator.conversation_trace import build_conversation_trace_message
from ros2_reception_orchestrator.conversation_trace import conversation_trace_to_dict
from ros2_reception_orchestrator.demo_dashboard import DashboardServer
from ros2_reception_orchestrator.demo_dashboard import DashboardStore
from ros2_reception_orchestrator.demo_dashboard import chat_status_to_dict
from ros2_reception_orchestrator.demo_dashboard import execution_event_to_dict
from ros2_reception_orchestrator.demo_dashboard import llm_status_to_dict
from ros2_reception_orchestrator.demo_dashboard import session_state_to_dict


def _stamp(sec: int, nanosec: int = 0) -> Time:
    stamp = Time()
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


def test_build_conversation_trace_message_and_decode_payload() -> None:
    msg = build_conversation_trace_message(
        timestamp=_stamp(10, 500_000_000),
        session_id='session-1',
        turn_seq=3,
        role=ConversationTrace.ROLE_USER,
        text='島中です',
        phase='collecting',
        utterance_id='utt-1',
        asr_confidence=0.93,
        event_type='UTTERANCE_RECEIVED',
        event_payload='{"captured_during_tts": false}',
        payload_json='{"captured_during_tts": false}',
    )

    payload = conversation_trace_to_dict(msg)

    assert payload['session_id'] == 'session-1'
    assert payload['turn_seq'] == 3
    assert payload['role'] == 'user'
    assert payload['phase'] == 'collecting'
    assert payload['event_type'] == 'UTTERANCE_RECEIVED'
    assert payload['payload'] == {'captured_during_tts': False}


def test_conversation_trace_to_dict_handles_invalid_payload_json() -> None:
    msg = build_conversation_trace_message(
        timestamp=_stamp(11),
        session_id='session-2',
        turn_seq=0,
        role=ConversationTrace.ROLE_SYSTEM,
        text='こんにちは, Welcome to SIT',
        dialog_act='system_ready',
        payload_json='{invalid',
    )

    payload = conversation_trace_to_dict(msg)

    assert payload['role'] == 'system'
    assert payload['payload'] == {'raw': '{invalid'}


def test_execution_event_to_dict_maps_labels_and_severity() -> None:
    msg = ExecutionEvent()
    msg.timestamp = _stamp(12)
    msg.command_id = 'cmd-1'
    msg.command_type = ExecutionCommand.COMMAND_TTS
    msg.session_id = 'session-3'
    msg.turn_seq = 4
    msg.status = ExecutionEvent.STATUS_CANCELED
    msg.reason_code = ExecutionEvent.REASON_REPLACED
    msg.detail = 'same_turn_replaced'

    payload = execution_event_to_dict(msg)

    assert payload['command_type_label'] == 'tts'
    assert payload['status_label'] == 'canceled'
    assert payload['reason_label'] == 'replaced'
    assert payload['severity'] == 'warn'


def test_status_normalizers_expose_ready_state() -> None:
    llm = LlmStatus()
    llm.status = LlmStatus.READY
    llm.model_name = 'Qwen/Test'
    llm.message = 'ready'
    llm_payload = llm_status_to_dict(llm)

    chat = ChatBridgeStatus()
    chat.status = ChatBridgeStatus.DEGRADED
    chat.gateway_connected = False
    chat.sidecar_reachable = True
    chat.active_adapters = ['discord']
    chat.message = 'gateway reconnecting'
    chat_payload = chat_status_to_dict(chat)

    assert llm_payload['ready'] is True
    assert llm_payload['meta']['model'] == 'Qwen/Test'
    assert chat_payload['ready'] is False
    assert chat_payload['severity'] == 'warn'


def test_dashboard_store_deduplicates_identical_session_state() -> None:
    store = DashboardStore(config={'profile_name': 'qwen_fullstack', 'topics': {}})
    msg = SessionStateV2()
    msg.timestamp = _stamp(13)
    msg.session_id = 'session-4'
    msg.phase = 'collecting'
    msg.response_language = 'en'
    msg.focus_slot = 'name'
    msg.last_system_act = 'ask_name'
    msg.pending_clarification_slot = ''
    msg.chat_delivery_state = 'queued'
    msg.working_info.name = '島中'
    msg.latest_applied_turn = 1
    msg.version = 7
    payload = session_state_to_dict(msg)

    store.update_session_state(payload)
    version_once = store.snapshot()['version']
    store.update_session_state(payload)
    version_twice = store.snapshot()['version']

    assert version_once == version_twice
    assert payload['response_language'] == 'en'
    assert payload['working_info']['name'] == '島中'


def test_dashboard_server_snapshot_and_root() -> None:
    store = DashboardStore(config={'profile_name': 'qwen_fullstack', 'topics': {}})
    store.record_conversation(
        conversation_trace_to_dict(
            build_conversation_trace_message(
                timestamp=_stamp(14),
                session_id='session-5',
                turn_seq=1,
                role=ConversationTrace.ROLE_ASSISTANT,
                text='ご所属を教えてください。',
                dialog_act='ask_affiliation',
            )
        )
    )
    server = DashboardServer(store=store, host='127.0.0.1', port=0)
    server.start()
    try:
        port = int(server._server.server_address[1])
        with urlopen(f'http://127.0.0.1:{port}/api/snapshot', timeout=3.0) as response:
            payload = json.loads(response.read().decode('utf-8'))
        with urlopen(f'http://127.0.0.1:{port}/', timeout=3.0) as response:
            html = response.read().decode('utf-8')
    finally:
        server.stop()

    assert payload['conversation'][0]['dialog_act'] == 'ask_affiliation'
    assert 'Reception Demo Dashboard' in html
