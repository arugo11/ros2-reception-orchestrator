from __future__ import annotations

import time
from typing import Any

from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent

from ros2_reception_orchestrator.effect_executor_v2 import EffectExecutor
from ros2_reception_orchestrator.v2_types import OrchestratorCommandData


def _cmd(turn_seq: int, cid: str) -> OrchestratorCommandData:
    return OrchestratorCommandData(
        command_type=ExecutionCommand.COMMAND_TTS,
        command_id=cid,
        session_id='s1',
        turn_seq=turn_seq,
        payload_json='{}',
    )


def test_effect_executor_same_turn_replace_cross_turn_keep() -> None:
    events: list[tuple[str, int, int]] = []
    executed: list[str] = []
    gate = {'open': False}

    def invoke_tts(command: OrchestratorCommandData) -> tuple[bool, str]:
        while not gate['open']:
            time.sleep(0.01)
        executed.append(command.command_id)
        return True, 'ok'

    def publish_event(command: OrchestratorCommandData, status: int, reason: int, detail: str) -> None:
        del detail
        events.append((command.command_id, status, reason))

    def completion_hook(command: OrchestratorCommandData, ok: bool, detail: str) -> None:
        del command, ok, detail

    executor = EffectExecutor(
        invoke_tts=invoke_tts,
        publish_event=publish_event,
        completion_hook=completion_hook,
        same_turn_replace=True,
    )
    try:
        executor.submit(_cmd(1, 'c1'), immediate_non_tts=lambda _c: (True, 'ok'))
        executor.submit(_cmd(2, 'c2'), immediate_non_tts=lambda _c: (True, 'ok'))
        executor.submit(_cmd(2, 'c3'), immediate_non_tts=lambda _c: (True, 'ok'))
        executor.submit(_cmd(3, 'c4'), immediate_non_tts=lambda _c: (True, 'ok'))

        gate['open'] = True
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if len(executed) >= 3:
                break
            time.sleep(0.01)

        assert executed == ['c1', 'c3', 'c4']
        assert ('c2', ExecutionEvent.STATUS_CANCELED, ExecutionEvent.REASON_REPLACED) in events
    finally:
        executor.shutdown()


def test_effect_executor_cancel_pending_tts_clears_queue() -> None:
    events: list[tuple[str, int, int, str]] = []
    gate = {'open': False}

    def invoke_tts(command: OrchestratorCommandData) -> tuple[bool, str]:
        while not gate['open']:
            time.sleep(0.01)
        return True, command.command_id

    def publish_event(command: OrchestratorCommandData, status: int, reason: int, detail: str) -> None:
        events.append((command.command_id, status, reason, detail))

    executor = EffectExecutor(
        invoke_tts=invoke_tts,
        publish_event=publish_event,
        completion_hook=lambda command, ok, detail: None,
        same_turn_replace=True,
    )
    try:
        executor.submit(_cmd(1, 'c1'), immediate_non_tts=lambda _c: (True, 'ok'))
        executor.submit(_cmd(2, 'c2'), immediate_non_tts=lambda _c: (True, 'ok'))
        executor.submit(_cmd(3, 'c3'), immediate_non_tts=lambda _c: (True, 'ok'))

        canceled = executor.cancel_pending_tts(detail='barge_in_pending_tts')
        gate['open'] = True

        assert [command.command_id for command in canceled] == ['c2', 'c3']
        assert ('c2', ExecutionEvent.STATUS_CANCELED, ExecutionEvent.REASON_REPLACED, 'barge_in_pending_tts') in events
        assert ('c3', ExecutionEvent.STATUS_CANCELED, ExecutionEvent.REASON_REPLACED, 'barge_in_pending_tts') in events
    finally:
        executor.shutdown()
