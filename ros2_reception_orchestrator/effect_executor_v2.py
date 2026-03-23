from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Callable

from reception_interfaces.msg import ExecutionCommand
from reception_interfaces.msg import ExecutionEvent

from .v2_types import OrchestratorCommandData


TtsInvoke = Callable[[OrchestratorCommandData], tuple[bool, str]]
PublishEvent = Callable[[OrchestratorCommandData, int, int, str], None]
CompletionHook = Callable[[OrchestratorCommandData, bool, str], None]


@dataclass(slots=True)
class _PendingTts:
    command: OrchestratorCommandData


class EffectExecutor:
    """Execute side effects with deterministic TTS queue policy."""

    def __init__(
        self,
        *,
        invoke_tts: TtsInvoke,
        publish_event: PublishEvent,
        completion_hook: CompletionHook,
        same_turn_replace: bool = True,
    ) -> None:
        self._invoke_tts = invoke_tts
        self._publish_event = publish_event
        self._completion_hook = completion_hook
        self._same_turn_replace = bool(same_turn_replace)
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        self._tts_busy = False
        self._tts_queue: list[_PendingTts] = []

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def is_tts_active(self) -> bool:
        with self._lock:
            return self._tts_busy

    def cancel_pending_tts(self, *, detail: str = 'barge_in') -> list[OrchestratorCommandData]:
        with self._lock:
            canceled = [pending.command for pending in self._tts_queue]
            self._tts_queue = []
        for command in canceled:
            self._publish_event(
                command,
                ExecutionEvent.STATUS_CANCELED,
                ExecutionEvent.REASON_REPLACED,
                detail,
            )
        return canceled

    def submit(
        self,
        command: OrchestratorCommandData,
        *,
        immediate_non_tts: Callable[[OrchestratorCommandData], tuple[bool, str]],
    ) -> None:
        if command.command_type != ExecutionCommand.COMMAND_TTS:
            self._publish_event(
                command,
                ExecutionEvent.STATUS_STARTED,
                ExecutionEvent.REASON_NONE,
                'non_tts_started',
            )

            def _run_non_tts() -> None:
                ok, detail = immediate_non_tts(command)
                self._publish_event(
                    command,
                    ExecutionEvent.STATUS_SUCCEEDED if ok else ExecutionEvent.STATUS_FAILED,
                    ExecutionEvent.REASON_NONE if ok else ExecutionEvent.REASON_INTERNAL_ERROR,
                    detail,
                )
                self._completion_hook(command, ok, detail)

            self._pool.submit(_run_non_tts)
            return

        with self._lock:
            if self._tts_busy:
                if self._same_turn_replace and self._tts_queue and self._tts_queue[-1].command.turn_seq == command.turn_seq:
                    replaced = self._tts_queue[-1].command
                    self._tts_queue[-1] = _PendingTts(command=command)
                    self._publish_event(
                        replaced,
                        ExecutionEvent.STATUS_CANCELED,
                        ExecutionEvent.REASON_REPLACED,
                        'same_turn_replaced',
                    )
                else:
                    self._tts_queue.append(_PendingTts(command=command))
                return
            self._tts_busy = True

        self._dispatch_tts(command)

    def _dispatch_tts(self, command: OrchestratorCommandData) -> None:
        self._publish_event(
            command,
            ExecutionEvent.STATUS_STARTED,
            ExecutionEvent.REASON_NONE,
            'tts_started',
        )

        def _run() -> None:
            ok, detail = self._invoke_tts(command)
            self._publish_event(
                command,
                ExecutionEvent.STATUS_SUCCEEDED if ok else ExecutionEvent.STATUS_FAILED,
                ExecutionEvent.REASON_NONE if ok else ExecutionEvent.REASON_INTERNAL_ERROR,
                detail,
            )
            self._completion_hook(command, ok, detail)

            next_command = None
            with self._lock:
                if self._tts_queue:
                    next_command = self._tts_queue.pop(0).command
                else:
                    self._tts_busy = False
            if next_command is not None:
                self._dispatch_tts(next_command)

        self._pool.submit(_run)
