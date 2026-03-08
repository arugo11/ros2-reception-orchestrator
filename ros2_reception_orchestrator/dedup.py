from __future__ import annotations

from collections import deque
import hashlib


class RecentValueDeduplicator:
    """Keep a bounded set of recent values and report whether a value is new."""

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max(1, int(max_size))
        self._queue: deque[str] = deque()
        self._seen: set[str] = set()

    def mark(self, value: str) -> bool:
        key = str(value)
        if key in self._seen:
            return False
        self._queue.append(key)
        self._seen.add(key)
        while len(self._queue) > self._max_size:
            stale = self._queue.popleft()
            self._seen.discard(stale)
        return True


def stable_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()
