"""Owner-bound read-only preparation state shared by short HTTP requests."""

from __future__ import annotations

import secrets
import threading
from time import monotonic


class PreparationStopped(RuntimeError):
    """Preparation lost its observer before the next work boundary."""


class PackPreparation:
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.identity = secrets.token_hex(32)
        self._lock = threading.Lock()
        self._started = self._updated = monotonic()
        self._state = "preparing"
        self._sequence = 0
        self._progress: dict[str, object] = {
            "sequence": 0,
            "stage": "queued",
            "completed": 0,
            "total": 0,
        }
        self._result: dict[str, object] | None = None
        self._error = ""

    def _check_deadline(self) -> None:
        now = monotonic()
        if self._state == "preparing":
            if now - self._started >= 1800:
                self._state, self._error = "failed", "preparation-expired"
            elif now - self._updated >= 120:
                self._state, self._error = "failed", "preparation-stalled"
        elif self._state == "ready" and now - self._updated >= 900:
            self._state, self._error, self._result = "failed", "plan-stale", None

    def progress(self, stage: str, completed: int, total: int) -> None:
        with self._lock:
            self._check_deadline()
            if self._state != "preparing":
                raise PreparationStopped(self._state)
            self._sequence += 1
            self._progress = {
                "sequence": self._sequence,
                "stage": stage,
                "completed": completed,
                "total": total,
            }
            self._updated = monotonic()

    def complete(self, result: dict[str, object]) -> None:
        with self._lock:
            self._check_deadline()
            if self._state == "preparing":
                self._state, self._result = "ready", result
                self._updated = monotonic()

    def fail(self, error: str) -> None:
        with self._lock:
            if self._state == "preparing":
                self._state, self._error = "failed", error

    def cancel(self) -> None:
        with self._lock:
            if self._state in {"preparing", "ready"}:
                self._state, self._result = "cancelled", None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._check_deadline()
            return {
                "preparation": self.identity,
                "state": self._state,
                "progress": self._progress,
                **({"result": self._result} if self._result is not None else {}),
                **({"error": self._error} if self._error else {}),
            }
