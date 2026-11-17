"""Pathless IOCP wakeups for contained Windows worker event loops."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import threading
from collections.abc import Callable
from typing import Protocol, cast

if sys.platform == "win32":

    class _Overlapped(Protocol):
        address: int

    class _CompletionAPI(Protocol):
        def Overlapped(self, event: int) -> _Overlapped: ...
        def PostQueuedCompletionStatus(
            self, port: int, size: int, key: int, address: int
        ) -> None: ...

    _completion = cast(_CompletionAPI, importlib.import_module("_overlapped"))

    class _ContainedProactor(asyncio.IocpProactor):
        _iocp: int
        _cache: dict[
            int, tuple[asyncio.Future[None], _Overlapped, None, Callable[[int, int, object], None]]
        ]

        def post_wakeup(
            self, loop: asyncio.AbstractEventLoop, completed: Callable[[int, int, object], None]
        ) -> None:
            overlapped = _completion.Overlapped(0)
            future: asyncio.Future[None] = loop.create_future()
            # Keep OVERLAPPED alive until the normal IOCP dispatcher consumes
            # its completion, including when close cancels the future first.
            self._cache[overlapped.address] = (future, overlapped, None, completed)
            try:
                _completion.PostQueuedCompletionStatus(self._iocp, 0, 0, overlapped.address)
            except OSError as exc:
                self._cache.pop(overlapped.address)
                raise OSError(
                    exc.errno, f"Windows IOCP PostQueuedCompletionStatus failed: {exc}"
                ) from exc

    class ContainedLoop(asyncio.ProactorEventLoop):
        def __init__(self) -> None:
            # BaseProactorEventLoop installs a socket self-pipe and signal fd.
            # Workers receive cancellation over their inherited transport instead.
            asyncio.BaseEventLoop.__init__(self)
            self._wakeup_lock = threading.Lock()
            self._wakeup_pending = False
            self._wakeup_closed = False
            self._proactor = _ContainedProactor()
            self._selector = self._proactor
            self._self_reading_future = None
            self._accept_futures: dict[int, asyncio.Future[object]] = {}
            self._proactor.set_loop(self)

        def _loop_self_reading(self) -> None:
            # ProactorEventLoop schedules this on every run; IOCP needs no reader.
            pass

        def _write_to_self(self) -> None:
            with self._wakeup_lock:
                if self._wakeup_closed or self._wakeup_pending:
                    return
                proactor = self._proactor
                if proactor is None:
                    return

                def completed(_size: int, _key: int, _operation: object) -> None:
                    with self._wakeup_lock:
                        self._wakeup_pending = False

                self._wakeup_pending = True
                try:
                    proactor.post_wakeup(self, completed)
                except OSError:
                    self._wakeup_pending = False
                    raise

        def close(self) -> None:
            if self.is_running():
                raise RuntimeError("Cannot close a running event loop")
            if self.is_closed():
                return
            with self._wakeup_lock:
                self._wakeup_closed = True
            for future in self._accept_futures.values():
                future.cancel()
            self._accept_futures.clear()
            if self._proactor is not None:
                self._proactor.close()
            self._proactor = None
            self._selector = None
            asyncio.BaseEventLoop.close(self)

    class ContainedPolicy(asyncio.WindowsProactorEventLoopPolicy):
        _loop_factory = ContainedLoop


def install_worker_loop_policy() -> None:
    """Pack-owned HTTP threads also create loops; every worker loop must avoid TCP."""
    if sys.platform == "win32":
        # Pack libraries call asyncio.run() without accepting a loop factory.
        asyncio.set_event_loop_policy(ContainedPolicy())  # pyright: ignore[reportDeprecated]


def worker_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return ContainedLoop()
    return asyncio.new_event_loop()


def check_wakeup() -> None:
    """Wake a blocked production loop from another thread, including after reuse."""
    loop = worker_loop()
    received: list[bool] = []
    expired: list[bool] = []
    errors: list[Exception] = []

    def timeout() -> None:
        expired.append(True)
        loop.stop()

    def delivered() -> None:
        received.append(True)
        loop.stop()

    def send() -> None:
        try:
            loop.call_soon_threadsafe(delivered)
        except (OSError, RuntimeError) as exc:
            errors.append(exc)

    try:
        for _ in range(2):
            received.clear()
            watchdog = loop.call_later(2, timeout)
            sender = threading.Timer(0.02, send)
            sender.start()
            try:
                loop.run_forever()
            finally:
                watchdog.cancel()
                sender.join()
            if errors:
                raise errors[0]
            if not received or expired:
                raise OSError("Windows IOCP wakeup did not dispatch the cross-thread callback")
    finally:
        loop.close()


if __name__ == "__main__":
    try:
        check_wakeup()
    except (OSError, RuntimeError) as exc:
        print(json.dumps({"format": "dinkster.windows-wakeup/1", "ok": False, "detail": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps({"format": "dinkster.windows-wakeup/1", "ok": True}))
