"""OS-owned descendants of a dedicated supervisor, independent of engine handshakes."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path
from types import TracebackType


class ProcessFamily:
    """Used only by the dedicated supervisor process, never an embedded engine host."""

    def __init__(self) -> None:
        self.proven = sys.platform == "linux" or os.name == "nt"
        self._previous = ctypes.c_int()
        self._libc = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
        self._job: int | None = None

    def __enter__(self) -> ProcessFamily:
        if sys.platform == "linux":
            assert self._libc is not None
            prctl = self._libc.prctl
            prctl.restype = ctypes.c_int
            if prctl(37, ctypes.byref(self._previous), 0, 0, 0) != 0 or prctl(36, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "cannot establish child subreaper ownership")
        if os.name == "nt":
            assert self._kernel is not None
            kernel = self._kernel
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.QueryInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel.QueryInformationJobObject.restype = wintypes.BOOL
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.IsProcessInJob.argtypes = [
                wintypes.HANDLE,
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.BOOL),
            ]
            kernel.IsProcessInJob.restype = wintypes.BOOL
            kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel.TerminateProcess.restype = wintypes.BOOL
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            self._job = kernel.CreateJobObjectW(None, None)
            if not self._job:
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.AssignProcessToJobObject(self._job, kernel.GetCurrentProcess()):
                error = ctypes.get_last_error()
                kernel.CloseHandle(self._job)
                self._job = None
                raise ctypes.WinError(error)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if sys.platform == "linux":
            assert self._libc is not None
            if self._libc.prctl(36, self._previous.value, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "cannot restore child subreaper ownership")
        if os.name == "nt" and self._job is not None:
            assert self._kernel is not None
            self._kernel.CloseHandle(self._job)
            self._job = None

    def stop_children(self) -> None:
        """After engine exit, stop and reap only descendants still owned by this supervisor."""
        deadline = time.monotonic() + 10
        if sys.platform == "linux":
            import signal

            while True:
                try:
                    while os.waitpid(-1, os.WNOHANG)[0] != 0:
                        pass
                except ChildProcessError:
                    return
                children: set[int] = set()
                for task in Path("/proc/self/task").iterdir():
                    try:
                        children.update(int(pid) for pid in (task / "children").read_text().split())
                    except FileNotFoundError:
                        continue
                for pid in children:
                    try:
                        # An unreaped child cannot have its PID reused between this check and kill.
                        if os.waitpid(pid, os.WNOHANG)[0] == 0:
                            os.kill(
                                pid,
                                signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL,
                            )
                    except (ChildProcessError, ProcessLookupError):
                        continue
                if time.monotonic() > deadline + 10:
                    raise RuntimeError("owned descendants did not exit; cleanup remains unproven")
                time.sleep(0.05)
        if os.name == "nt":
            assert self._kernel is not None
            kernel = self._kernel
            size = 1024
            while True:
                buffer = ctypes.create_string_buffer(size)
                returned = wintypes.DWORD()
                if not kernel.QueryInformationJobObject(
                    self._job, 3, buffer, size, ctypes.byref(returned)
                ):
                    if ctypes.get_last_error() == 234:
                        size = max(size * 2, returned.value)
                        continue
                    raise ctypes.WinError(ctypes.get_last_error())
                assigned = wintypes.DWORD.from_buffer(buffer).value
                count = wintypes.DWORD.from_buffer(buffer, 4).value
                if count > (size - 8) // ctypes.sizeof(ctypes.c_size_t) or count > assigned:
                    raise RuntimeError("invalid owned process list")
                pids = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                if assigned == 1 and list(pids) == [os.getpid()]:
                    return
                for pid in pids:
                    if pid == os.getpid():
                        continue
                    handle = kernel.OpenProcess(0x100000 | 0x1000 | 1, False, pid)
                    if not handle:
                        if ctypes.get_last_error() == 87:
                            continue
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        member = wintypes.BOOL()
                        if not kernel.IsProcessInJob(handle, self._job, ctypes.byref(member)):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if not member.value:
                            continue
                        if kernel.WaitForSingleObject(handle, 0) != 0:
                            if not kernel.TerminateProcess(handle, 0):
                                error = ctypes.get_last_error()
                                # A member can exit after the probe; the wait still proves its exit.
                                if error != 5:
                                    raise ctypes.WinError(error)
                        if kernel.WaitForSingleObject(handle, 5000) != 0:
                            raise RuntimeError("owned child exit is unproven")
                    finally:
                        kernel.CloseHandle(handle)
                if time.monotonic() > deadline + 10:
                    raise RuntimeError("owned descendants did not exit; cleanup remains unproven")
                time.sleep(0.05)
