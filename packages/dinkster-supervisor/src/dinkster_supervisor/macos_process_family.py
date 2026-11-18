"""Kernel coalition ownership for supervisors launched as separate launchd jobs."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import plistlib
import signal
import subprocess
import time
from pathlib import Path

from .environment_state import (
    EnvironmentError,
    atomic_write,
    content_digest,
    read_document,
    write_document,
)

RESTART_REQUIRED = "Restart the computer once, then open the environment again."


class _Timeval(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("microseconds", ctypes.c_int)]


def boot_time() -> float:
    """Read the kernel's boot time, not a process-list observation."""
    libc = ctypes.CDLL(None, use_errno=True)
    value = _Timeval()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(b"kern.boottime", ctypes.byref(value), ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), "cannot read kernel boot time")
    if size.value != ctypes.sizeof(value) or value.seconds <= 0:
        raise EnvironmentError("invalid kernel boot time")
    return value.seconds + value.microseconds / 1_000_000


def supersede_stopped_receipt(
    lifecycle: Path, *, publish: bool = False, allow_unpublished: bool = False
) -> bool:
    """Append restart proof without changing an unproven historical receipt."""
    import math

    state = lifecycle / "state.json"
    document = read_document(state)
    recorded: list[float] = []
    for key in ("stoppedAt", "updatedAt", "startedAt"):
        if key not in document:
            continue
        value = document[key]
        if (
            type(value) not in (int, float)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            return False
        recorded.append(float(value))
    if not recorded:
        return False
    stopped = max(recorded)
    boot = boot_time()
    if document.get("state") != "stopped" or stopped >= boot:
        return False
    proof = lifecycle / "restart-proof.json"
    expected = {
        "format": "dinkster.supervisor-restart-proof/1",
        "instance": document["instance"],
        "receiptDigest": content_digest(state),
        "bootTime": boot,
    }
    if proof.is_symlink() or proof.is_junction():
        raise EnvironmentError("restart proof must not be linked")
    if proof.exists():
        previous = read_document(proof)
        prior_boot = previous.get("bootTime")
        return (
            previous.get("format") == expected["format"]
            and previous.get("instance") == expected["instance"]
            and previous.get("receiptDigest") == expected["receiptDigest"]
            and type(prior_boot) in (int, float)
            and isinstance(prior_boot, (int, float))
            and stopped < prior_boot <= boot
        )
    if not publish:
        return allow_unpublished
    write_document(proof, expected)
    return True


def launch_supervisor(
    command: list[str], lifecycle: Path, root: Path, environment: dict[str, str], log: Path
) -> None:
    label = "org.dinkster.supervisor." + hashlib.sha256(str(lifecycle).encode()).hexdigest()
    document = {
        "Label": label,
        "ProgramArguments": command,
        "WorkingDirectory": str(root),
        "EnvironmentVariables": environment | {"DINKSTER_SUPERVISOR_JOB": label},
        "RunAtLoad": True,
        "KeepAlive": False,
        "LimitLoadToSessionType": "Background",
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    }
    path = lifecycle / "launchd.plist"
    atomic_write(path, plistlib.dumps(document))
    path.chmod(0o600)
    uid = getattr(os, "getuid")()  # noqa: B009 - Windows stubs omit this macOS-only API.
    subprocess.run(
        ["/bin/launchctl", "bootstrap", f"user/{uid}", str(path)],
        check=True,
        capture_output=True,
    )


def remove_supervisor(lifecycle: Path) -> None:
    path = lifecycle / "launchd.plist"
    if not path.exists():
        return
    label = "org.dinkster.supervisor." + hashlib.sha256(str(lifecycle).encode()).hexdigest()
    uid = getattr(os, "getuid")()  # noqa: B009 - Windows stubs omit this macOS-only API.
    result = subprocess.run(
        ["/bin/launchctl", "bootout", f"user/{uid}/{label}"], capture_output=True
    )
    if result.returncode not in (0, 3):
        raise EnvironmentError("cannot unload the stopped supervisor job")


class MacOSProcessFamily:
    def __init__(self) -> None:
        self.proven = False
        self._lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.proc_pidinfo.restype = ctypes.c_int
        self._lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.proc_listallpids.restype = ctypes.c_int
        self._lib.proc_signal_with_audittoken.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.proc_signal_with_audittoken.restype = ctypes.c_int
        self._lib.coalition_info_resource_usage.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.coalition_info_resource_usage.restype = ctypes.c_int
        self._coalition = self._coalition_id(os.getpid())
        job = os.environ.get("DINKSTER_SUPERVISOR_JOB")
        if not job or os.environ.get("XPC_SERVICE_NAME") != job or not self._coalition:
            raise EnvironmentError("macOS supervisor requires its own launchd job")
        if self._coalition_id(os.getppid()) == self._coalition or self._active_tasks() != 1:
            raise EnvironmentError("macOS supervisor coalition is not exclusive")

    def _info(self, pid: int, flavor: int, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        if self._lib.proc_pidinfo(pid, flavor, 0, buffer, size) != size:
            raise OSError(ctypes.get_errno(), "cannot inspect coalition member")
        return buffer.raw

    def _coalition_id(self, pid: int) -> int:
        # PROC_PIDCOALITIONINFO: resource and jetsam coalition IDs.
        return int.from_bytes(self._info(pid, 20, 40)[:8], "little")

    def _active_tasks(self) -> int:
        # The leading fields are tasks_started/tasks_exited; request the full ABI buffer.
        usage = (ctypes.c_uint64 * 256)()
        if self._lib.coalition_info_resource_usage(self._coalition, usage, ctypes.sizeof(usage)):
            raise OSError(ctypes.get_errno(), "cannot account for owned tasks")
        if usage[0] < usage[1]:
            raise EnvironmentError("invalid coalition task accounting")
        return usage[0] - usage[1]

    def stop_children(self) -> None:
        self.proven = False
        deadline = time.monotonic() + 10
        sigkill = getattr(signal, "SIGKILL")  # noqa: B009 - Windows stubs omit this POSIX signal.
        while True:
            if time.monotonic() > deadline + 10:
                raise EnvironmentError("owned descendants did not exit; cleanup remains unproven")
            if self._active_tasks() == 1:
                # The supervisor is still alive and the only task. No descendant can fork now.
                self.proven = True
                return
            size = self._lib.proc_listallpids(None, 0) + 128
            if size <= 128:
                raise EnvironmentError("cannot enumerate owned tasks")
            pids = (ctypes.c_int * size)()
            count = self._lib.proc_listallpids(pids, ctypes.sizeof(pids))
            if count <= 0 or count >= size:
                continue
            for pid in pids[:count]:
                if pid <= 0 or pid == os.getpid():
                    continue
                try:
                    # Read identity before membership; the audit token prevents PID-reuse kills.
                    identity = self._info(pid, 17, 56)
                    if self._coalition_id(pid) != self._coalition:
                        continue
                    if self._info(pid, 17, 56) != identity:
                        continue
                except OSError:
                    # Uninspectable tasks still count, so they cannot establish exit.
                    continue
                token = (ctypes.c_uint32 * 8)(
                    0, 0, 0, 0, 0, pid, 0, int.from_bytes(identity[32:36], "little")
                )
                error = self._lib.proc_signal_with_audittoken(
                    ctypes.byref(token),
                    signal.SIGTERM if time.monotonic() < deadline else sigkill,
                )
                if error not in (0, errno.ESRCH):
                    raise OSError(error, "cannot stop owned task")
            time.sleep(0.05)
