"""Diagnostic-only Windows produced-lease probe; never part of a repair."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path


def _write(record: dict[str, object]) -> None:
    folder = Path(os.environ["DINKSTER_LEASE_PROBE_EVIDENCE"])
    with (folder / f"process-{os.getpid()}.jsonl").open("a", encoding="ascii") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")


def _holders(path: str) -> list[int]:
    from ctypes import wintypes

    class UniqueProcess(ctypes.Structure):
        _fields_ = [("pid", wintypes.DWORD), ("started", wintypes.FILETIME)]

    class ProcessInfo(ctypes.Structure):
        _fields_ = [
            ("process", UniqueProcess),
            ("name", wintypes.WCHAR * 256),
            ("service", wintypes.WCHAR * 64),
            ("kind", ctypes.c_int),
            ("status", wintypes.ULONG),
            ("session", wintypes.DWORD),
            ("restartable", wintypes.BOOL),
        ]

    api = ctypes.WinDLL("Rstrtmgr")
    handle = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)
    result = api.RmStartSession(ctypes.byref(handle), 0, key)
    if result:
        raise OSError(result, "RmStartSession")
    try:
        files = (wintypes.LPCWSTR * 1)(path)
        result = api.RmRegisterResources(handle, 1, files, 0, None, 0, None)
        if result:
            raise OSError(result, "RmRegisterResources")
        needed, count, reasons = wintypes.UINT(), wintypes.UINT(), wintypes.DWORD()
        result = api.RmGetList(
            handle, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reasons)
        )
        if result == 0:
            return []
        if result != 234:
            raise OSError(result, "RmGetList size")
        count.value = needed.value
        entries = (ProcessInfo * count.value)()
        result = api.RmGetList(
            handle, ctypes.byref(needed), ctypes.byref(count), entries, ctypes.byref(reasons)
        )
        if result:
            raise OSError(result, "RmGetList processes")
        return [entry.process.pid for entry in entries[: count.value]]
    finally:
        api.RmEndSession(handle)


def _process(pid: int) -> dict[str, object]:
    import psutil

    try:
        process = psutil.Process(pid)
        args = process.cmdline()
        redacted = []
        hide = False
        for argument in args:
            sensitive = any(
                name in argument.lower()
                for name in ("password", "secret", "credential", "api-key", "token")
            )
            redacted.append("[redacted]" if hide or sensitive else argument)
            hide = sensitive and "=" not in argument
        return {"pid": pid, "parent": process.ppid(), "image": process.exe(), "argv": redacted}
    except Exception as error:
        return {"pid": pid, "inspection_error": repr(error)}


def capture_delete_failure(error: PermissionError, service: object) -> None:
    """Observe only after deletion fails, without replacing the original error."""
    try:
        path = str(error.filename)
        try:
            holders = _holders(path)
            holder_error = None
        except Exception as inspection:
            holders = []
            holder_error = repr(inspection)
        record: dict[str, object] = {
            "event": "delete-failed",
            "time_ns": time.time_ns(),
            "path": path,
            "error": repr(error),
            "test_process": _process(os.getpid()),
            "service_pid": getattr(service, "pid", None),
            "service_returncode": getattr(service, "returncode", None),
            "holders": [_process(pid) for pid in holders],
            "holder_inspection_error": holder_error,
        }
        import psutil
        from filelock import BaseFileLock

        pid = getattr(service, "pid", None)
        record["service_pid_exists"] = psutil.pid_exists(pid) if isinstance(pid, int) else None
        record["service_process"] = _process(pid) if isinstance(pid, int) else None
        record["own_open_lease_locks"] = [
            {"path": str(item.lock_file), "locked": item.is_locked}
            for item in gc.get_objects()
            if isinstance(item, BaseFileLock) and item.is_locked
        ]
        _write(record)
        print(json.dumps(record, ensure_ascii=True), file=sys.stderr, flush=True)
    except Exception as inspection:
        print(f"delete failure inspection failed: {inspection!r}", file=sys.stderr, flush=True)


def enable_lease_logging(root: Path) -> None:
    """Instrument only the separate logging arm before any pytest process starts."""
    path = root / "packages/dinkster-assets/src/dinkster_assets/scratch.py"
    text = path.read_text()
    acquire = "                    lock.acquire(timeout=0)\n"
    release = "            self._locks.pop().release()\n"
    if text.count(acquire) != 1 or text.count(release) != 1:
        raise ValueError("lease logging source anchors changed")
    text = text.replace(
        acquire,
        acquire
        + "                    from ._lease_probe import log_lease\n"
        + '                    log_lease("acquired", lock)\n',
    ).replace(
        release,
        "            lock = self._locks.pop()\n"
        "            from ._lease_probe import log_lease\n"
        '            log_lease("releasing", lock)\n'
        "            lock.release()\n"
        '            log_lease("released", lock)\n',
    )
    path.write_text(text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        from filelock import FileLock

        folder = Path(sys.argv[2]).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        os.environ["DINKSTER_LEASE_PROBE_EVIDENCE"] = str(folder)
        lock = FileLock(folder / "owner-selfcheck.lock", preserve_lock_file=True)
        with lock:
            found = _holders(str(folder / "owner-selfcheck.lock"))
            if os.getpid() not in found:
                raise AssertionError(f"Restart Manager missed the known holder: {found}")
            capture_delete_failure(
                PermissionError(13, "synthetic holder self-check", str(lock.lock_file)), None
            )
        saved = json.loads((folder / f"process-{os.getpid()}.jsonl").read_text())
        if not saved["own_open_lease_locks"] or not saved["holders"]:
            raise AssertionError("holder evidence self-check was incomplete")
        print("Windows holder and own-lease evidence self-check passed")
    else:
        enable_lease_logging(Path(__file__).resolve().parents[1])
