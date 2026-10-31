"""Identity-checked local supervisor processes for environment engines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .environment_process import cancel_pending_process, prepare_process, process_state
from .environment_state import EnvironmentError, Generation, atomic_write, interpreter, read_current

__all__ = [
    "engine_assignments",
    "engine_status",
    "preserve_stopped_engine",
    "require_idle",
    "start_engine",
    "stop_engine",
    "validate_assignment",
]

_ENGINE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_GPU_ID = re.compile(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+)\Z")
_STATES = frozenset({"starting", "ready", "failed", "stopped"})
_STARTUP_TIMEOUT = 30.0
_POLL_INTERVAL = 0.05
_STARTUP_LOG_TAIL_BYTES = 16 * 1024


def start_engine(
    root: Path,
    generation: int,
    name: str,
    port: int,
    gpus: tuple[str, ...] = (),
) -> dict[str, object]:
    """Start one generation through an identity-bound loopback supervisor."""
    root = Path(root)
    validate_assignment(generation, name, port, gpus)
    target = root / "generations" / str(generation)
    Generation.read(target / "generation.json")
    python = interpreter(target)
    if not python.is_file():
        raise EnvironmentError(f"generation {generation} has no interpreter: {python}")

    previous = _load_record(root, name, generation)
    if previous is not None:
        existing = _inspect(previous)
        if existing["state"] == "ready":
            if _assignment(existing) == (generation, port, gpus):
                if existing["pid"] != previous["pid"]:
                    _write_record(root, name, existing)
                return _public_row(existing)
            raise EnvironmentError("engine assignment changed; stop it first")
        if existing["state"] != "stopped":
            raise EnvironmentError("engine record is live but not safely reachable; stop it first")

    instance = secrets.token_urlsafe(24)
    token = secrets.token_urlsafe(32)
    lifecycle = _process_directory(_runtime_path(root, name, generation), instance)
    prepare_process(lifecycle, instance)
    url = f"http://127.0.0.1:{port}"
    record: dict[str, object] = {
        "name": name,
        "generation": generation,
        "port": port,
        "gpus": list(gpus),
        "instance": instance,
        "shutdownToken": token,
        "pid": 0,
        "url": url,
        "state": "starting",
        "_processState": lifecycle,
    }
    _write_record(root, name, record)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.supervisor",
        f"--instance={instance}",
        f"--shutdown-token={token}",
        "--process-state",
        str(lifecycle),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--",
        str(python),
        "-m",
        "dinkster.serve",
        "--generation",
        str(target),
        "--library-root",
        environment.get("DINKSTER_LIBRARY_ROOT", str(root / "dinkster-library")),
        "--prepare-stale-catalogs",
        "--allow-mount-changes",
    ]
    log_path = _runtime_path(root, name, generation).with_suffix(".log")
    with log_path.open("ab") as log:
        if os.name != "nt":
            log_path.chmod(0o600)
        try:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=log)
        except OSError:
            record["state"] = "stopped"
            _write_record(root, name, record)
            raise
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        inspected = _inspect(record)
        # A Python launcher may forward to a supervisor with a different PID.
        if _identity_matches(inspected) and inspected["pid"] != record["pid"]:
            record = inspected
            _write_record(root, name, record)
        if inspected["state"] == "ready" and _health_is_ready(inspected):
            return _public_row(inspected)
        if inspected["state"] == "failed" and _identity_matches(inspected):
            stop_engine(root, name, generation=generation)
            raise EnvironmentError(_startup_failure(name, log_path, inspected))
        if process.poll() is not None:
            record["state"] = "failed"
            _write_record(root, name, record)
            raise EnvironmentError(
                _startup_failure(name, log_path, record, exit_code=process.returncode)
            )
        time.sleep(_POLL_INTERVAL)
    stop_engine(root, name, generation=generation)
    raise EnvironmentError(_startup_failure(name, log_path, record, timed_out=True))


def _startup_failure(
    name: str,
    log_path: Path,
    record: dict[str, object],
    *,
    exit_code: int | None = None,
    timed_out: bool = False,
) -> str:
    """Preserve the supervisor's startup cause with the child log tail."""
    if timed_out:
        cause = "did not become ready"
    else:
        detail = record.get("_startupDetail")
        if isinstance(detail, str) and detail:
            cause = f"failed before becoming ready: {detail}"
        elif exit_code is not None:
            cause = f"exited before becoming ready with exit code {exit_code}"
        else:
            cause = "failed before becoming ready"
    try:
        with log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - _STARTUP_LOG_TAIL_BYTES))
            tail = log.read().decode("utf-8", errors="replace").rstrip()
    except OSError as exc:
        tail = f"unavailable: {exc}"
    return f"engine {name} {cause}\nstartup log {log_path}:\n{tail}"


def stop_engine(root: Path, name: str, *, generation: int | None = None) -> None:
    """Stop only the supervisor whose live status proves its recorded identity."""
    _validate_name(name)
    root = Path(root)
    selected = generation if generation is not None else read_current(root)
    if selected is None:
        raise EnvironmentError("project has no active generation")
    record = _load_record(root, name, selected)
    if record is None:
        return
    lifecycle = cast("Path", record["_processState"])
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while True:
        claimed = lifecycle.is_dir() and cancel_pending_process(
            lifecycle, cast("str", record["instance"])
        )
        inspected = _inspect(record)
        if inspected["state"] == "stopped":
            return
        if _identity_matches(inspected):
            break
        if not claimed or time.monotonic() >= deadline:
            raise EnvironmentError(
                "engine identity is unreachable or mismatched; refusing to stop it"
            )
        time.sleep(_POLL_INTERVAL)
    headers = {
        "X-Dinkster-Instance": cast("str", inspected["instance"]),
        "X-Dinkster-Shutdown-Token": cast("str", inspected["shutdownToken"]),
    }
    try:
        shutdown_url = cast("str", inspected["url"]) + "/supervisor/shutdown"
        _request(shutdown_url, method="POST", headers=headers)
    except (HTTPError, URLError, ValueError):
        # GracefulExit may close the listener before an HTTP response reaches us.
        pass
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if not _process_alive(inspected):
            if (
                lifecycle.is_dir()
                and process_state(
                    lifecycle, cast("str", inspected["instance"]), pid=cast("int", inspected["pid"])
                )
                != "stopped"
            ):
                raise EnvironmentError("supervisor exited without proving child cleanup")
            inspected["state"] = "stopped"
            inspected["pid"] = 0
            _write_record(root, name, inspected)
            return
        time.sleep(_POLL_INTERVAL)
    raise EnvironmentError("authenticated supervisor did not stop")


def engine_status(root: Path) -> tuple[dict[str, object], ...]:
    """Return the selected generation's engines without trusting a reused PID."""
    runtime = Path(root) / "runtime"
    if not runtime.is_dir():
        return ()
    rows: list[dict[str, object]] = []
    selected = read_current(Path(root))
    paths = (
        runtime.glob("*/*.json") if selected is None else (runtime / str(selected)).glob("*.json")
    )
    for path in sorted(paths):
        rows.append(_public_row(_inspect(_parse_record(path))))
    return tuple(rows)


def engine_assignments(root: Path, generation: int) -> list[dict[str, object]]:
    """Snapshot the assignments that a generation transaction must preserve."""
    rows: list[dict[str, object]] = []
    for path in sorted((root / "runtime" / str(generation)).glob("*.json")):
        record = _inspect(_parse_record(path))
        if record["state"] not in {"ready", "stopped"}:
            raise EnvironmentError("cannot preserve an ambiguous engine assignment")
        rows.append(
            {key: record[key] for key in ("name", "port", "gpus")}
            | {
                "running": record["state"] == "ready",
            }
        )
    return rows


def preserve_stopped_engine(root: Path, name: str, previous: int, target: int, port: int) -> None:
    """Carry a stopped binding forward without launching it during activation."""
    record = _load_record(root, name, previous)
    existing = _load_record(root, name, target)
    if record is None or _inspect(record)["state"] != "stopped":
        raise EnvironmentError("previous binding is not safely stopped")
    if existing is not None and _inspect(existing)["state"] != "stopped":
        raise EnvironmentError("target binding is not safely stopped")
    _write_record(
        root,
        name,
        record
        | {
            "generation": target,
            "port": port,
            "url": f"http://127.0.0.1:{port}",
            "pid": 0,
            "state": "stopped",
        },
    )


def require_idle(root: Path) -> None:
    """Fail closed unless every live ready engine reports an empty queue."""
    runtime = Path(root) / "runtime"
    if not runtime.is_dir():
        return
    for path in sorted(runtime.glob("*/*.json")):
        record = _parse_record(path)
        inspected = _inspect(record)
        if inspected["state"] == "stopped":
            continue
        if not _identity_matches(inspected) or inspected["state"] != "ready":
            raise EnvironmentError("cannot verify an engine is idle")
        try:
            queue = _request(cast("str", inspected["url"]) + "/api/queue")
        except (HTTPError, URLError, ValueError) as exc:
            raise EnvironmentError("cannot verify an engine is idle") from exc
        if not isinstance(queue, dict):
            raise EnvironmentError("cannot verify an engine is idle")
        document = cast("dict[str, object]", queue)
        running, queued = document.get("running"), document.get("queued")
        if not isinstance(running, list) or not isinstance(queued, list):
            raise EnvironmentError("cannot verify an engine is idle")
        if running or queued:
            raise EnvironmentError("Finish or cancel the running job before updating.")


def validate_assignment(generation: int, name: str, port: int, gpus: tuple[str, ...]) -> None:
    if type(generation) is not int or generation < 1:
        raise EnvironmentError("generation must be a positive integer")
    _validate_name(name)
    if type(port) is not int or not 1 <= port <= 65535:
        raise EnvironmentError("port must be in 1..65535")
    if any(_GPU_ID.fullmatch(gpu) is None for gpu in gpus):
        raise EnvironmentError("GPU ids must be decimal indexes or GPU UUIDs")


def _validate_name(name: str) -> None:
    if type(name) is not str or _ENGINE_NAME.fullmatch(name) is None:
        raise EnvironmentError("engine name must be a portable engine id")


def _runtime_path(root: Path, name: str, generation: int) -> Path:
    return root / "runtime" / str(generation) / f"{name}.json"


def _process_directory(path: Path, instance: str) -> Path:
    return path.with_suffix(".process") / hashlib.sha256(instance.encode()).hexdigest()


def _load_record(root: Path, name: str, generation: int) -> dict[str, object] | None:
    path = _runtime_path(root, name, generation)
    return None if not path.is_file() else _parse_record(path)


def _parse_record(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"cannot read runtime record {path.name}") from exc
    if not isinstance(raw, dict):
        raise EnvironmentError(f"runtime record {path.name} must be an object")
    record = cast("dict[str, object]", raw)
    name, instance, token, url, state = (
        record.get("name"),
        record.get("instance"),
        record.get("shutdownToken"),
        record.get("url"),
        record.get("state"),
    )
    generation, port, pid, gpus = (
        record.get("generation"),
        record.get("port"),
        record.get("pid"),
        record.get("gpus"),
    )
    if not isinstance(gpus, list):
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    gpu_values = cast("list[object]", gpus)
    if any(not isinstance(gpu, str) for gpu in gpu_values):
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    if type(generation) is not int or not isinstance(name, str) or type(port) is not int:
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    gpu_ids = tuple(cast("str", gpu) for gpu in gpu_values)
    validate_assignment(generation, name, port, gpu_ids)
    if path.stem != name or path.parent.name != str(generation):
        raise EnvironmentError(f"runtime record {path.name} does not match its assignment")
    if (
        not isinstance(instance, str)
        or not instance
        or not isinstance(token, str)
        or not token
        or not isinstance(url, str)
        or url != f"http://127.0.0.1:{port}"
        or state not in _STATES
        or type(pid) is not int
        or pid < 0
    ):
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    record["_processState"] = _process_directory(path, instance)
    return record


def _write_record(root: Path, name: str, record: dict[str, object]) -> None:
    path = _runtime_path(root, name, cast("int", record["generation"]))
    persisted = {key: value for key, value in record.items() if not key.startswith("_")}
    atomic_write(path, json.dumps(persisted, sort_keys=True, ensure_ascii=True) + "\n")
    if os.name != "nt":
        path.chmod(0o600)


def _inspect(record: dict[str, object]) -> dict[str, object]:
    inspected = dict(record)
    inspected["_verifiedIdentity"] = False
    lifecycle = cast("Path", record["_processState"])
    if lifecycle.is_dir():
        state = process_state(
            lifecycle, cast("str", record["instance"]), pid=cast("int", record["pid"])
        )
        if state in {"stopped", "cancelled"}:
            inspected.update(state="stopped", pid=0)
            return inspected
    if inspected["state"] == "stopped":
        if _process_alive(inspected):
            inspected["state"] = "failed"
        return inspected
    if inspected["pid"] != 0 and not _process_alive(inspected):
        # A crashed supervisor does not prove that its engine child exited.
        inspected["state"] = "failed"
        return inspected
    try:
        status = _request(cast("str", inspected["url"]) + "/supervisor/status")
    except (HTTPError, URLError, ValueError):
        inspected["state"] = "failed"
        return inspected
    if not isinstance(status, dict):
        inspected["state"] = "failed"
        return inspected
    status_document = cast("dict[str, object]", status)
    if status_document.get("instance") != inspected["instance"]:
        inspected["state"] = "failed"
        return inspected
    detail = status_document.get("detail")
    if isinstance(detail, str) and detail:
        inspected["_startupDetail"] = detail
    supervisor = status_document.get("supervisor")
    if not isinstance(supervisor, dict):
        inspected["state"] = "failed"
        return inspected
    supervisor_pid = cast("dict[str, object]", supervisor).get("pid")
    if type(supervisor_pid) is not int or supervisor_pid < 1:
        inspected["state"] = "failed"
        return inspected
    recorded_pid = cast("int", inspected["pid"])
    if recorded_pid not in {0, supervisor_pid}:
        inspected["state"] = "failed"
        return inspected
    inspected["pid"] = supervisor_pid
    state = status_document.get("state")
    if state not in _STATES:
        inspected["state"] = "failed"
        return inspected
    inspected["state"] = state
    inspected["_verifiedIdentity"] = True
    return inspected


def _identity_matches(record: dict[str, object]) -> bool:
    return record.get("_verifiedIdentity") is True and _process_alive(record)


def _health_is_ready(record: dict[str, object]) -> bool:
    if not _identity_matches(record):
        return False
    try:
        health = _request(cast("str", record["url"]) + "/api/health")
    except (HTTPError, URLError, ValueError):
        return False
    return isinstance(health, dict) and cast("dict[str, object]", health).get("ok") is True


def _process_alive(record: dict[str, object]) -> bool:
    pid = record.get("pid")
    if type(pid) is not int or pid < 1:
        return False
    if os.name == "nt":
        return _windows_process_alive(pid)
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return False
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise EnvironmentError("cannot determine supervisor process state") from exc
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 - Linux stubs omit it
    kernel32: Any = win_dll("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        get_last_error = getattr(ctypes, "get_last_error")  # noqa: B009 - Linux stubs omit it
        if get_last_error() == 87:
            return False
        raise EnvironmentError("cannot query supervisor process")
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise EnvironmentError("cannot query supervisor exit code")
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _assignment(record: dict[str, object]) -> tuple[int, int, tuple[str, ...]]:
    return (
        cast("int", record["generation"]),
        cast("int", record["port"]),
        tuple(cast("list[str]", record["gpus"])),
    )


def _public_row(record: dict[str, object]) -> dict[str, object]:
    return {key: record[key] for key in ("name", "generation", "state", "url")}


def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None) -> object:
    request = Request(url, method=method, headers=headers or {})
    # Supervisor identity and shutdown credentials must never traverse an ambient proxy.
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        # urllib wraps connection timeouts, but response-body reads can raise directly.
        raise URLError(exc) from exc
