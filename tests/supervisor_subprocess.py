"""Retain child-side diagnostics before supervisor CLI timeout cleanup."""

from __future__ import annotations

import faulthandler
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType
from typing import Any, cast
from uuid import uuid4

_WRAPPER = f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"
operation_reports: list[str] = []


@contextmanager
def _trace_operations(path: Path):
    # Diagnostic parents can use a base interpreter without supervisor packages.
    if importlib.util.find_spec("dinkster_supervisor") is None:
        yield
        return
    from dinkster_supervisor import (
        environment_archives,
        environment_artifacts,
        environment_downloads,
        environment_generations,
        environment_move,
        environment_runtime,
        environment_state,
        environment_transactions,
    )

    operations = {
        function.__code__: f"{module.__name__}.{name}"
        for module, names in (
            (environment_archives, ("extract_base", "_verify_frame", "_validate_staging")),
            (environment_artifacts, ("acquire", "acquire_base")),
            (environment_downloads, ("fetch_artifact",)),
            (
                environment_generations,
                (
                    "prepare_engine_interpreter",
                    "build_generation",
                    "stage_pack_sources",
                    "prepare_pack_catalogs",
                    "pack_operation",
                ),
            ),
            (environment_move, ("execute_move",)),
            (environment_runtime, ("start_engine", "stop_engine", "_request")),
            (environment_state, ("atomic_write",)),
            (environment_transactions, ("activate",)),
        )
        for name in names
        if (function := getattr(module, name))
    }
    operations[subprocess.run.__code__] = "subprocess.run"
    operations[cast(FunctionType, shutil.rmtree).__code__] = "shutil.rmtree"
    monitoring = sys.monitoring
    tool = monitoring.PROFILER_ID
    monitoring.use_tool_id(tool, "supervisor operation timings")
    with _open_trace(path) as trace:
        origin = time.monotonic()

        def record(event, code):
            trace.write(f"{time.monotonic() - origin:.6f}s {event} {operations[code]}\n")
            trace.flush()

        def start(code, offset):
            record("start", code)

        def complete(code, offset, result):
            record("return", code)

        def failed(code, offset, error):
            if code in operations:
                record("unwind", code)

        callbacks = {
            monitoring.events.PY_START: start,
            monitoring.events.PY_RETURN: complete,
            monitoring.events.PY_UNWIND: failed,
        }
        try:
            for event, callback in callbacks.items():
                monitoring.register_callback(tool, event, callback)
            monitoring.set_events(tool, monitoring.events.PY_UNWIND)
            for code in operations:
                monitoring.set_local_events(
                    tool, code, monitoring.events.PY_START | monitoring.events.PY_RETURN
                )
            yield
        finally:
            monitoring.set_events(tool, 0)
            for code in operations:
                monitoring.set_local_events(tool, code, 0)
            for event in callbacks:
                monitoring.register_callback(tool, event, None)
            monitoring.free_tool_id(tool)


def _retain_operations(directory: Path, elapsed: float) -> str:
    path = directory / "child.operations"
    trace = path.read_text() if path.exists() else "operation trace absent\n"
    report = (
        f"{os.environ.get('PYTEST_CURRENT_TEST', 'outside pytest')}: "
        f"supervisor child wait {elapsed:.6f}s\n"
        "Operation timestamps are relative to tracing startup; nested intervals overlap.\n" + trace
    )
    operation_reports.append(report)
    return report


def _process_state(pid: int) -> tuple[str, int | None]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32: Any = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000 | 0x100000, False, pid)
        if not handle:
            return "unavailable", None
        try:
            # An exited process may legitimately return STILL_ACTIVE (259).
            state = kernel32.WaitForSingleObject(handle, 0)
            if state == 258:
                return "alive", None
            if state != 0:
                return "unavailable", None
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return "unavailable", None
            return "exited", code.value
        finally:
            kernel32.CloseHandle(handle)
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return "exited", os.waitstatus_to_exitcode(int(stat[49]))
    except FileNotFoundError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited", None
    return "alive", None


def _nested_diagnostics(directory: Path) -> str:
    rows = []
    for path in sorted(directory.glob("nested-*.json")):
        record = json.loads(path.read_text())
        pid = None
        for suffix in (".pid", ".launcher"):
            pid_path = path.with_suffix(suffix)
            if pid_path.exists():
                pid = int(pid_path.read_text())
                break
        exit_path = path.with_suffix(".exit")
        code = int(exit_path.read_text()) if exit_path.exists() else None
        state, code = (
            ("exited", code)
            if code is not None
            else ("starting", None)
            if pid is None
            else _process_state(pid)
        )
        stack = path.with_suffix(".stack")
        saved = stack.read_text() if stack.exists() else ""
        rows.append(
            f"Nested CLI child pid={pid}, module={record['module']}, "
            f"state before cleanup={state}, returncode before cleanup={code}\n"
            f"Nested child stack before cleanup:\n"
            f"{saved or '(no nested child stack recorded)'}"
        )
    for path in sorted(directory.glob("probe-*.json")):
        record = json.loads(path.read_text())
        pid = record.get("pid")
        code = record.get("returncode")
        state, code = (
            ("exited", code)
            if code is not None
            else ("starting", None)
            if pid is None
            else _process_state(pid)
        )
        lifecycle = {
            key: record[key] - record["started"]
            for key in ("spawned", "observed_exit")
            if key in record
        }
        rows.append(
            f"Unwrapped interpreter probe pid={pid}, "
            f"state before cleanup={state}, returncode before cleanup={code}\n"
            f"Probe lifecycle (seconds since launch started): {json.dumps(lifecycle)}\n"
            f"Probe observed after {time.monotonic() - record['started']:.6f}s"
        )
    return "\n".join(rows)


def _open_trace(path: Path):
    if os.name != "nt":
        return path.open("w")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32: Any = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    # Allow the owning helper to remove its temporary directory even if a nested child lives on.
    handle = kernel32.CreateFileW(str(path), 0x40000000, 7, None, 2, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError("cannot open diagnostic trace")
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_WRONLY), "w")


def _request_snapshot(directory: Path) -> str:
    # The CLI has already timed out; this grace only collects diagnostics before cleanup.
    (directory / "snapshot.request").touch()
    deadline = time.monotonic() + 1
    while True:
        targets = {directory / "child.snapshot"} | {
            path.with_suffix(".snapshot") for path in directory.glob("nested-*.json")
        }
        pending = {path for path in targets if not path.exists()}
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    return (
        "Timeout snapshot not acknowledged (exited or unresponsive): "
        + ", ".join(sorted(path.stem for path in pending))
        if pending
        else "Timeout snapshot acknowledged"
    )


def run(command: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    with TemporaryDirectory(prefix="dinkster-supervisor-trace-") as directory:
        trace = Path(directory) / "child.stack"
        started = time.monotonic()
        with subprocess.Popen(
            [
                command[0],
                "-c",
                _WRAPPER,
                "supervisor",
                directory,
                *(command[1:] if command[1] == "-c" else command[2:]),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONFAULTHANDLER": "1"},
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                elapsed = time.monotonic() - started
                # poll must precede kill: an exited child can leave inherited pipes open.
                returncode = process.poll()
                snapshot = _request_snapshot(Path(directory))
                stack = trace.read_text() if trace.exists() else "child trace file absent"
                exc.cmd = command
                exc.add_note(
                    f"Supervisor CLI child pid={process.pid}, "
                    f"returncode before cleanup={returncode}\n"
                    f"{snapshot}\n"
                    f"Child stack before cleanup:\n{stack or '(no child stack recorded)'}\n"
                    f"{_nested_diagnostics(Path(directory))}\n"
                    f"{_retain_operations(Path(directory), elapsed)}"
                )
                process.kill()
                if os.name == "nt":
                    # Windows subprocess.run retains text despite the bytes-only exception stubs.
                    cast(Any, exc).stdout, cast(Any, exc).stderr = process.communicate()
                else:
                    process.wait()
                raise
            except BaseException:
                process.kill()
                raise
            if process.returncode:
                for stack in sorted(Path(directory).glob("*.stack")):
                    saved = stack.read_text()
                    if "Fatal Python error:" in saved or "Windows fatal exception:" in saved:
                        stderr += f"\nFatal child diagnostics ({stack.stem}):\n{saved}"
            _retain_operations(Path(directory), time.monotonic() - started)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _trace_nested_commands(directory: Path) -> None:
    class TracedPopen(subprocess.Popen):
        def __init__(self, args, *positional, **keywords):
            self.trace_path: Path | None = None
            self.probe_path: Path | None = None
            self.probe_record: dict[str, Any] = {}
            module = None
            if isinstance(args, (list, tuple)) and tuple(args[1:4]) == ("-I", "-B", "-c"):
                # Keep the actual interpreter and inline source untouched.
                self.probe_path = directory / f"probe-{uuid4().hex}.json"
                self.probe_record = {"started": time.monotonic()}
                self._save_probe()
            if isinstance(args, (list, tuple)) and "-m" in args:
                index = args.index("-m")
                module = args[index + 1]
                if module == "dinkster.generation_packs" or (
                    module == "dinkster.manager"
                    and args[index + 2 : index + 5] == ["--root", "", "prepare-catalogs"]
                ):
                    self.trace_path = directory / f"nested-{uuid4().hex}.json"
                    args = [
                        *args[:index],
                        "-c",
                        _WRAPPER,
                        "nested",
                        str(self.trace_path),
                        *args[index + 1 :],
                    ]
                else:
                    module = None
            if self.trace_path is not None:
                # Register before spawning so timeout collection includes children still starting.
                temporary = self.trace_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"module": module}))
                temporary.replace(self.trace_path)
            super().__init__(args, *positional, **keywords)
            if self.probe_path is not None:
                self.probe_record.update(pid=self.pid, spawned=time.monotonic())
                self._save_probe()
            if self.trace_path is not None:
                temporary = self.trace_path.with_suffix(".tmp")
                temporary.write_text(str(self.pid))
                temporary.replace(self.trace_path.with_suffix(".launcher"))

        def _save_probe(self):
            if self.probe_path is not None:
                temporary = self.probe_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.probe_record))
                temporary.replace(self.probe_path)

        def _record_exit(self, code):
            if self.probe_path is not None and code is not None:
                self.probe_record.update(returncode=code, observed_exit=time.monotonic())
                self._save_probe()
                self.probe_path = None
            if self.trace_path is not None and code is not None:
                temporary = self.trace_path.with_suffix(".tmp")
                temporary.write_text(str(code))
                temporary.replace(self.trace_path.with_suffix(".exit"))
                self.trace_path = None

        def poll(self):
            code = super().poll()
            self._record_exit(code)
            return code

        def wait(self, timeout=None):
            code = super().wait(timeout=timeout)
            self._record_exit(code)
            return code

    subprocess.Popen = TracedPopen


def _main() -> None:
    mode, location, module, *arguments = sys.argv[1:]
    if mode == "supervisor":
        trace_path = Path(location) / "child.stack"
        _trace_nested_commands(Path(location))
    else:
        trace_path = Path(location).with_suffix(".stack")
        # A Windows virtualenv launcher has a different PID from its interpreter.
        temporary = Path(location).with_suffix(".pid.tmp")
        temporary.write_text(str(os.getpid()))
        temporary.replace(Path(location).with_suffix(".pid"))
    sys.argv = [module, *(arguments[1:] if module == "-c" else arguments)]
    operations = (
        _trace_operations(trace_path.with_suffix(".operations"))
        if mode == "supervisor"
        else nullcontext()
    )
    with _open_trace(trace_path) as trace, operations:
        faulthandler.enable(file=trace)
        finished = threading.Event()
        interval = 25 if mode == "supervisor" else 5
        request = trace_path.parent / "snapshot.request"
        acknowledgment = trace_path.with_suffix(".snapshot")

        def observe() -> None:
            next_sample = time.monotonic() + interval
            while not finished.wait(0.05):
                requested = request.exists() and not acknowledgment.exists()
                if not requested and time.monotonic() < next_sample:
                    continue
                # The asynchronous walker can race freed frames on CPython 3.12.
                # An attached Python thread holds the GIL while dumping stacks.
                trace.write(
                    "Requested timeout snapshot\n"
                    if requested
                    else f"Timeout (0:00:{interval:02})!\n"
                )
                trace.flush()
                faulthandler.dump_traceback(file=trace, all_threads=True)
                if requested:
                    acknowledgment.touch()
                next_sample = time.monotonic() + interval

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            if module == "-c":
                exec(arguments[0], {"__name__": "__main__"})
            else:
                runpy.run_module(module, run_name="__main__", alter_sys=True)
        finally:
            finished.set()
            observer.join()
            faulthandler.enable()


if __name__ == "__main__":
    _main()
