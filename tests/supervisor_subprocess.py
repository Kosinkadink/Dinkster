"""Retain child-side diagnostics before supervisor CLI timeout cleanup."""

from __future__ import annotations

import faulthandler
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

_WRAPPER = f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"


def _process_state(pid: int) -> tuple[str, int | None]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32: Any = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return "unavailable", None
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return "unavailable", None
            return ("alive", None) if code.value == 259 else ("exited", code.value)
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
        code = record.get("returncode")
        state, code = ("exited", code) if code is not None else _process_state(record["pid"])
        stack = path.with_suffix(".stack")
        saved = stack.read_text() if stack.exists() else ""
        rows.append(
            f"Nested CLI child pid={record['pid']}, module={record['module']}, "
            f"state before cleanup={state}, returncode before cleanup={code}\n"
            f"Nested child stack before cleanup:\n"
            f"{saved or '(no nested child stack recorded)'}"
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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    with TemporaryDirectory(prefix="dinkster-supervisor-trace-") as directory:
        trace = Path(directory) / "child.stack"
        with subprocess.Popen(
            [command[0], "-c", _WRAPPER, "supervisor", directory, *command[2:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONFAULTHANDLER": "1"},
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired as exc:
                # poll must precede kill: an exited child can leave inherited pipes open.
                returncode = process.poll()
                stack = trace.read_text() if trace.exists() else "child trace file absent"
                exc.cmd = command
                exc.add_note(
                    f"Supervisor CLI child pid={process.pid}, "
                    f"returncode before cleanup={returncode}\n"
                    f"Child stack before cleanup:\n{stack or '(no child stack recorded)'}\n"
                    f"{_nested_diagnostics(Path(directory))}"
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
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _trace_nested_commands(directory: Path) -> None:
    class TracedPopen(subprocess.Popen):
        def __init__(self, args, *positional, **keywords):
            self.trace_path: Path | None = None
            module = None
            if isinstance(args, (list, tuple)) and "-m" in args:
                index = args.index("-m")
                module = args[index + 1]
                if module == "dinkster.generation_packs" or (
                    module == "dinkster.manager"
                    and args[index + 2 : index + 3] == ["prepare-catalogs"]
                ):
                    args = [
                        *args[:index],
                        "-c",
                        _WRAPPER,
                        "nested",
                        str(directory),
                        *args[index + 1 :],
                    ]
                else:
                    module = None
            super().__init__(args, *positional, **keywords)
            if module is not None:
                self.trace_path = directory / f"nested-{self.pid}.json"
                temporary = self.trace_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"pid": self.pid, "module": module}))
                temporary.replace(self.trace_path)

        def _record_exit(self, code):
            if self.trace_path is not None and code is not None:
                record = json.loads(self.trace_path.read_text())
                record["returncode"] = code
                temporary = self.trace_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record))
                temporary.replace(self.trace_path)

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
    directory = Path(location)
    trace_path = directory / (
        "child.stack" if mode == "supervisor" else f"nested-{os.getpid()}.stack"
    )
    if mode == "supervisor":
        _trace_nested_commands(directory)
    sys.argv = [module, *arguments]
    with _open_trace(trace_path) as trace:
        faulthandler.enable(file=trace)
        faulthandler.dump_traceback_later(
            25 if mode == "supervisor" else 5, repeat=True, file=trace
        )
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()


if __name__ == "__main__":
    _main()
