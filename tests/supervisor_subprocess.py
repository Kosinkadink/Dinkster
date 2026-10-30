"""Retain child-side diagnostics before supervisor CLI timeout cleanup."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    with TemporaryDirectory(prefix="dinkster-supervisor-trace-") as directory:
        trace = Path(directory) / "child.stack"
        script = (
            "import faulthandler, runpy; "
            f"trace = open({str(trace)!r}, 'w'); "
            "faulthandler.enable(file=trace); "
            "faulthandler.dump_traceback_later(25, repeat=True, file=trace); "
            f"runpy.run_module({command[2]!r}, run_name='__main__', alter_sys=True)"
        )
        with subprocess.Popen(
            [command[0], "-c", script, *command[3:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
                    f"Child stack before cleanup:\n{stack or '(no child stack recorded)'}"
                )
                process.kill()
                if os.name == "nt":
                    stdout, stderr = process.communicate()
                    # Windows subprocess.run retains text despite the bytes-only exception stubs.
                    setattr(exc, "stdout", stdout)
                    setattr(exc, "stderr", stderr)
                else:
                    process.wait()
                raise
            except BaseException:
                process.kill()
                raise
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
