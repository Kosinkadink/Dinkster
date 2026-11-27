"""Run installation operations in Desktop's explicitly configured private WSL host."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from .environment_state import EnvironmentError

_UNC = re.compile(r"^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(?:\\(.*))?$", re.IGNORECASE)


@dataclass(frozen=True)
class WslContext:
    distribution: str
    python: str

    def guest_path(self, value: str) -> str:
        match = _UNC.fullmatch(value)
        if match is None or match[1].lower() != self.distribution.lower():
            raise EnvironmentError("path does not belong to Desktop's WSL host")
        parts = (match[2] or "").split("\\")
        if any(part in {".", ".."} or re.search(r"[/:\x00-\x1f]", part) for part in parts):
            raise EnvironmentError("invalid WSL installation path")
        return "/" + "/".join(part for part in parts if part)

    def windows_path(self, value: str) -> Path:
        if not value.startswith("/") or "\\" in value or ".." in value.split("/"):
            raise EnvironmentError("invalid guest installation path")
        return Path(f"\\\\wsl.localhost\\{self.distribution}" + value.replace("/", "\\"))

    def command(self, module: str, *arguments: str) -> list[str]:
        return [
            "wsl.exe",
            "--distribution",
            self.distribution,
            "--user",
            "dinkster",
            "--cd",
            "/home/dinkster",
            "--exec",
            "/usr/bin/env",
            "-i",
            "HOME=/home/dinkster",
            f"PATH={PurePosixPath(self.python).parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            f"DINKSTER_SERVICES={os.environ.get('DINKSTER_SERVICES', 'production')}",
            self.python,
            "-I",
            "-m",
            module,
            *(self.guest_path(value) if _UNC.fullmatch(value) else value for value in arguments),
        ]

    def call(self, root: Path, operation: str, *arguments: object) -> object:
        try:
            result = subprocess.run(
                self.command("dinkster_supervisor.home_context"),
                input=json.dumps(
                    {
                        "distribution": self.distribution,
                        "root": self.guest_path(str(root)),
                        "operation": operation,
                        "arguments": arguments,
                    }
                ),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EnvironmentError("WSL installation operation did not finish") from error
        if result.returncode:
            raise EnvironmentError(result.stderr.strip() or "WSL installation operation failed")
        answer: object = json.loads(result.stdout)
        if not isinstance(answer, dict):
            raise EnvironmentError("invalid WSL installation answer")
        answer = cast("dict[str, object]", answer)
        if "error" in answer:
            raise EnvironmentError(str(answer["error"]))
        if "result" not in answer:
            raise EnvironmentError("missing WSL installation result")
        return answer["result"]

    def environment_command(self, *arguments: str) -> list[str]:
        return self.command(
            "dinkster_supervisor.home_context", "--environment-owner", self.distribution, *arguments
        )


def configured_wsl() -> WslContext | None:
    config = os.environ.get("DINKSTER_DESKTOP_WSL_CONTEXT")
    if sys.platform != "win32" or not config or not Path(config).exists():
        return None
    value = json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EnvironmentError("invalid Desktop WSL context")
    value = cast("dict[str, object]", value)
    distribution, python = value.get("distribution"), value.get("python")
    if (
        not isinstance(distribution, str)
        or not re.fullmatch(r"Dinkster-Desktop-[a-f0-9]{12}", distribution)
        or not isinstance(python, str)
        or not re.fullmatch(
            r"/home/dinkster/tools/[a-f0-9]{64}/[A-Za-z0-9._+-]+/payload/python/bin/python3", python
        )
    ):
        raise EnvironmentError("invalid Desktop WSL context")
    return WslContext(distribution, python)


def installation_context(root: Path) -> WslContext | None:
    if _UNC.fullmatch(str(root)) is None:
        return None
    context = configured_wsl()
    if context is None:
        raise EnvironmentError("Desktop has not prepared this WSL execution host")
    context.guest_path(str(root))
    return context


def resolve_installation_root(root: Path) -> Path:
    context = installation_context(root)
    return (
        root.resolve()
        if context is None
        else context.windows_path(cast("str", context.call(root, "resolve")))
    )


def _dispatch(root: Path, operation: str, arguments: list[object]) -> object:
    from . import environment_runtime as runtime
    from . import environment_sandbox as sandbox
    from . import home, home_frontend, home_setup
    from .environment_state import read_current

    if operation == "installations":
        return [str(path) for path in home.known_installations()]
    if operation == "setup-facts":
        channel, variant, cancelled = arguments
        return home_setup.setup_facts(
            root, cast("str", channel), cast("str | None", variant), cancelled=bool(cancelled)
        )
    if operation == "setup-refusal":
        return home_setup.library_refusal()
    if operation == "cancelled-setup":
        return home_setup.cancelled_setup_identity(root)
    if operation == "resolve":
        return str(root.resolve())
    if operation == "port":
        import socket

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]
    if operation == "progress":
        identifier, action = arguments
        if not isinstance(identifier, str) or not re.fullmatch("[a-f0-9]{32}", identifier):
            raise EnvironmentError("invalid setup job")
        from .environment_state import dinkster_home, read_document

        progress = dinkster_home() / "setup-jobs" / f"{identifier}.json"
        if action == "read":
            return read_document(progress)
        if action == "cancel":
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.with_name(progress.name + ".cancel").touch()
            return None
        if action == "clean":
            progress.unlink(missing_ok=True)
            progress.with_name(progress.name + ".cancel").unlink(missing_ok=True)
            return None
        raise EnvironmentError("invalid setup progress operation")
    if operation == "project-installation":
        project, roots = arguments
        found = home.project_installation(
            cast("str", project), frozenset(Path(value) for value in cast("list[str]", roots))
        )
        return str(found) if found is not None else None
    root = root.resolve()
    if root not in home.known_installations():
        raise EnvironmentError("not an installation Dinkster knows")
    if operation == "current":
        return read_current(root)
    if operation == "library-root":
        from .environment_state import library_root

        return library_root(root)
    if operation == "status":
        return runtime.engine_status(root)
    if operation == "stop":
        runtime.stop_engine(
            root, cast("str", arguments[0]), generation=cast("int | None", arguments[1])
        )
        return None
    if operation == "log":
        return runtime.engine_log_tail(root, cast("str", arguments[0]))
    if operation == "packs":
        return runtime.installation_pack_metadata(
            root, cast("str", arguments[0]), snapshot=cast("int | None", arguments[1])
        )
    if operation == "settings":
        return sandbox.read_sandbox_settings(root)
    if operation == "switch":
        return sandbox.set_sandbox_switch(root, cast("bool", arguments[0]))
    if operation == "readiness":
        return runtime.sandbox_readiness(root, cast("int", arguments[0]))
    if operation == "internet":
        pack, address, requested, snapshot, allow = arguments
        return sandbox.change_internet_address(
            root,
            cast("str", pack),
            cast("str", address),
            cast("list[str]", requested),
            cast("str", snapshot),
            allow=bool(allow),
        )
    if operation == "node-table":
        return home.recorded_node_table(home.OpenTarget(root, cast("str", arguments[0])))
    if operation == "frontend":
        found = home_frontend.installation_frontend(root, cast("str", arguments[0]))
        return {
            "wire": found.wire,
            "selected": found.selected,
            "source": None if found.source is None else found.source.wire(),
        }
    raise EnvironmentError("unknown WSL installation operation")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] in {"--environment-owner", "--library-owner"}:
        if Path("/etc/dinkster-desktop-owner").read_text().strip() != sys.argv[2]:
            raise EnvironmentError("WSL distribution is not owned by Desktop")
        context = WslContext(sys.argv[2], sys.executable)
        arguments = [
            context.guest_path(value) if _UNC.fullmatch(value) else value for value in sys.argv[3:]
        ]
        module = (
            "dinkster_supervisor.environment"
            if sys.argv[1] == "--environment-owner"
            else "dinkster_library"
        )
        os.execv(
            sys.executable,
            [sys.executable, "-I", "-m", module, *arguments],
        )
    try:
        request = json.load(sys.stdin)
        distribution = request["distribution"]
        if Path("/etc/dinkster-desktop-owner").read_text().strip() != distribution:
            raise EnvironmentError("WSL distribution is not owned by Desktop")
        root = Path(request["root"])
        if not root.is_absolute():
            raise EnvironmentError("guest installation root must be absolute")
        with contextlib.redirect_stdout(sys.stderr):
            result = _dispatch(root, request["operation"], request["arguments"])
        print(json.dumps({"result": result}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}))


if __name__ == "__main__":
    main()
