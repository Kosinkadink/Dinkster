"""Lazy public commands, with generation lifecycle operations owned by launch."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class _Command:
    module: str
    """Import path of the module carrying the script's main()."""

    passes_argv: bool
    """True: main(argv) -> int. False: main() -> None over sys.argv."""

    summary: str
    """One usage line, mirroring the script's own argparse description."""


_COMMANDS: dict[str, _Command] = {
    "doctor": _Command(
        "dinkster_workers.doctor",
        passes_argv=True,
        summary="validate a pack: the publish gate (alias: dinkster-doctor)",
    ),
    "serve": _Command(
        "dinkster.launch",
        passes_argv=True,
        summary="start the active managed generation (alias: dinkster-serve)",
    ),
    "run": _Command("dinkster.launch", True, "run a command in the active generation"),
    "stop": _Command("dinkster.launch", True, "stop the active generation's engine"),
    "update": _Command("dinkster.launch", True, "plan and apply a generation update"),
    "rollback": _Command("dinkster.launch", True, "restore the previous software snapshot"),
    "setup": _Command(
        "dinkster.setup",
        passes_argv=True,
        summary="plan and build a locked managed generation",
    ),
    "develop": _Command(
        "dinkster.development",
        passes_argv=True,
        summary="build explicit local sources with unlocked development dependencies",
    ),
    "pack": _Command(
        "dinkster.manager",
        passes_argv=False,
        summary="manage a Dinkster install root (alias: dinkster-pack)",
    ),
    "workflow": _Command(
        "dinkster.workflow_requirements",
        passes_argv=True,
        summary="print a read-only plan for saved workflow requirements",
    ),
    "installs": _Command(
        "dinkster_supervisor.install_manager",
        passes_argv=False,
        summary="manage the station's install registry (alias: dinkster-installs)",
    ),
    "demo": _Command(
        "dinkster.demo",
        passes_argv=False,
        summary="run the local demo (previously bare `dinkster`)",
    ),
    "isolated-demo": _Command(
        "dinkster.isolated_demo",
        passes_argv=False,
        summary="run the isolated-worker demo (alias: dinkster-isolated)",
    ),
}


def _usage() -> str:
    width = max(len(name) for name in _COMMANDS)
    lines = [
        "usage: dinkster <command> [args...]",
        "",
        "commands:",
        *(f"  {name.ljust(width)}  {command.summary}" for name, command in _COMMANDS.items()),
        "",
        "run `dinkster <command> --help` for that command's own flags",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("-"):
        return int(importlib.import_module("dinkster.launch").main(args))
    head, *tail = args
    if head == "help":
        print(_usage())
        return 0
    command = _COMMANDS.get(head)
    if command is None:
        print(f"dinkster: unknown command {head!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    entry = importlib.import_module(command.module).main
    if command.module == "dinkster.launch":
        return int(entry(args))
    if command.passes_argv:
        return int(entry(tail))
    saved = sys.argv
    sys.argv = [f"dinkster {head}", *tail]
    try:
        entry()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
