"""The umbrella `dinkster` CLI: one entry point fronting every console script.

DESIGN 5 (doctor exposure plan): `dinkster doctor` is the main-CLI spelling
of the publish gate. Every subcommand here fronts a standalone
`dinkster-*` console script, and every standalone script stays installed
as an alias - same modules, same flags, same exit codes, two doors.
Named projects and installed system generations select their own supervisor;
source-development commands retain their standalone entry points.

Subcommand modules import lazily: `dinkster doctor` must not pay for (or
fail on) the server stack, and vice versa. Two delegation shapes exist
because the underlying mains do: argv-style mains (doctor, port) take
an argument list and return an exit code; the rest parse sys.argv
internally, so the dispatcher rebinds sys.argv around the call with
`dinkster <name>` as argv[0] (argparse derives usage prose from it).
"""

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

    include_command: bool = False
    """Pass the command name to a module that owns several subcommands."""


_COMMANDS: dict[str, _Command] = {
    **{
        name: _Command("dinkster.engine_cli", True, summary, include_command=True)
        for name, summary in {
            "install": "install an engine from a verified mirror feed",
            "activate": "activate a staged engine generation",
            "generations": "list installed system generations",
            "rollback": "restore the previously active system generation",
            "gc": "preview or remove unreferenced engine content",
            "project": "create and list independent project install roots",
        }.items()
    },
    "doctor": _Command(
        "dinkster_workers.doctor",
        passes_argv=True,
        summary="validate a pack: the publish gate (alias: dinkster-doctor)",
    ),
    "serve": _Command(
        "dinkster.serve",
        passes_argv=False,
        summary="run a Dinkster server (alias: dinkster-serve)",
    ),
    "setup": _Command(
        "dinkster.setup",
        passes_argv=True,
        summary="prepare the local installation used by the default launcher",
    ),
    "pack": _Command(
        "dinkster.manager",
        passes_argv=False,
        summary="manage a Dinkster install root (alias: dinkster-pack)",
    ),
    "installs": _Command(
        "dinkster_supervisor.install_manager",
        passes_argv=False,
        summary="manage the station's install registry (alias: dinkster-installs)",
    ),
    "port": _Command(
        "dinkster.port",
        passes_argv=True,
        summary="generate a native pack skeleton from a legacy pack (alias: dinkster-port)",
    ),
    "p2p-diagnostics": _Command(
        "dinkster.p2p_diagnostics",
        passes_argv=True,
        summary=(
            "inspect public acquisition receipts and P2P grants (alias: dinkster-p2p-diagnostics)"
        ),
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
    if args and args[0] == "--project":
        if len(args) < 3:
            print("usage: dinkster --project NAME <command> [args...]", file=sys.stderr)
            return 2
        _, project, head, *tail = args
        return int(
            importlib.import_module("dinkster.engine_cli").main([head, "--project", project, *tail])
        )
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
    if head == "serve":
        from .setup import default_roots

        if any(
            arg in {"--root", "--project"} or arg.startswith(("--root=", "--project="))
            for arg in tail
        ):
            return int(importlib.import_module("dinkster.engine_cli").main(args))
        _, root = default_roots()
        if (root / "current").is_file():
            from .installer import Installer

            installer = Installer(root)
            number = installer.current_number()
            if number is not None and installer.environment_of(number) is not None:
                return int(importlib.import_module("dinkster.engine_cli").main(args))
    entry = importlib.import_module(command.module).main
    if command.passes_argv:
        return int(entry(args if command.include_command else tail))
    saved = sys.argv
    sys.argv = [f"dinkster {head}", *tail]
    try:
        entry()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
