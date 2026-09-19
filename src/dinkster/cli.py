"""The umbrella `dinkster` CLI: one entry point fronting every console script.

DESIGN 5 (doctor exposure plan): `dinkster doctor` is the main-CLI spelling
of the publish gate. Every subcommand here fronts a standalone
`dinkster-*` console script, and every standalone script stays installed
as an alias - same modules, same flags, same exit codes, two doors.
The dispatcher adds NO behavior of its own: it resolves a name to the
script's existing main() and gets out of the way, so nothing can drift
between the `dinkster foo` and `dinkster-foo` spellings.

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


_COMMANDS: dict[str, _Command] = {
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
    "pack": _Command(
        "dinkster.manager",
        passes_argv=False,
        summary="manage a Dinkster install root (alias: dinkster-pack)",
    ),
    "installs": _Command(
        "dinkster.install_manager",
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
    if not args:
        print(_usage(), file=sys.stderr)
        return 2
    head, *tail = args
    if head in ("-h", "--help", "help"):
        print(_usage())
        return 0
    command = _COMMANDS.get(head)
    if command is None:
        print(f"dinkster: unknown command {head!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    entry = importlib.import_module(command.module).main
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
