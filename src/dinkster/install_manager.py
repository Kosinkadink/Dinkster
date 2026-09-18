"""dinkster-installs: manage the station's install registry.

The station (dinkster-station, in dinkster-supervisor) READS ``installs.toml``
and deliberately imports no engine code; this CLI is the WRITER, and it
lives in the umbrella because registering an install is an operator
action on the machine, not a fleet-runtime action. The split keeps one
format (both sides go through dinkster_supervisor.installs) and one
direction of knowledge: the umbrella knows about the supervisor package,
never the reverse.

Config mutations follow the plan/apply discipline every other mutating
Dinkster command follows: print exactly what would change, then apply only
on explicit confirmation (interactive yes or --yes). Registering an
install does NOT create or provision its root - environments are built
with 'dinkster-pack --root <root> install ...' (plan/apply there too);
this file is just the map of what the station should manage.

Runtime actions (start/stop/restart) talk to a RUNNING station's
management port - they are process control, not environment mutation,
so they take no plan step; the station answers with the resulting state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from dinkster_supervisor import (
    InstallDef,
    InstallsError,
    dump_station_config,
    load_installs,
    load_station_config,
    parse_station_config,
)
from dinkster_supervisor.installs import DEFAULT_INSTALLS_FILE

__all__ = ["main"]


def _config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    env = os.environ.get("DINKSTER_INSTALLS", "")
    return Path(env) if env else DEFAULT_INSTALLS_FILE


def _write_config(path: Path, installs: tuple[InstallDef, ...]) -> None:
    """Atomic rewrite: the station never observes a half-written registry."""
    existing = load_station_config(path)
    text = dump_station_config(type(existing)(installs, existing.ingress))
    parse_station_config(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".installs-", suffix=".toml")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _confirmed(args: argparse.Namespace) -> bool:
    """Explicit confirmation or a loud refusal - same contract as
    dinkster-pack: interactive asks, noninteractive requires --yes."""
    if args.yes:
        return True
    if sys.stdin.isatty():
        return input("apply this change? [y/N] ").strip().lower() in ("y", "yes")
    raise InstallsError("refusing to mutate without confirmation: re-run with --yes")


def _describe(install: InstallDef) -> str:
    parts = [f"root={install.root}", f"port={install.port}"]
    if install.autostart:
        parts.append("autostart")
    if install.engine:
        parts.append("engine=" + " ".join(install.engine))
    return f"{install.name}: " + "  ".join(parts)


def _cmd_list(args: argparse.Namespace) -> None:
    path = _config_path(args)
    installs = load_installs(path)
    if not installs:
        print(f"no installs configured ({path})")
        return
    for install in installs:
        print(_describe(install))


def _cmd_show(args: argparse.Namespace) -> None:
    installs = load_installs(_config_path(args))
    for install in installs:
        if install.name == args.name:
            print(_describe(install))
            return
    raise InstallsError(f"no install named {args.name!r}")


def _cmd_add(args: argparse.Namespace) -> None:
    path = _config_path(args)
    installs = load_installs(path)
    if any(entry.name == args.name for entry in installs):
        raise InstallsError(f"install {args.name!r} already exists; remove it first to reconfigure")
    new = InstallDef(
        name=args.name,
        root=Path(args.root).expanduser().resolve(),
        port=args.port,
        autostart=args.autostart,
        engine=tuple(args.engine or ()),
    )
    combined = (*installs, new)
    # Validation (name grammar, port/root collisions) is the parser's,
    # exercised by round-tripping the exact text we would write.
    existing = load_station_config(path)
    parse_station_config(dump_station_config(type(existing)(combined, existing.ingress)), str(path))
    print("plan: add " + _describe(new))
    if not new.root.is_dir():
        print(
            f"note: root {new.root} does not exist yet; provision it with "
            f"'dinkster-pack --root {new.root} install ...'"
        )
    if not _confirmed(args):
        print("plan not applied; nothing changed")
        return
    _write_config(path, combined)
    print(f"added {new.name} to {path}")


def _cmd_remove(args: argparse.Namespace) -> None:
    path = _config_path(args)
    installs = load_installs(path)
    keep = tuple(entry for entry in installs if entry.name != args.name)
    if len(keep) == len(installs):
        raise InstallsError(f"no install named {args.name!r}")
    gone = next(entry for entry in installs if entry.name == args.name)
    existing = load_station_config(path)
    parse_station_config(dump_station_config(type(existing)(keep, existing.ingress)), str(path))
    print("plan: remove " + _describe(gone))
    print(f"note: the install root {gone.root} is NOT deleted - only the registration")
    if not _confirmed(args):
        print("plan not applied; nothing changed")
        return
    _write_config(path, keep)
    print(f"removed {gone.name} from {path}")


def _station_post(station: str, route: str) -> dict[str, object]:
    url = station.rstrip("/") + route
    request = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body: object = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise InstallsError(f"station answered {exc.code} for {route}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise InstallsError(
            f"cannot reach station at {station}: {exc.reason} (is 'dinkster-station' running?)"
        ) from exc
    if not isinstance(body, dict):
        raise InstallsError(f"station answered non-object JSON for {route}")
    return body


def _cmd_runtime(args: argparse.Namespace) -> None:
    body = _station_post(args.station, f"/supervisor/installs/{args.name}/{args.action}")
    state = body.get("state", "unknown")
    print(f"{args.name}: {args.action} -> {state}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dinkster-installs",
        description="Manage the Dinkster station's install registry "
        "(installs.toml) and drive a running station",
    )
    parser.add_argument(
        "--config",
        default="",
        metavar="FILE",
        help=f"installs config (default: $DINKSTER_INSTALLS, else {DEFAULT_INSTALLS_FILE})",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_cmd = commands.add_parser("list", help="show every configured install")
    list_cmd.set_defaults(handler=_cmd_list)

    show = commands.add_parser("show", help="show one install's configuration")
    show.add_argument("name")
    show.set_defaults(handler=_cmd_show)

    add = commands.add_parser("add", help="register an install with the station")
    add.add_argument("name", help="install name (lowercase alphanumerics and hyphens)")
    add.add_argument("--root", required=True, metavar="PATH", help="install root directory")
    add.add_argument("--port", required=True, type=int, metavar="N", help="public supervisor port")
    add.add_argument(
        "--autostart",
        action="store_true",
        help="start this install's engine when the station starts",
    )
    add.add_argument(
        "--engine",
        nargs="+",
        metavar="ARG",
        help="engine command override (interpreter first), e.g. a different "
        "Dinkster version's venv python; default is the station's own "
        "'<python> -m dinkster.serve'",
    )
    add.add_argument(
        "--yes", "-y", action="store_true", help="apply without asking (noninteractive)"
    )
    add.set_defaults(handler=_cmd_add)

    remove = commands.add_parser("remove", help="unregister an install (root is kept)")
    remove.add_argument("name")
    remove.add_argument(
        "--yes", "-y", action="store_true", help="apply without asking (noninteractive)"
    )
    remove.set_defaults(handler=_cmd_remove)

    for action, description in (
        ("start", "start an install's engine via a running station"),
        ("stop", "stop an install's engine via a running station"),
        ("restart", "restart an install's engine via a running station"),
    ):
        runtime = commands.add_parser(action, help=description)
        runtime.add_argument("name")
        runtime.add_argument(
            "--station",
            default=os.environ.get("DINKSTER_STATION", "http://127.0.0.1:3649"),
            metavar="URL",
            help="station management URL (default: $DINKSTER_STATION, else http://127.0.0.1:3649)",
        )
        runtime.set_defaults(handler=_cmd_runtime, action=action)

    args = parser.parse_args()
    try:
        args.handler(args)
    except InstallsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
