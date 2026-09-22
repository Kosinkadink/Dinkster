"""Engine installation and project selection above generation-owned supervisors."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dinkster_registry import InstallError

from . import projects
from .engine_feed import EngineFeedError, Mirror
from .engine_install import EngineInstaller, native_cell
from .installer import Installer
from .setup import default_roots


def _nonempty_instance(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("instance must not be empty")
    return value


def _generation(installer: Installer, number: int) -> dict[str, object]:
    environment = installer.environment_of(number)
    result: dict[str, object] = {
        "generation": number,
        "current": installer.current_number() == number,
        "root": str(installer.root),
        "engine": environment.record() if environment else None,
    }
    if environment is not None:
        control, execution = EngineInstaller(installer).interpreters(number)
        result.update(controlPython=str(control), executionPython=str(execution))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dinkster", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "activate", "generations", "rollback", "gc", "serve"):
        command = commands.add_parser(name)
        roots = command.add_mutually_exclusive_group()
        roots.add_argument("--root", type=Path)
        roots.add_argument("--project")
        command.add_argument("--json", action="store_true")
        if name == "install":
            command.add_argument("--mirror", default=os.environ.get("DINKSTER_ENGINE_MIRROR"))
            command.add_argument("--channel", choices=("stable", "github-live"), default="stable")
            command.add_argument("--cell")
            command.add_argument("--allow-local-http", action="store_true")
            command.add_argument("--shared-store", type=Path)
            command.add_argument("--stage-only", action="store_true")
        elif name == "activate":
            command.add_argument("--generation", type=int, required=True)
        elif name == "gc":
            command.add_argument(
                "--apply", action="store_true", help="delete previewed engine data"
            )
        elif name == "serve":
            command.add_argument("--data-root", type=Path)
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=3639)
            command.add_argument("--instance", type=_nonempty_instance)
    project = commands.add_parser("project")
    project_commands = project.add_subparsers(dest="action", required=True)
    create = project_commands.add_parser("create")
    create.add_argument("name")
    create.add_argument("--data-root", type=Path)
    create.add_argument("--json", action="store_true")
    project_commands.add_parser("list").add_argument("--json", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if remaining and args.command != "serve":
        parser.error("unrecognized arguments: " + " ".join(remaining))
    try:
        if args.command == "project":
            result = (
                projects.create(args.name, args.data_root)
                if args.action == "create"
                else projects.list_projects()
            )
        else:
            library, root = default_roots()
            if args.project:
                record = projects.read(args.project)
                root, library = Path(record["root"]), Path(record["dataRoot"])
            elif args.root:
                root = args.root.expanduser().resolve()
            if args.command == "serve":
                if not 1 <= args.port <= 65535:
                    raise InstallError("port must be in 1..65535")
                return projects.serve(
                    root,
                    args.data_root or library,
                    host=args.host,
                    port=args.port,
                    instance=args.instance,
                    args=tuple(remaining),
                )
            installer = Installer(
                root,
                shared_store=args.shared_store if args.command == "install" else None,
            )
            engine = EngineInstaller(installer)
            if args.command == "install":
                if not args.mirror:
                    raise InstallError("set --mirror or DINKSTER_ENGINE_MIRROR")
                number = engine.install(
                    Mirror(args.mirror, allow_local_http=args.allow_local_http),
                    channel=args.channel,
                    cell=args.cell or native_cell(),
                    activate=not args.stage_only,
                )
                result = _generation(installer, number)
            elif args.command == "activate":
                engine.activate(args.generation)
                result = _generation(installer, args.generation)
            elif args.command == "rollback":
                result = _generation(installer, engine.rollback())
            elif args.command == "generations":
                result = [
                    _generation(installer, number) for number in installer.generation_numbers()
                ]
            else:
                candidates = engine.gc() if args.apply else engine.gc_candidates()
                result = {"deleted": args.apply, "paths": [str(path) for path in candidates]}
        print(json.dumps(result, sort_keys=True, indent=None if args.json else 2))
        return 0
    except (
        InstallError,
        EngineFeedError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        print(f"dinkster: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
