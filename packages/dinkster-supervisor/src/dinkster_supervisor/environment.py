"""Shared command-line boundary for environment management and Desktop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from .environment_artifacts import discover
from .environment_generations import pack_operation
from .environment_state import EnvironmentError, Generation, read_current, read_document


def _settings(root: Path) -> dict[str, object] | None:
    path = root / "project.json"
    if not path.exists():
        return None
    document = read_document(path)
    if document.get("format") != "dinkster.project/1":
        raise EnvironmentError("unsupported project format")
    if document.get("channel") not in ("stable", "main"):
        raise EnvironmentError("project channel must be stable or main")
    variant, mirrors, port = document.get("variant"), document.get("mirrors"), document.get("port")
    if not isinstance(variant, str) or not variant:
        raise EnvironmentError("project variant must be a nonempty string")
    if not isinstance(mirrors, list) or not mirrors:
        raise EnvironmentError("project mirrors must be a nonempty array")
    if any(not isinstance(mirror, str) or not mirror for mirror in cast("list[object]", mirrors)):
        raise EnvironmentError("project mirrors must contain nonempty strings")
    if type(port) is not int or not 1 <= port <= 65535:
        raise EnvironmentError("project port must be in 1..65535")
    return document


def _query(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    settings = _settings(root)
    current = read_current(root)
    result["current"] = current
    if args.operation == "catalog":
        mirrors = tuple(args.mirror or (cast("list[str]", settings["mirrors"]) if settings else ()))
        channel = args.channel or (cast("str", settings["channel"]) if settings else "stable")
        catalog = discover(mirrors, channel)
        result.update(
            channel=channel, channelManifest=catalog.document, variants=catalog.variants()
        )
    elif args.operation == "status":
        from .environment_runtime import engine_status

        result.update(
            channel=settings["channel"] if settings else None,
            variant=settings["variant"] if settings else None,
            engines=list(engine_status(root)),
        )
    elif args.operation == "doctor":
        if current is None:
            raise EnvironmentError("project has no active generation")
        generation_root = root / "generations" / str(current)
        generation = Generation.read(generation_root / "generation.json")
        if generation.number != current:
            raise EnvironmentError("generation directory and recorded number disagree")
        changes = list(generation.drift(generation_root))
        if (generation_root / "packs").exists() and not changes:
            store_file = root / "store"
            try:
                store = Path(store_file.read_text(encoding="utf-8").strip())
            except OSError as exc:
                raise EnvironmentError("cannot read project store pointer") from exc
            if not store.is_absolute():
                raise EnvironmentError("project store must be absolute")
            try:
                pack_operation(
                    "doctor",
                    generation=generation_root,
                    store=store,
                    variant=generation.variant,
                    worker_protocol=generation.worker_protocol,
                    own_environment_size=0,
                    in_process_packs=generation.in_process_packs,
                )
            except EnvironmentError as exc:
                changes.append(str(exc))
        result.update(changes=changes, status="drift" if changes else "clean")
        return 1 if changes else 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="absolute prospective project root"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    catalog = commands.add_parser("catalog", help="inspect a channel and intact variant manifests")
    catalog.add_argument("--mirror", action="append", default=[])
    catalog.add_argument("--channel", choices=("stable", "main"))
    status = commands.add_parser(
        "status", help="read configured project and observed engine status"
    )
    doctor = commands.add_parser("doctor", help="report active generation and pack drift")
    for command in (catalog, status, doctor):
        command.add_argument("--json", action="store_true", help="emit exactly one result object")
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    result: dict[str, object] = {
        "format": "dinkster.environment-result/1",
        "operation": args.operation,
        "root": str(args.root),
        "current": None,
        "plan": [],
        "status": "clean",
    }
    try:
        code = _query(args, result)
    except (OSError, ValueError) as exc:
        code = 1
        result.update(status="error", error=str(exc))
    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    else:
        print(f"{result['operation']}: {result['status']}")
        for change in cast("list[str]", result.get("changes", [])):
            print(change)
        if "error" in result:
            print(result["error"], file=sys.stderr)
        if args.operation == "catalog" and code == 0:
            print(json.dumps(result["variants"], indent=2, ensure_ascii=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
