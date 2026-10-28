"""Shared command-line boundary for environment management and Desktop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from .environment_artifacts import discover
from .environment_generations import pack_operation
from .environment_runtime import (
    engine_assignments,
    engine_status,
    start_engine,
    stop_engine,
    validate_assignment,
)
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    read_current,
    read_document,
    writer_lock,
)
from .environment_transactions import activate, begin_activation, recover_activation


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


def _confirmed(args: argparse.Namespace, plan: list[str]) -> bool:
    if args.yes:
        return True
    if not args.plan and not args.json and sys.stdin.isatty():
        print("\n".join(plan), file=sys.stderr)
        return input("Apply this plan? [y/N] ").strip().lower() in {"y", "yes"}
    return False


def _recover_pending(root: Path) -> None:
    for path in sorted((root / "journal").glob("*.json")):
        if Journal.read(path).incomplete:
            recover_activation(root, path)


def _rollback(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    current = read_current(root)
    settings = _settings(root)
    if current is None or settings is None:
        raise EnvironmentError("project has no active generation")
    snapshot = (root / "project.json").read_bytes()
    target = None
    for path in sorted((root / "journal").glob("*.json"), reverse=True):
        journal = Journal.read(path)
        if journal.stage == "complete" and journal.target == current and journal.previous:
            target = journal.previous
            break
    if target is None:
        raise EnvironmentError("project has no previous successful generation")
    plan = [f"reactivate retained generation {target} without rebuilding generation {current}"]
    result.update(current=current, plan=plan, status="planned")
    if not _confirmed(args, plan):
        return 0

    def revalidate() -> None:
        if read_current(root) != current or (root / "project.json").read_bytes() != snapshot:
            raise EnvironmentError("project changed after planning; request a new plan")

    with writer_lock(root):
        _recover_pending(root)
        revalidate()
        path = begin_activation(root, target, "rollback")
        activate(root, path, revalidate=revalidate)
        result.update(current=target, status="applied", engines=list(engine_status(root)))
    return 0


def _runtime(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    settings = _settings(root)
    current = read_current(root)
    result["current"] = current
    if settings is None or current is None:
        raise EnvironmentError("project has no active generation; run setup first")
    snapshot = (root / "project.json").read_bytes()
    generation = Generation.read(root / "generations" / str(current) / "generation.json")
    if generation.number != current:
        raise EnvironmentError("generation directory and recorded number disagree")
    port = args.port if args.operation == "start" else None
    if port is None:
        if args.operation == "start" and args.engine != "main":
            raise EnvironmentError("an additional engine binding requires --port")
        port = cast("int", settings["port"])
        if args.operation == "start":
            for binding in engine_assignments(root, current):
                if binding["name"] == args.engine:
                    port = cast("int", binding["port"])
                    break
    gpus = tuple(args.gpu) if args.operation == "start" else ()
    validate_assignment(current, args.engine, port, gpus)
    plan = [f"{args.operation} engine {args.engine} in generation {current}"]
    if args.operation == "start":
        plan.append(f"listen on 127.0.0.1:{port}; CUDA_VISIBLE_DEVICES={','.join(gpus)}")
    result.update(plan=plan, engines=list(engine_status(root)))
    if not _confirmed(args, plan):
        result["status"] = "planned"
        return 0
    with writer_lock(root):
        _recover_pending(root)
        if (
            read_current(root) != current
            or (root / "project.json").read_bytes() != snapshot
            or _settings(root) != settings
        ):
            raise EnvironmentError("project changed after planning; request a new plan")
        if args.operation == "start":
            row = start_engine(root, current, args.engine, port, gpus)
            result["url"] = row["url"]
        else:
            stop_engine(root, args.engine)
        result.update(status="applied", engines=list(engine_status(root)))
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
    start = commands.add_parser("start", help="start the current generation's assigned engine")
    start.add_argument("--port", type=int)
    start.add_argument("--gpu", action="append", default=[])
    stop = commands.add_parser("stop", help="stop one identity-checked supervisor and engine")
    rollback = commands.add_parser("rollback", help="reactivate the previous successful generation")
    for command in (start, stop):
        command.add_argument("--engine", default="main")
    for command in (start, stop, rollback):
        confirmation = command.add_mutually_exclusive_group()
        confirmation.add_argument("--yes", action="store_true", help="apply the printed plan")
        confirmation.add_argument("--plan", action="store_true", help="inspect without mutation")
    for command in (catalog, status, doctor, start, stop, rollback):
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
        if args.operation == "rollback":
            code = _rollback(args, result)
        elif args.operation in {"start", "stop"}:
            code = _runtime(args, result)
        else:
            code = _query(args, result)
    except (OSError, ValueError) as exc:
        code = 1
        result.update(status="error", error=str(exc))
    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    else:
        print(f"{result['operation']}: {result['status']}")
        for step in cast("list[str]", result["plan"]):
            print(step)
        for change in cast("list[str]", result.get("changes", [])):
            print(change)
        if "error" in result:
            print(result["error"], file=sys.stderr)
        if args.operation == "catalog" and code == 0:
            print(json.dumps(result["variants"], indent=2, ensure_ascii=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
