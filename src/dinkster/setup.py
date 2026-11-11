"""Inspect managed setup inputs and initialize local library folders."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine
from typing import cast

from dinkster_assets import MountDef, dump_mounts, load_mounts, load_output_mount
from dinkster_registry import InstallError
from dinkster_workers.provision import ProvisionError

from .installer import Installer
from .project_folders import ProjectFolders


def default_roots() -> tuple[Path, Path]:
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    return home / "library", home / "packs"


def ensure_default_mounts(library_root: Path) -> None:
    """Initialize visible folders once without restoring or widening existing grants."""
    library_root.mkdir(parents=True, exist_ok=True)
    for name in ("output", "imports"):
        _ensure_default_mount(library_root, name)


def _ensure_default_mount(library_root: Path, name: str) -> None:
    initialized = library_root / f".{name}-initialized"
    if initialized.exists():
        return
    mounts_path = library_root / "mounts.toml"
    mounts = load_mounts(mounts_path)
    root = library_root / name
    output_mount = load_output_mount(mounts_path)
    existing = next((mount for mount in mounts if mount.id == name), None)
    if existing is None:
        existing = next((mount for mount in mounts if mount.path == root), None)
    if name == "output" and output_mount is not None and existing is None:
        initialized.touch()
        return
    project_path = library_root / "project-folders.json"
    projects = ProjectFolders(project_path) if project_path.exists() else None
    if existing is not None:
        if projects is not None:
            projects.initialize(tuple(mount.id for mount in mounts), shared_defaults=(existing.id,))
        initialized.touch()
        return
    root.mkdir(parents=True, exist_ok=True)
    text = dump_mounts(
        (*mounts, MountDef(id=name, path=root, mode="readwrite")),
        output_mount=name if name == "output" else output_mount,
    )
    temporary = mounts_path.with_name(f"{mounts_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, mounts_path)
    finally:
        temporary.unlink(missing_ok=True)
    if projects is not None:
        projects.initialize(tuple(mount.id for mount in mounts) + (name,), shared_defaults=(name,))
    initialized.touch()


def _plan(args: argparse.Namespace) -> dict[str, object]:
    from dinkster_workers.pack_environment import fetch_wheel

    from .registries import (
        RegistryConfig,
        StaticPackIndex,
        load_registries,
        plan_registry_sources,
        select_registry,
    )

    artifacts = importlib.import_module("dinkster_supervisor.environment_artifacts")
    state = importlib.import_module("dinkster_supervisor.environment_state")
    stores = importlib.import_module("dinkster_supervisor.environment_stores")
    runtime_module = importlib.import_module("dinkster_supervisor.environment_runtime")
    root = args.root.expanduser().resolve()
    runtime_module.validate_assignment(1, args.engine, args.port, tuple(args.gpu))
    if state.read_current(root) is not None:
        raise state.EnvironmentError("project already has an active generation; use update")
    variant = args.variant
    if variant is None and sys.platform == "darwin" and machine().lower() == "arm64":
        variant = "mac-arm64"
    if variant is None:
        raise state.EnvironmentError("setup requires an explicit --variant")
    store = stores.select_store(root, args.data)
    channel = args.channel or "stable"
    mirrors = tuple(args.mirror or ("https://releases.dinkster.org",))
    catalog = artifacts.discover(mirrors, channel, store=store, offline=args.offline)
    base_id = catalog.channel.base_for(variant)
    base = catalog.bases[variant]
    defaults = catalog.engine.get("defaultPacks")
    if not isinstance(defaults, dict) or not defaults:
        raise state.EnvironmentError("target engine has no default pack declaration")
    config_path = args.registries or root / "registries.toml"
    config = (
        load_registries(config_path)
        if args.registries is not None or config_path.is_file()
        else RegistryConfig()
    )
    selector = args.registry or ("" if config.registries else "https://registry.dinkster.org")
    sources, locks = plan_registry_sources(
        config,
        select_registry(config, selector),
        cast("dict[str, str]", defaults),
        engine_version=cast("str", catalog.engine["version"]),
        variant=variant,
        base_id=base_id,
        index=StaticPackIndex(
            store / "metadata" / "registry", offline=args.offline, read_only=True
        ),
    )
    runtime = artifacts.runtime_inputs(catalog, variant, store)
    payloads = {
        (wheel.sha256, wheel.filename): wheel
        for lock in locks
        for package in lock.packages
        for wheel in package.wheels
    }
    cached_payloads: dict[str, str | None] = {}
    for wheel in payloads.values():
        identity = wheel.sha256 + "/" + wheel.filename
        try:
            cached_payloads[identity] = str(fetch_wheel(wheel, store / "wheels", offline=True))
        except ProvisionError:
            cached_payloads[identity] = None
    plan = [
        f"Project: {root}",
        f"Store: {store}",
        f"Engine binding: {args.engine}; listen on 127.0.0.1:{args.port}; "
        f"CUDA_VISIBLE_DEVICES={','.join(args.gpu)}",
        f"Engine: {catalog.engine['version']} ({catalog.channel.engine_release})",
        f"Runtime: Python {base['python']}; {variant}; base {base_id}",
        *(
            f"Pack: {entry.pack}@{entry.version}; {entry.source}; {entry.artifact_digest}"
            for entry in sources.packs
        ),
        *(
            f"Dependency lock: {lock.pack}; "
            f"sha256:{hashlib.sha256(lock.to_json().encode('ascii')).hexdigest()}; "
            f"Python {lock.requires_python}; engine API {lock.engine or 'exact packages'}"
            for lock in locks
        ),
        *(
            f"Pack payload: {wheel.filename}; sha256:{wheel.sha256}; "
            f"{wheel.size} bytes; {wheel.url}; "
            f"{'cached' if cached_payloads[wheel.sha256 + '/' + wheel.filename] else 'not cached'}"
            for wheel in payloads.values()
        ),
        *(
            f"Runtime payload: {row['path']}; sha256:{row['sha256']}; "
            f"{row['size']} bytes; {'cached' if row['cached'] else 'not cached'}"
            for row in runtime
        ),
        "Missing runtime payload bytes: "
        f"{sum(cast('int', row['size']) for row in runtime if not row['cached'])}",
        f"Pack payload upper bound: {sum(wheel.size for wheel in payloads.values())} bytes; "
        "includes all locked alternatives before target-interpreter selection",
        "No runtime or pack payloads acquired; no generation built or activated",
    ]
    if catalog.cached_channel:
        plan.append("Use the last acquired channel selection")
    return {
        "root": str(root),
        "store": str(store),
        "channel": channel,
        "engineVersion": catalog.engine["version"],
        "engineRelease": catalog.channel.engine_release,
        "baseId": base_id,
        "variant": variant,
        "engine": args.engine,
        "port": args.port,
        "gpus": args.gpu,
        "runtimeInputs": runtime,
        "sources": json.loads(sources.record_json()),
        "dependencyLocks": {lock.pack: json.loads(lock.to_json()) for lock in locks},
        "maximumPackPayloadBytes": sum(wheel.size for wheel in payloads.values()),
        "cachedPackPayloads": cached_payloads,
        "plan": plan,
    }


def _apply(args: argparse.Namespace, approved: dict[str, object]) -> int:
    from dinkster_registry import Lockfile
    from dinkster_store import pack_archive_path
    from dinkster_workers.pack_environment import fetch_wheel
    from dinkster_workers.pack_lock import PackLock

    from .registries import registry_source_lock

    observed = _plan(args)
    for key in ("engineRelease", "baseId", "sources", "dependencyLocks"):
        if observed[key] != approved[key]:
            raise InstallError("setup inputs changed after planning; request a new plan")
    if args.offline:
        raise InstallError("offline setup apply is not available; --offline --plan changes nothing")
    store = Path(cast("str", approved["store"]))
    sources = Lockfile.from_record_json(json.dumps(approved["sources"]))
    locks = {
        name: PackLock.from_json(json.dumps(document))
        for name, document in cast("dict[str, object]", approved["dependencyLocks"]).items()
    }
    with tempfile.TemporaryDirectory(prefix="dinkster-setup-") as directory:
        staging = Path(directory)
        archives = staging / "artifacts"
        archives.mkdir()
        admitted = []
        for entry in sources.packs:
            wheel = next(
                package for package in locks[entry.pack].packages if package.name == entry.pack
            )
            payload = fetch_wheel(wheel.wheels[0], store / "wheels")
            archive = pack_archive_path(archives, entry.artifact_digest)
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(payload, archive)
            admitted.append(
                registry_source_lock(
                    entry, archive, engine_version=cast("str", approved["engineVersion"])
                )
            )
        catalog = staging / "dinkster.lock"
        catalog.write_text(Lockfile.of(admitted).record_json(), encoding="ascii")
        command = [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            cast("str", approved["root"]),
            "setup",
            "--yes",
            "--data",
            str(store),
            "--channel",
            cast("str", approved["channel"]),
            "--variant",
            cast("str", approved["variant"]),
            "--engine-release",
            cast("str", approved["engineRelease"]),
            "--base-id",
            cast("str", approved["baseId"]),
            "--engine",
            args.engine,
            "--port",
            str(args.port),
            "--catalog-lock",
            str(catalog),
            "--artifacts",
            str(archives),
        ]
        for lock in locks.values():
            path = staging / (lock.pack + ".pack-lock.json")
            lock.write(path)
            command.extend(("--pack-lock", str(path)))
        for mirror in args.mirror or ("https://releases.dinkster.org",):
            command.extend(("--mirror", mirror))
        for gpu in args.gpu:
            command.extend(("--gpu", gpu))
        if args.json:
            command.append("--json")
        return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dinkster setup",
        description="Inspect published generation inputs or prepare the local library",
    )
    confirmation = parser.add_mutually_exclusive_group()
    confirmation.add_argument("--yes", action="store_true", help="apply without prompting")
    confirmation.add_argument(
        "--plan", action="store_true", help="inspect published setup without applying"
    )
    parser.add_argument("--json", action="store_true", help="emit the setup plan as JSON")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--engine", default="main")
    parser.add_argument("--port", type=int, default=3639)
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--channel", choices=("stable", "main"))
    parser.add_argument("--mirror", action="append", default=[])
    parser.add_argument("--registry", default=os.environ.get("DINKSTER_REGISTRY", ""))
    parser.add_argument(
        "--registries", type=Path, default=os.environ.get("DINKSTER_REGISTRIES") or None
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.plan
        or args.yes
        or args.root
        or args.variant
        or args.mirror
        or args.json
        or args.offline
    ):
        if args.root is None:
            args.root = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")) / "project"
        try:
            result = _plan(args)
            if not args.plan:
                print("\n".join(cast("list[str]", result["plan"])), file=sys.stderr)
                if not args.yes:
                    if not sys.stdin.isatty():
                        raise InstallError(
                            "confirmation requires a terminal; use --yes to apply "
                            "or --plan to inspect"
                        )
                    if input("Apply this plan? [y/N] ").strip().lower() not in {"y", "yes"}:
                        return 0
                return _apply(args, result)
        except (InstallError, ProvisionError, OSError, ValueError) as exc:
            print(f"setup refused: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(result, sort_keys=True)
            if args.json
            else "\n".join(cast("list[str]", result["plan"]))
        )
        return 0
    if (
        args.json
        or args.root
        or args.data
        or args.variant
        or args.mirror
        or args.offline
        or args.channel
        or args.engine != "main"
        or args.port != 3639
        or args.gpu
    ):
        parser.error("managed selection options require --plan")
    library_root, install_root = default_roots()
    ensure_default_mounts(library_root)
    Installer(install_root)
    print(f"Library: {library_root}")
    print(f"Packs: {install_root}")
    print("Setup complete. Run `dinkster` to open the editor.")
    return 0
