"""Shared command-line boundary for environment management and Desktop."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path
from platform import machine
from typing import cast

from dinkster_store import object_path, pack_archive_path

from .environment_adoption import dependency_comparison, standalone_inputs
from .environment_artifacts import acquire, acquire_pack_locks, discover, validate_manifest
from .environment_collection import (
    automatic_generation_collection,
    cleanup_collection,
    collect_generations,
    collection_plan,
    collection_policy,
    collection_preview,
    recover_collection,
    recover_store_collection,
    retry_generation_collection,
    store_collection_plan,
)
from .environment_custom import (
    base_details,
    build_custom_base,
    custom_selection,
    native_template,
    prepare_custom_pack_locks,
    variant_file,
)
from .environment_generations import (
    build_generation,
    pack_operation,
    prepare_engine_interpreter,
    prepare_registry_inputs,
    project_operation,
    stage_pack_sources,
    verify_base_inputs,
)
from .environment_history import activation_entries, attempts, history, import_generation
from .environment_move import execute_move, moved_to, plan_move, recover_moves
from .environment_pack_install import apply_pack_install, plan_pack_install
from .environment_restore import restore_entry
from .environment_runtime import (
    engine_assignments,
    engine_status,
    require_idle,
    start_engine,
    stop_engine,
    validate_assignment,
)
from .environment_space import space_report
from .environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    atomic_write,
    content_digest,
    generation_number,
    read_current,
    read_document,
    write_document,
    writer_lock,
)
from .environment_stores import (
    register_project,
    registry_path,
    select_store,
    setup_registration,
    store_projects,
)
from .environment_tools import acquire_tools
from .environment_transactions import (
    activate,
    begin_activation,
    begin_setup,
    next_generation,
    recover_activation,
)


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
    if not isinstance(mirrors, list):
        raise EnvironmentError("project mirrors must be an array")
    if any(not isinstance(mirror, str) or not mirror for mirror in cast("list[object]", mirrors)):
        raise EnvironmentError("project mirrors must contain nonempty strings")
    if document.get("packageIndex") is not None and (
        not isinstance(document["packageIndex"], str) or not document["packageIndex"]
    ):
        raise EnvironmentError("project packageIndex must be a nonempty URL or null")
    if type(port) is not int or not 1 <= port <= 65535:
        raise EnvironmentError("project port must be in 1..65535")
    return document


def _query(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    current = read_current(root)
    result["current"] = current
    if args.operation == "space":
        result["space"] = space_report(root)
        return 0
    if args.operation == "history":
        entries = history(root)
        if args.history_action == "show":
            selected = next((entry for entry in entries if entry["id"] == args.entry), None)
            if selected is None:
                raise EnvironmentError(f"installation history has no entry {args.entry}")
            result["entry"] = selected
        else:
            result["history"] = entries
            result["attempts"] = attempts(root)
        return 0
    settings = _settings(root)
    if current is not None and args.operation in {"status", "doctor"}:
        generation_root = root / "generations" / str(current)
        generation = Generation.read(generation_root / "generation.json")
        manifest = generation_root / "base-manifest.json"
        if generation.source != "local" and manifest.is_file():
            base = read_document(manifest)
            if (
                base.get("id") != generation.base
                or base.get("variant") != generation.variant
                or dict(generation.files).get("base-manifest.json") != content_digest(manifest)
            ):
                raise EnvironmentError("retained base manifest does not match active generation")
            result["base"] = base_details(base)
        elif generation.source != "local" and (root / "store").is_file():
            store = Path((root / "store").read_text(encoding="utf-8").strip())
            if not store.is_absolute():
                raise EnvironmentError("project store must be absolute")
            manifest = object_path(store / "bases", generation.base) / "manifest.json"
            if manifest.is_file():
                base = validate_manifest(manifest.read_bytes(), generation.base, generation.variant)
                result["base"] = base_details(base)
        manifest = generation_root / "engine-manifest.json"
        if generation.source != "local" and manifest.is_file():
            engine = validate_manifest(manifest.read_bytes(), generation.engine_release, None)
            if engine["version"] != generation.engine_version:
                raise EnvironmentError("retained engine version does not match active generation")
            if engine.get("label") is not None and not isinstance(engine["label"], str):
                raise EnvironmentError("engine manifest label must be a string or null")
            result["engine"] = {
                "release": generation.engine_release,
                "version": generation.engine_version,
                "label": engine.get("label"),
            }
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
        if current is not None:
            try:
                result["cleanup"] = collection_preview(root)
            except (OSError, ValueError) as exc:
                result["cleanup"] = {"warnings": [str(exc)], "generations": [], "quarantine": []}
    elif args.operation == "doctor":
        if current is None:
            raise EnvironmentError("project has no active generation")
        generation_root = root / "generations" / str(current)
        generation = Generation.read(generation_root / "generation.json")
        if generation.number != current:
            raise EnvironmentError("generation directory and recorded number disagree")
        if generation.source == "local":
            result["source"] = "local"
            result["plan"] = [
                "locally resolved, no base manifest; checking recorded local inventory"
            ]
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
    if args.plan or args.json:
        return False
    if not sys.stdin.isatty():
        raise EnvironmentError(
            "confirmation requires a terminal; use --yes to apply or --plan to inspect"
        )
    print("\n".join(plan), file=sys.stderr)
    return input("Apply this plan? [y/N] ").strip().lower() in {"y", "yes"}


def _recover_pending(root: Path, *, locked_store: Path | None = None) -> None:
    if locked_store is None:
        recover_moves(root)
    if locked_store is not None:
        recover_store_collection(locked_store, locked_root=root)
    elif (root / "store").exists():
        store = Path((root / "store").read_text(encoding="utf-8").strip())
        if not store.is_absolute():
            raise EnvironmentError("project store must be absolute")
        if (store / "collection").exists() or (store / "collection").is_symlink():
            with writer_lock(store):
                recover_store_collection(store)
    for path in sorted((root / "journal").glob("*.json")):
        journal = Journal.read(path)
        if journal.operation == "move":
            if journal.incomplete:
                raise EnvironmentError("move requires recovery before acquiring project locks")
            continue
        if journal.stage == "complete" and journal.history_entry is not None:
            with ExitStack() as locks:
                if locked_store is None:
                    locks.enter_context(writer_lock(root))
                recover_activation(root, path)
            continue
        if not journal.incomplete and not (
            journal.operation == "gc" and journal.stage == "complete"
        ):
            continue
        with ExitStack() as locks:
            if journal.collection_store is not None:
                store = Path(journal.collection_store)
                if locked_store is None:
                    locks.enter_context(writer_lock(store))
                elif store != locked_store:
                    raise EnvironmentError("pending collection uses another store")
            if journal.setup_registration is not None:
                registry = Path(cast("str", journal.setup_registration["registry"])).parent
                store = Path(cast("str", journal.setup_registration["store"]))
                if locked_store is None:
                    locks.enter_context(writer_lock(registry))
                    locks.enter_context(writer_lock(store))
                elif store != locked_store or registry != registry_path().parent:
                    raise EnvironmentError("pending setup uses another store; retry recovery")
            if locked_store is None:
                locks.enter_context(writer_lock(root))
            if journal.operation == "gc":
                if journal.incomplete:
                    recover_collection(root, path)
                else:
                    cleanup_collection(root, path)
            else:
                recover_activation(root, path)


def _install(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root).resolve()
    adopting = getattr(args, "adopt_from", None)
    if adopting is not None and (root.exists() or args.root.is_symlink()):
        raise EnvironmentError("adoption requires a new, nonexistent --root")
    if adopting is not None and (
        args.from_scratch or args.catalog_lock is not None or args.artifacts is not None
    ):
        raise EnvironmentError(
            "adoption uses recorded standalone sources and a published engine/base"
        )
    if adopting is not None and (not args.engine_release or not args.base_id or not args.variant):
        raise EnvironmentError(
            "adoption requires explicit --engine-release, --base-id and --variant"
        )
    if adopting is None and (
        getattr(args, "engine_release", None) or getattr(args, "base_id", None)
    ):
        raise EnvironmentError("--engine-release and --base-id require --adopt-from")
    if args.yes and root.is_dir():
        _recover_pending(root)
    settings = _settings(root)
    current = read_current(root)
    if args.operation == "setup" and current is not None:
        raise EnvironmentError("project already has an active generation")
    if args.operation == "update" and (settings is None or current is None):
        raise EnvironmentError("project has no active generation; run setup first")
    if args.operation == "update" and current is not None:
        active = Generation.read(root / "generations" / str(current) / "generation.json")
        if active.source == "local":
            raise EnvironmentError(
                "cannot channel-update a locally resolved generation; "
                "install a fresh published generation with setup in a new --root "
                "without --from-scratch"
            )
    if args.operation == "update" and (args.port is not None or args.gpu or args.engine != "main"):
        raise EnvironmentError("update preserves engine assignments; stop/start to change them")
    has_requirements = bool(settings and (settings.get("workflows") or settings.get("additions")))
    snapshot = (root / "project.json").read_bytes() if settings else None
    channel = args.channel or (cast("str", settings["channel"]) if settings else "stable")
    variant = args.variant or (cast("str", settings["variant"]) if settings else None)
    if variant is None and sys.platform == "darwin" and machine().lower() == "arm64":
        variant = "mac-arm64"
    mirrors = tuple(args.mirror or (cast("list[str]", settings["mirrors"]) if settings else ()))
    package_index = args.package_index or (settings.get("packageIndex") if settings else None)
    port = (
        args.port
        if args.port is not None
        else (cast("int", settings["port"]) if settings else 3639)
    )
    if not variant:
        raise EnvironmentError("setup requires an explicit --variant")
    project_settings = None
    if settings and (
        channel != settings["channel"]
        or variant != settings["variant"]
        or package_index != settings.get("packageIndex")
    ):
        if args.operation != "update":
            raise EnvironmentError("change channel or variant with update")
        project_settings = {
            "previous": cast("bytes", snapshot).decode("utf-8"),
            "target": json.dumps(
                {
                    **settings,
                    "channel": channel,
                    "variant": variant,
                    "mirrors": list(mirrors),
                    "packageIndex": package_index,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n",
        }
    if settings and args.operation == "setup" and port != settings["port"]:
        raise EnvironmentError("retry setup with the configured initial port")
    if (args.catalog_lock is None) != (args.artifacts is None):
        raise EnvironmentError("source staging requires both --catalog-lock and --artifacts")
    for path in (args.catalog_lock, args.artifacts, *args.pack_lock):
        if path is not None and not path.is_absolute():
            raise EnvironmentError("source and dependency input paths must be absolute")
    previous_root = root / "generations" / str(current) if current is not None else None
    in_process = tuple(args.in_process_pack or ())
    if args.in_process_pack is None and previous_root is not None:
        in_process = Generation.read(previous_root / "generation.json").in_process_packs
    validate_assignment(current or 1, args.engine, port, tuple(args.gpu))
    require_idle(root)
    pointer = root / "store"
    requested = args.data
    store_snapshot = pointer.read_bytes() if pointer.exists() else None
    if store_snapshot is not None:
        configured = Path(store_snapshot.decode("utf-8").strip())
        if not configured.is_absolute() or (requested is not None and requested != configured):
            raise EnvironmentError("project store differs from requested --data")
        requested = configured
    store = select_store(root, requested)
    adoption = standalone_inputs(adopting, root, store) if adopting is not None else None
    adoption_locks = (
        {str(path): content_digest(path) for path in args.pack_lock} if adoption else {}
    )
    if adoption is not None:
        if adoption["venvGroups"]:
            raise EnvironmentError("adoption does not support standalone shared-venv groups")
        recorded_placement = tuple(cast("list[str]", adoption["inProcessPacks"]))
        if args.in_process_pack is not None and set(in_process) != set(recorded_placement):
            raise EnvironmentError("adoption must preserve recorded in-process placement")
        in_process = recorded_placement
        pack_names = {row["pack"] for row in cast("list[dict[str, object]]", adoption["packs"])}
        supplied_names: set[str] = set()
        for path in args.pack_lock:
            document = read_document(path)
            name = document.get("pack")
            if (
                path.is_symlink()
                or document.get("format") != "dinkster.pack-lock/1"
                or document.get("variant") != variant
                or not isinstance(name, str)
            ):
                raise EnvironmentError("adoption requires intact target-variant dependency locks")
            supplied_names.add(name)
        if supplied_names != pack_names or len(args.pack_lock) != len(pack_names):
            raise EnvironmentError(
                "adoption requires one explicit --pack-lock for each recorded pack"
            )
    local = None
    catalog = None
    custom = None
    selection = None
    template = None
    dependency_variant = variant
    if args.operation == "setup" and args.from_scratch:
        if args.pack_lock or args.catalog_lock is not None or args.artifacts is not None:
            raise EnvironmentError(
                "--from-scratch resolves checkout packs; "
                "--pack-lock, --catalog-lock and --artifacts cannot be combined with it"
            )
        from .environment_local import DEFAULT_VARIANTS, local_selection

        local = local_selection(
            args.source.resolve(), variant, args.variants or DEFAULT_VARIANTS, args.python
        )
        size = 0
        plan = [
            f"resolve {args.source.resolve() / 'uv.lock'} locally for {variant}",
            f"interpreter {local['python']}; torch index {local['index']}",
            "from-scratch is opt-in: download size is unknown until dependency resolution",
            "build a local generation, check health, then activate; no published bundle is used",
        ]
    else:
        catalog = discover(mirrors, channel)
        if adoption is not None and (
            catalog.channel.engine_release != args.engine_release
            or catalog.channel.base_for(variant) != args.base_id
        ):
            raise EnvironmentError(
                "selected channel does not match the approved engine release/base"
            )
        retained = previous_root / "base-manifest.json" if previous_root is not None else None
        if (
            retained is not None
            and retained.is_file()
            and settings is not None
            and variant == settings["variant"]
            and args.variants is None
        ):
            manifest = read_document(retained)
            if manifest.get("source") == "user-defined":
                active = Generation.read(retained.parent / "generation.json")
                if (
                    manifest.get("id") != active.base
                    or manifest.get("variant") != active.variant
                    or dict(active.files).get("base-manifest.json") != content_digest(retained)
                ):
                    raise EnvironmentError("retained custom base differs from active generation")
                custom = manifest
        if custom is None:
            definitions = variant_file(root, store, args.variants)
            if definitions is not None:
                template = native_template(catalog.bases)
                selection = custom_selection(definitions, variant, template)
        if adoption is not None and selection is not None:
            raise EnvironmentError(
                "adoption requires an exact published base, not a custom definition"
            )
        if custom is not None or selection is not None:
            template = template or native_template(catalog.bases)
            dependency_variant = cast("str", template["variant"])
            base = custom or template
            base_id = cast("str", custom["id"]) if custom is not None else "pending local build"
        else:
            base_id = catalog.channel.base_for(variant)
            base = catalog.bases[variant]
        parts = cast("list[dict[str, object]]", cast("dict[str, object]", base["archive"])["parts"])
        artifacts = cast("list[dict[str, object]]", catalog.engine["artifacts"])
        size = sum(cast("int", row["size"]) for row in (*parts, *artifacts))
        plan = [
            f"{args.operation} engine {catalog.engine['version']} with {variant} base {base_id}",
            f"artifact payload size {size} bytes; reuse verified local artifacts in {store}",
            "build an offline generation, check health, then activate; retain the previous tree",
        ]
        if selection is not None:
            plan.append(
                f"build user-defined base with torch {selection['torch']} from "
                f"{selection['torch-index']}; extras from {selection['extras-index']}"
            )
            plan.append("custom base download size is unknown until dependency resolution")
        elif custom is not None:
            plan.append(f"retain user-defined base {base_id}; do not resolve its definition again")
    if project_settings is not None and settings is not None:
        plan.append(
            f"change project selection from {settings['channel']}/{settings['variant']} "
            f"to {channel}/{variant} only after candidate health and workflow revalidation"
        )
    if args.operation == "setup":
        plan.insert(
            1,
            f"engine binding {args.engine}; listen on 127.0.0.1:{port}; "
            f"CUDA_VISIBLE_DEVICES={','.join(args.gpu)}",
        )
    result.update(
        current=current, plan=plan, status="planned", downloadBytes=size, store=str(store)
    )
    if adoption is not None:
        result["adoption"] = adoption
        result["dependencyLocks"] = {str(path): read_document(path) for path in args.pack_lock}
        result["dependencyComparison"] = dependency_comparison(adoption, tuple(args.pack_lock))
        plan.extend(
            [
                f"adopt standalone generation {adoption['generation']} from {adopting} into {root}",
                f"preserve source lock: {adoption['sourceLock']}",
                f"preserve in-process placement: {', '.join(in_process) or 'none'}",
                "old engine/base are not recorded; use the explicitly selected identities",
                "source dependency inspection reads local distribution metadata only; "
                "ambient, inherited and .pth-provided packages are not captured",
                "shared-store scopes are displayed independently; no source accelerator is guessed",
                "comparison is against pack locks, not the total target engine/base closure; "
                "a null targetLock means not declared in that pack lock, not necessarily removed",
                "source dependencies: "
                f"{json.dumps(adoption['installedDependencies'], sort_keys=True)}",
                "dependency comparison: "
                f"{json.dumps(result['dependencyComparison'], sort_keys=True)}",
                f"target dependency locks: {json.dumps(result['dependencyLocks'], sort_keys=True)}",
                "leave the old installation, shared store and all user data untouched",
            ]
        )
    workflow = None
    workflow_catalog = None
    workflow_catalog_digest = None
    workflow_artifacts = None
    workflow_target = None
    workflow_device = None
    workflow_applied = False
    if has_requirements:
        if catalog is None:
            raise EnvironmentError("local setup requires a fresh project without workflow settings")
        if previous_root is None:
            raise EnvironmentError("workflow planning requires an existing generation interpreter")
        workflow = project_operation(
            root,
            generation=previous_root,
            variant=variant,
            requirements_only=True,
        )
        plan.append(
            f"workflow requirements: {json.dumps(workflow['requirements'], sort_keys=True)}"
        )
        plan.append("workflow source feasibility: pending target preparation after confirmation")
        if (root / "project.json").read_bytes() != snapshot:
            raise EnvironmentError("project changed during planning; request a new plan")
    if not _confirmed(args, plan):
        return 0

    def preflight_workflow(engine: Path, catalog_lock: Path, artifacts: Path) -> None:
        nonlocal workflow, workflow_catalog, workflow_artifacts
        nonlocal workflow_catalog_digest, workflow_target, workflow_device
        assert workflow is not None and previous_root is not None
        digest = content_digest(catalog_lock)
        if custom is not None:
            from .environment_local import run_local_command
            from .environment_state import interpreter

            workflow_device = run_local_command(
                [
                    str(interpreter(engine)),
                    "-I",
                    "-B",
                    "-c",
                    "import json,sys\nfrom pathlib import Path\n"
                    "from dinkster.workflow_requirements import acquired_base_device\n"
                    "base=json.load(sys.stdin)\n"
                    "row=next(row for row in base['wheels'] if row['name']=='torch')\n"
                    "print(acquired_base_device(base,Path(sys.argv[1])/row['filename']))\n",
                    str(object_path(store / "bases", cast("str", custom["id"])) / "wheels"),
                ],
                cwd=engine,
                input=json.dumps(custom),
            )
        observed = project_operation(
            root,
            generation=previous_root,
            target_generation=engine,
            variant=variant,
            catalog_lock=catalog_lock,
            artifacts=artifacts,
            target_device=workflow_device,
        )
        if any(observed[key] != workflow[key] for key in ("requirements", "inputs")):
            raise EnvironmentError("workflow inputs changed after planning; request a new plan")
        actions = cast("dict[str, object]", observed["plan"])
        if actions["errors"]:
            raise EnvironmentError(
                f"workflow planning refused: {'; '.join(cast('list[str]', actions['errors']))}"
            )
        for kind in ("packs", "assets"):
            for action in cast("list[object]", actions[kind]):
                plan.append(f"workflow {kind}: {json.dumps(action, sort_keys=True)}")
        workflow = observed
        workflow_catalog, workflow_artifacts = catalog_lock, artifacts
        workflow_catalog_digest, workflow_target = digest, engine

    def revalidate() -> None:
        if adoption is not None and (
            standalone_inputs(cast("Path", adopting), root, store) != adoption
            or any(content_digest(Path(path)) != value for path, value in adoption_locks.items())
        ):
            raise EnvironmentError("standalone inputs changed after planning; request a new plan")
        actual = (root / "project.json").read_bytes() if (root / "project.json").exists() else None
        actual_store = pointer.read_bytes() if pointer.exists() else None
        if read_current(root) != current or actual != snapshot or actual_store != store_snapshot:
            raise EnvironmentError("project changed after planning; request a new plan")
        if workflow is not None and previous_root is not None:
            if (
                workflow_catalog is not None
                and content_digest(workflow_catalog) != workflow_catalog_digest
            ):
                raise EnvironmentError("target catalog changed after planning; request a new plan")
            observed = project_operation(
                root,
                generation=root / "generations" / str(target)
                if workflow_applied
                else previous_root,
                variant=variant,
                catalog_lock=workflow_catalog,
                artifacts=workflow_artifacts,
                target_generation=workflow_target if not workflow_applied else None,
                requirements_only=workflow_target is None and not workflow_applied,
                target_device=workflow_device,
            )
            if any(observed[key] != workflow[key] for key in ("requirements", "inputs")):
                raise EnvironmentError("workflow inputs changed after planning; request a new plan")
            observed_actions = cast("dict[str, object]", observed["plan"])
            if observed_actions["errors"] or (workflow_applied and any(observed_actions.values())):
                raise EnvironmentError(
                    "workflow requirements failed revalidation; request a new plan"
                )

    registry = registry_path().parent
    registry.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=adoption is None)
    with writer_lock(registry):
        if select_store(root, requested) != store:
            raise EnvironmentError("store changed after planning; request a new plan")
        store.mkdir(parents=True, exist_ok=True)
        with writer_lock(store), writer_lock(root), ExitStack() as preparation:
            _recover_pending(root, locked_store=store)
            revalidate()
            published_constraints = None
            catalog_lock, source_artifacts = args.catalog_lock, args.artifacts
            pack_locks = tuple(args.pack_lock)
            if adoption is not None:
                scratch = Path(preparation.enter_context(tempfile.TemporaryDirectory(dir=store)))
                catalog_lock, source_artifacts = scratch / "dinkster.lock", scratch / "artifacts"
                atomic_write(catalog_lock, cast("str", adoption["sourceLock"]))
                source_artifacts.mkdir()
                archives = Path(cast("str", adoption["artifacts"]))
                archive_digests = cast("dict[str, str]", adoption["archiveDigests"])
                for row in cast("list[dict[str, object]]", adoption["packs"]):
                    identity = cast("str", row["artifactDigest"])
                    archive = pack_archive_path(archives, identity)
                    expected = archive_digests[archive.relative_to(archives).as_posix()]
                    copied = source_artifacts / f"{identity[7:]}.zip"
                    shutil.copyfile(archive, copied)
                    if content_digest(copied) != expected:
                        raise EnvironmentError("standalone archive changed during adoption")
            if local is None:
                assert catalog is not None
                if selection is not None:
                    assert template is not None
                    custom = build_custom_base(
                        selection, template, catalog.base_sources[dependency_variant], store
                    )
                base = custom or catalog.bases[variant]
                base_id = cast("str", base["id"])
                published_constraints = acquire(
                    catalog,
                    dependency_variant,
                    store,
                    package_index=cast("str | None", package_index),
                    base=custom,
                    base_sources=(str(object_path(store / "downloads", base_id)),)
                    if custom is not None
                    else None,
                )

                def custom_locks(
                    engine: Path, selected: Path, archives: Path, output: Path
                ) -> tuple[Path, ...]:
                    assert custom is not None
                    return prepare_custom_pack_locks(
                        engine,
                        store=store,
                        base=custom,
                        catalog_lock=selected,
                        artifacts=archives,
                        output=output,
                    )

                if catalog_lock is None:
                    catalog_lock, source_artifacts, pack_locks, unavailable = (
                        preparation.enter_context(
                            prepare_registry_inputs(
                                store=store,
                                base=base_id,
                                constraints=published_constraints,
                                variant=variant,
                                engine_version=cast("str", catalog.engine["version"]),
                                installed_lock=previous_root / "dinkster.lock"
                                if previous_root is not None
                                else None,
                                project=root,
                                pack_locks=pack_locks,
                                prepare_locks=custom_locks if custom is not None else None,
                                preflight=preflight_workflow if workflow is not None else None,
                            )
                        )
                    )
                    result["unavailablePacks"] = unavailable
                    for pack, reason in unavailable.items():
                        print(f"default pack {pack} unavailable: {reason}", file=sys.stderr)
                else:
                    assert source_artifacts is not None
                    scratch = (
                        Path(
                            preparation.enter_context(
                                tempfile.TemporaryDirectory(dir=store, prefix=".workflow-")
                            )
                        )
                        / "engine"
                    )
                    prepare_engine_interpreter(
                        scratch,
                        store=store,
                        base=base_id,
                        constraints=published_constraints,
                    )
                    if workflow is not None:
                        preflight_workflow(scratch, catalog_lock, source_artifacts)
                    if not pack_locks and adoption is None:
                        pack_locks = (
                            custom_locks(
                                scratch, catalog_lock, source_artifacts, scratch.parent / "locks"
                            )
                            if custom is not None
                            else acquire_pack_locks(catalog, variant, store)
                        )
                    stage_pack_sources(
                        scratch, catalog_lock=catalog_lock, artifacts=source_artifacts
                    )
            revalidate()
            target = next_generation(root)
            journal = (
                begin_setup(
                    root,
                    target,
                    port,
                    tuple(args.gpu),
                    name=args.engine,
                    registration=setup_registration(root, store),
                )
                if current is None
                else begin_activation(root, target, "update", project_settings=project_settings)
            )
            try:
                register_project(root, store)
                if settings is None:
                    write_document(
                        root / "project.json",
                        {
                            "format": "dinkster.project/1",
                            "channel": channel,
                            "variant": variant,
                            "mirrors": list(mirrors),
                            "packageIndex": package_index,
                            "port": port,
                        },
                    )
                    snapshot = (root / "project.json").read_bytes()
                atomic_write(pointer, str(store) + "\n")
                store_snapshot = pointer.read_bytes()
                replace(Journal.read(journal), stage="building").write(journal)
                prepare_packs = None
                if local is not None:
                    from .environment_local import prepare_local, prepare_local_packs

                    print("\n".join(plan), file=sys.stderr, flush=True)
                    inputs = prepare_local(args.source.resolve(), local, store)
                    base_id = inputs["base"]
                    constraints = inputs["constraints"]
                    engine_version = inputs["engine_version"]
                    engine_release = inputs["engine_release"]
                    worker_protocol = inputs["worker_protocol"]
                    pack_locks = tuple(args.pack_lock)
                    prepare_packs = partial(
                        prepare_local_packs,
                        source=args.source.resolve(),
                        store=store,
                        release=engine_release,
                        selection=local,
                        in_process=in_process,
                    )
                else:
                    assert catalog is not None
                    assert published_constraints is not None
                    base = custom or catalog.bases[variant]
                    base_id = cast("str", base["id"])
                    constraints = published_constraints
                    engine_version = cast("str", catalog.engine["version"])
                    engine_release = catalog.channel.engine_release
                    worker_protocol = cast("int", base["workerProtocol"])
                measured = sum(
                    path.stat().st_size
                    for path in (object_path(store / "bases", base_id) / "python").rglob("*")
                    if path.is_file()
                )
                build_generation(
                    root=root,
                    number=target,
                    store=store,
                    base=base_id,
                    engine_version=engine_version,
                    engine_release=engine_release,
                    constraints=constraints,
                    variant=variant,
                    worker_protocol=worker_protocol,
                    source_generation=(
                        previous_root
                        if previous_root is not None
                        and ((previous_root / "packs").exists() or pack_locks or in_process)
                        else None
                    ),
                    pack_locks=pack_locks,
                    in_process_packs=in_process,
                    own_environment_size=measured,
                    catalog_lock=catalog_lock,
                    artifacts=source_artifacts,
                    source_kind="local" if local is not None else "published",
                    prepare_packs=prepare_packs,
                    acquire_pack_wheels=local is None,
                    artifact_sources={
                        "channel": catalog.document,
                        "engine": list(catalog.engine_sources),
                        "base": [str(object_path(store / "downloads", base_id))]
                        if custom is not None
                        else list(catalog.base_sources[dependency_variant]),
                        "dependencyVariant": dependency_variant,
                        "dependencyBase": catalog.bases[dependency_variant],
                        "packageIndex": package_index,
                        "mirrors": list(mirrors),
                    }
                    if catalog is not None
                    else None,
                )
                if workflow is not None and previous_root is not None:
                    workflow_target = root / "generations" / str(target)
                    workflow_device = None
                    revalidate()
                    applied = project_operation(
                        root,
                        generation=previous_root,
                        variant=variant,
                        catalog_lock=workflow_catalog,
                        artifacts=workflow_artifacts,
                        approved=workflow,
                        target_generation=root / "generations" / str(target),
                    )
                    if any(cast("dict[str, object]", applied["plan"]).values()):
                        raise EnvironmentError("workflow execution left unsatisfied requirements")
                    workflow_applied = True
                activate(root, journal, revalidate=revalidate)
            except Exception as exc:
                replace(Journal.read(journal), error=str(exc)).write(journal)
                recover_activation(root, journal)
                raise
            engines = list(engine_status(root))
            result.update(current=target, status="applied", engines=engines)
            ready = next((row for row in engines if row["state"] == "ready"), None)
            if ready is not None:
                result["url"] = ready["url"]
    return 0


def _repair(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root).resolve()
    if args.yes:
        _recover_pending(root)
    current = read_current(root)
    if current is None:
        raise EnvironmentError("project has no active generation")
    previous = root / "generations" / str(current)
    record = previous / "generation.json"
    snapshot = record.read_bytes()
    generation = Generation.read(record)
    if generation.number != current:
        raise EnvironmentError("generation directory and recorded number disagree")
    pointer = root / "store"
    store_snapshot = pointer.read_bytes()
    store = Path(store_snapshot.decode("utf-8").strip())
    if not store.is_absolute():
        raise EnvironmentError("project store must be absolute")
    select_store(root, store)
    settings = (root / "project.json").read_bytes()
    expected = dict(generation.files)
    base = object_path(store / "bases", generation.base) / "python"
    release = object_path(store / "engines", generation.engine_release)
    constraints = previous / "constraints.txt"
    if (
        not constraints.is_file()
        or constraints.is_symlink()
        or content_digest(constraints) != generation.constraints
    ):
        constraints = release / f"constraints-{generation.variant}.txt"
    catalog = previous / "dinkster.lock" if "dinkster.lock" in expected else None

    def revalidate() -> None:
        if (
            read_current(root) != current
            or record.read_bytes() != snapshot
            or pointer.read_bytes() != store_snapshot
            or (root / "project.json").read_bytes() != settings
        ):
            raise EnvironmentError("project changed after planning; request a new plan")
        if (
            not constraints.is_file()
            or constraints.is_symlink()
            or content_digest(constraints) != generation.constraints
        ):
            raise EnvironmentError("repair requires intact cached effective constraints")
        verify_base_inputs(base, expected, operation="repair")
        if catalog is not None and (
            catalog.is_symlink() or content_digest(catalog) != expected["dinkster.lock"]
        ):
            raise EnvironmentError("repair requires intact retained source inventory")

    revalidate()
    require_idle(root)
    plan = [
        f"repair generation {current} offline from retained base {generation.base}",
        "clear the disposable uv cache and rebuild from local hash-verified wheels",
        "check target health before activation; retain the drifted tree for diagnosis",
    ]
    result.update(current=current, plan=plan, status="planned", store=str(store))
    if not _confirmed(args, plan):
        return 0
    with writer_lock(store), writer_lock(root):
        _recover_pending(root, locked_store=store)
        revalidate()
        target = next_generation(root)
        journal = begin_activation(root, target, "repair")
        try:
            replace(Journal.read(journal), stage="building").write(journal)
            repaired = build_generation(
                root=root,
                number=target,
                store=store,
                base=generation.base,
                engine_version=generation.engine_version,
                engine_release=generation.engine_release,
                constraints=constraints,
                variant=generation.variant,
                worker_protocol=generation.worker_protocol,
                in_process_packs=generation.in_process_packs,
                source_generation=previous if (previous / "packs").exists() else None,
                own_environment_size=sum(
                    path.stat().st_size for path in base.rglob("*") if path.is_file()
                ),
                catalog_lock=catalog,
                artifacts=release / "artifacts" if catalog is not None else None,
                refresh_cache=True,
                source_kind=generation.source,
                artifact_sources=read_document(previous / "artifact-sources.json")
                if "artifact-sources.json" in expected
                else None,
            )
            added_locks = {
                name
                for name, _ in repaired.files
                if len(Path(name).parts) == 3
                and Path(name).parts[0] == "packs"
                and Path(name).parts[2] == "pack-lock.json"
                and name not in expected
            }
            if set(dict(repaired.files)) - added_locks != set(expected):
                raise EnvironmentError(
                    "repaired file inventory differs from the retained generation"
                )
            activate(root, journal, revalidate=revalidate)
        except Exception as exc:
            replace(Journal.read(journal), error=str(exc)).write(journal)
            recover_activation(root, journal)
            raise
        engines = list(engine_status(root))
        result.update(current=target, status="applied", engines=engines)
        ready = next((row for row in engines if row["state"] == "ready"), None)
        if ready is not None:
            result["url"] = ready["url"]
    return 0


def _import(args: argparse.Namespace, result: dict[str, object]) -> int:
    number = generation_number(args.generation)
    root = Path(args.root)
    plan = [
        f"preserve generation {number} as an explicit legacy software import",
        "recover completed history publication; require intact software and completed ancestry",
        "keep current software and activation chronology; retain unrecorded pack payloads",
    ]
    result.update(current=read_current(root), plan=plan, status="planned")
    if not _confirmed(args, plan):
        return 0
    _recover_pending(root)
    with writer_lock(root):
        entry = import_generation(root, number)
        result.update(current=read_current(root), entry=entry, status="applied")
    return 0


def _rollback(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    with writer_lock(root):
        for path in sorted((root / "journal").glob("*.json")):
            journal = Journal.read(path)
            if journal.stage == "complete" and journal.history_entry is not None:
                recover_activation(root, path)
        current = read_current(root)
        if current is None:
            raise EnvironmentError("project has no active generation")
        entries = activation_entries(root)
        if len(entries) < 2:
            raise EnvironmentError("project has no previous successful history entry")
        if entries[-1]["generation"] != current:
            raise EnvironmentError("current generation does not match installation history")
    return _restore(args, result, cast("int", entries[-2]["id"]), reason="rollback")


def _restore(
    args: argparse.Namespace,
    result: dict[str, object],
    entry_id: int,
    *,
    reason: str = "restore",
) -> int:
    result.update(restore_entry(args.root, entry_id, apply=False, approval=args.approve))
    can_apply = result["status"] != "approval-required" or args.approve is not None
    if can_apply and (
        (args.approve is not None and not args.plan)
        or _confirmed(args, cast("list[str]", result["plan"]))
    ):
        _recover_pending(args.root)
        result.update(
            restore_entry(
                args.root,
                entry_id,
                apply=True,
                approval=args.approve,
                expected_plan=cast("dict[str, object]", result["restorePlan"]),
                reason=reason,
            )
        )
        engines = list(engine_status(args.root))
        result["engines"] = engines
        ready = next((row for row in engines if row["state"] == "ready"), None)
        if ready is not None:
            result["url"] = ready["url"]
    return 0


def _cleanup(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    if args.keep_previous is not None and not 1 <= args.keep_previous <= 100:
        raise EnvironmentError("previous generation count must be an integer from 1 to 100")
    changing_policy = args.automatic is not None or args.keep_previous is not None
    if not changing_policy and args.yes:
        attempt = retry_generation_collection(root)
        result.update(cleanupAttempt=attempt, status=attempt["status"], current=read_current(root))
        result["cleanup"] = collection_preview(root)
        return 0
    preview = collection_preview(root)
    plan = [
        f"generation {row['generation']}: {row['status']}: "
        + "; ".join(cast("list[str]", row["reasons"]))
        for row in cast("list[dict[str, object]]", preview["generations"])
    ] + cast("list[str]", preview["warnings"])
    result.update(cleanup=preview, current=preview["current"], status="planned", plan=plan)
    if not changing_policy:
        return 0
    requested = dict(cast("dict[str, object]", preview["policy"]))
    if args.automatic is not None:
        requested["automatic"] = args.automatic == "on"
        plan.append(f"set automatic generation cleanup {args.automatic}; do not collect now")
        result["requestedAutomaticCleanup"] = requested["automatic"]
    if args.keep_previous is not None:
        requested["previousSuccessfulActivations"] = args.keep_previous
        plan.append(
            f"keep {args.keep_previous} previous successful generations; do not collect now"
        )
        result["requestedPreviousSuccessfulActivations"] = args.keep_previous
    if not _confirmed(args, plan):
        return 0
    with writer_lock(root):
        policy = collection_policy(root)
        if policy != preview["policy"]:
            raise EnvironmentError("cleanup policy changed after planning; request a new plan")
        write_document(
            root / "cleanup.json",
            {
                "format": "dinkster.cleanup/1",
                "automatic": requested["automatic"],
                "previousSuccessfulActivations": requested["previousSuccessfulActivations"],
            },
        )
        result["cleanup"] = collection_preview(root)
    result["status"] = "applied"
    return 0


def _gc(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root)
    if args.yes:
        _recover_pending(root)
    approved = collection_plan(root)
    current, retained, collected = approved
    pointer = root / "store"
    store = Path(pointer.read_text(encoding="utf-8").strip()) if pointer.exists() else None
    if store is not None and (
        not store.is_absolute() or store.is_symlink() or pointer.is_symlink()
    ):
        raise EnvironmentError("collection requires an absolute unlinked store")
    projects = store_projects(store) if store is not None else []
    store_paths = store_collection_plan(root, store, collected) if store is not None else ()
    plan = [
        f"retain generations {', '.join(map(str, retained))}",
        f"collect obsolete generations {', '.join(map(str, collected)) or 'none'}",
        "quarantine before committing deletion; preserve referenced cache and project libraries",
        *[f"collect unreferenced cache {name}" for name in store_paths],
    ]
    result.update(current=current, plan=plan, status="planned")
    if not _confirmed(args, plan):
        return 0
    _recover_pending(root)
    with ExitStack() as locks:
        if store is not None:
            registry_path().parent.mkdir(parents=True, exist_ok=True)
            locks.enter_context(writer_lock(registry_path().parent))
            locks.enter_context(writer_lock(store))
            if store_projects(store) != projects or pointer.read_text(
                encoding="utf-8"
            ).strip() != str(store):
                raise EnvironmentError(
                    "store references changed after planning; request a new plan"
                )
        for project in sorted(set(projects) | {str(root)}):
            locks.enter_context(writer_lock(Path(project)))
        collect_generations(root, approved, store=store, store_paths=store_paths)
    result.update(
        status="applied",
        collectedGenerations=list(collected),
        collectedStorePaths=list(store_paths),
    )
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
        port = cast("int", settings["port"])
        if args.operation == "start":
            binding = next(
                (row for row in engine_assignments(root, current) if row["name"] == args.engine),
                None,
            )
            if binding is not None:
                port = cast("int", binding["port"])
            elif args.engine != "main":
                raise EnvironmentError("an additional engine binding requires --port")
    gpus = tuple(args.gpu) if args.operation == "start" else ()
    validate_assignment(current, args.engine, port, gpus)
    plan = [f"{args.operation} engine {args.engine} in generation {current}"]
    if args.operation == "start":
        plan.append(f"listen on 127.0.0.1:{port}; CUDA_VISIBLE_DEVICES={','.join(gpus)}")
    result.update(plan=plan, engines=list(engine_status(root)))
    if not _confirmed(args, plan):
        result["status"] = "planned"
        return 0
    _recover_pending(root)
    with writer_lock(root):
        if any(Journal.read(path).incomplete for path in (root / "journal").glob("*.json")):
            raise EnvironmentError("project changed after recovery; retry the operation")
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


def _move(args: argparse.Namespace, result: dict[str, object]) -> int:
    root = Path(args.root).resolve()
    if args.yes:
        _recover_pending(root)
    selected = plan_move(root, args.destination, args.data)
    plan = [
        f"move {root} to {selected.target}; store {selected.target_store}",
        "stop idle engines temporarily; restore their named ports and GPU assignments",
        "rebuild retained generations offline on the destination volume"
        if selected.cross_volume
        else "keep package sharing and rewrite pack venv homes",
        "validate destination doctor and health before committing; retain source recovery journals",
    ]
    result.update(
        current=selected.current, plan=plan, status="planned", destination=str(selected.target)
    )
    if not _confirmed(args, plan):
        return 0
    _recover_pending(root)
    execute_move(selected)
    result.update(status="applied", engines=list(engine_status(selected.target)))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="absolute prospective project root"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    catalog = commands.add_parser("catalog", help="inspect a channel and intact variant manifests")
    catalog.add_argument("--mirror", action="append", default=[])
    catalog.add_argument("--channel", choices=("stable", "main"))
    tools = commands.add_parser("tools", help="acquire the channel's verified native tools ZIP")
    tools.add_argument("--mirror", action="append", required=True)
    tools.add_argument("--channel", choices=("stable", "main"), default="main")
    status = commands.add_parser(
        "status", help="read configured project and observed engine status"
    )
    space = commands.add_parser("space", help="observe managed software disk allocation")
    history_command = commands.add_parser("history", help="list durable software history")
    history_actions = history_command.add_subparsers(dest="history_action")
    history_show = history_actions.add_parser("show", help="show a snapshot and its diff")
    history_show.add_argument("entry", type=int)
    history_command.add_argument("--json", action="store_true", help="emit one result object")
    history_show.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    packs_plan = commands.add_parser("packs-plan", help="retain exact additive software inputs")
    packs_plan.add_argument("packs", nargs="+", metavar="PACK@VERSION")
    packs_plan.add_argument("--output", type=Path, required=True)
    packs_apply = commands.add_parser("packs-apply", help="apply a retained software review once")
    packs_apply.add_argument("--input", type=Path, required=True)
    packs_apply.add_argument("--approval", required=True)
    doctor = commands.add_parser("doctor", help="report active generation and pack drift")
    doctor.add_argument(
        "--repair", action="store_true", help="rebuild offline from the local store"
    )
    setup = commands.add_parser("setup", help="build and activate a first offline generation")

    setup.add_argument(
        "--from-scratch", action="store_true", help="resolve locally; development only"
    )
    setup.add_argument(
        "--source", type=Path, default=Path.cwd(), help="Dinkster checkout with uv.lock"
    )
    setup.add_argument("--python", type=Path, help="explicit host interpreter override")
    setup.add_argument(
        "--adopt-from", type=Path, help="standalone pack-only root to leave untouched"
    )
    setup.add_argument("--engine-release", help="explicit published engine digest for adoption")
    setup.add_argument("--base-id", help="explicit published base digest for adoption")
    update = commands.add_parser("update", help="build and activate a channel's next generation")
    for command in (setup, update):
        command.add_argument("--mirror", action="append", default=[])
        command.add_argument(
            "--package-index", help="simple package index for exact locked third-party wheels"
        )
        command.add_argument("--channel", choices=("stable", "main"))
        command.add_argument("--variant")
        command.add_argument("--variants", type=Path, help="explicit custom variant definitions")
        command.add_argument("--data", type=Path)
        command.add_argument("--port", type=int)
        command.add_argument("--engine", default="main")
        command.add_argument("--gpu", action="append", default=[])
        command.add_argument("--pack-lock", type=Path, action="append", default=[])
        command.add_argument("--in-process-pack", action="append")
        command.add_argument("--catalog-lock", type=Path)
        command.add_argument("--artifacts", type=Path)
    start = commands.add_parser("start", help="start the current generation's assigned engine")
    start.add_argument("--port", type=int)
    start.add_argument("--gpu", action="append", default=[])
    stop = commands.add_parser("stop", help="stop one identity-checked supervisor and engine")
    rollback = commands.add_parser("rollback", help="rebuild the previous successful snapshot")
    restore = commands.add_parser("restore", help="rebuild a recorded software snapshot")
    restore.add_argument("entry", type=int)
    legacy_import = commands.add_parser("import", help="preserve identified legacy software")
    legacy_import.add_argument("generation", type=int)
    for command in (rollback, restore):
        command.add_argument("--approve", help="approve one complete restore proposal digest")
    gc = commands.add_parser("gc", help="collect obsolete generation trees after confirmation")
    cleanup = commands.add_parser("cleanup", help="preview generation retention or set its policy")
    cleanup.add_argument(
        "--automatic", choices=("on", "off"), help="persist automatic cleanup preference"
    )
    cleanup.add_argument(
        "--keep-previous",
        type=int,
        help="keep 1-100 previous generations besides current (default 1)",
    )
    move = commands.add_parser("move", help="relocate a project and validate retained generations")
    move.add_argument("destination", type=Path)
    move.add_argument("--data", type=Path)
    for command in (start, stop):
        command.add_argument("--engine", default="main")
    for command in (
        tools,
        doctor,
        setup,
        update,
        start,
        stop,
        rollback,
        restore,
        legacy_import,
        gc,
        cleanup,
        move,
    ):
        confirmation = command.add_mutually_exclusive_group()
        confirmation.add_argument("--yes", action="store_true", help="apply the printed plan")
        confirmation.add_argument("--plan", action="store_true", help="inspect without mutation")
    for command in (
        tools,
        catalog,
        status,
        space,
        packs_plan,
        packs_apply,
        doctor,
        setup,
        update,
        start,
        stop,
        rollback,
        restore,
        legacy_import,
        gc,
        cleanup,
        move,
    ):
        command.add_argument("--json", action="store_true", help="emit exactly one result object")
    args = parser.parse_args(argv)
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
        current = read_current(args.root)
        if current is not None and (args.root / "generations" / f"{current}.json").is_file():
            raise EnvironmentError(
                "standalone pack-only root has no recorded engine release/base; "
                "choose a new --root and run setup --adopt-from with explicit "
                "--engine-release, --base-id, --variant and target --pack-lock inputs; "
                "the old root and user data will stay untouched"
            )
        destination = moved_to(args.root)
        if destination is not None:
            if getattr(args, "yes", False):
                recover_moves(args.root)
            raise EnvironmentError(f"project moved to {destination}")
        if args.operation == "packs-plan":
            result.update(plan_pack_install(args.root, tuple(args.packs), args.output))
            code = 0
        elif args.operation == "packs-apply":
            result.update(apply_pack_install(args.root, args.input, args.approval))
            code = 0
        elif args.operation == "tools":
            selected = acquire_tools(tuple(args.mirror), args.channel, args.root, download=args.yes)
            result.update(
                tools=selected,
                status="applied" if args.yes else "planned",
                plan=[f"verified tools {selected['label']}: {selected['archive']}"],
            )
            code = 0
        elif args.operation == "doctor" and args.repair:
            code = _repair(args, result)
        elif args.operation in {"setup", "update"}:
            code = _install(args, result)
        elif args.operation == "rollback":
            code = _rollback(args, result)
        elif args.operation == "restore":
            code = _restore(args, result, args.entry)
        elif args.operation == "import":
            code = _import(args, result)
        elif args.operation == "gc":
            code = _gc(args, result)
        elif args.operation == "cleanup":
            code = _cleanup(args, result)
        elif args.operation == "move":
            code = _move(args, result)
        elif args.operation in {"start", "stop"}:
            code = _runtime(args, result)
        else:
            code = _query(args, result)
    except (OSError, ValueError) as exc:
        code = 1
        result.update(status="error", error=str(exc))
    if (
        code == 0
        and result["status"] == "applied"
        and args.operation
        in {
            "setup",
            "update",
            "doctor",
            "rollback",
            "restore",
            "start",
            "stop",
        }
    ):
        result["cleanupAttempt"] = automatic_generation_collection(args.root)
    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    else:
        print(f"{result['operation']}: {result['status']}")
        if "base" in result:
            base = cast("dict[str, object]", result["base"])
            print(f"Variant: {base['variant']} ({base['origin']})")
            print(f"Torch: {base['torchVersion']}")
            print(f"Torch index: {base['torchIndex']}")
            if base["label"] is not None:
                print(f"Base label: {base['label']}")
        if "engine" in result:
            engine = cast("dict[str, object]", result["engine"])
            print(f"Engine: {engine['version']} ({engine['release']})")
            if engine["label"] is not None:
                print(f"Engine label: {engine['label']}")
        cleanup = cast("dict[str, object]", result.get("cleanup", {}))
        if "policy" in cleanup:
            policy = cast("dict[str, object]", cleanup["policy"])
            print(
                f"Automatic cleanup: {'on' if policy['automatic'] else 'off'}; "
                f"keep current and {policy['previousSuccessfulActivations']} previous generations"
            )
        for step in cast("list[str]", result["plan"]):
            print(step)
        for change in cast("list[str]", result.get("changes", [])):
            print(change)
        if "error" in result:
            print(result["error"], file=sys.stderr)
        if args.operation == "history" and code == 0:
            print(
                json.dumps(result.get("entry", result.get("history")), indent=2, ensure_ascii=True)
            )
        if args.operation == "space" and code == 0:
            print(json.dumps(result["space"], indent=2, ensure_ascii=True))
        if args.operation == "catalog" and code == 0:
            print(json.dumps(result["variants"], indent=2, ensure_ascii=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
