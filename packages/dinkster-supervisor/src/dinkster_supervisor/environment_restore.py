"""Rebuild recorded software without restoring project data or shared tools."""

from __future__ import annotations

import secrets
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import cast

from dinkster_store import object_path, readable_label

from .environment_artifacts import (
    EnvironmentCatalog,
    acquire,
    acquire_sources,
    artifact_rows,
    validate_manifest,
)
from .environment_downloads import fetch_artifact
from .environment_generations import build_generation
from .environment_history import (
    activation_entries,
    history,
    is_software_metadata,
    prepare_entry,
    record_failed_restore,
    validate_import,
)
from .environment_library import library_plan
from .environment_restore_plan import plan_restore
from .environment_state import (
    Channel,
    EnvironmentError,
    Journal,
    atomic_write,
    content_digest,
    digest,
    interpreter,
    journal_paths,
    library_root,
    read_current,
    read_document,
    write_document,
    writer_lock,
)
from .environment_stores import register_project, registry_path, select_store, setup_registration
from .environment_transactions import (
    activate,
    begin_activation,
    begin_setup,
    next_generation,
    recover_activation,
)


def restore_entry(
    root: Path,
    entry_id: int,
    *,
    apply: bool,
    approval: str | None = None,
    expected_plan: dict[str, object] | None = None,
    reason: str = "restore",
    destination: Path | None = None,
) -> dict[str, object]:
    """Rebuild recorded software only after exact or proposal-bound approval."""
    entries = history(root)
    selected = next((entry for entry in entries if entry["id"] == entry_id), None)
    if selected is None:
        raise EnvironmentError(f"installation history has no entry {entry_id}")
    snapshot = cast("dict[str, object]", selected["snapshot"])
    tools = snapshot.get("tools")
    recorded_tools = (
        cast("dict[str, object]", tools).get("version") if isinstance(tools, dict) else "unknown"
    )
    if snapshot.get("format") != "dinkster.install-snapshot/1":
        raise EnvironmentError(f"unsupported snapshot recorded by tools {recorded_tools}")
    current = read_current(root)
    if current is None:
        raise EnvironmentError("environment has no active snapshot")
    activations = activation_entries(root)
    if reason == "rollback" and (
        len(activations) < 2
        or activations[-1]["generation"] != current
        or activations[-2]["id"] != entry_id
    ):
        raise EnvironmentError("restore previous selection changed; inspect a new plan")
    pointer = root / "store"
    pointer_bytes = pointer.read_bytes()
    store = Path(pointer_bytes.decode("utf-8").strip())
    if pointer.is_symlink() or not store.is_absolute() or store.is_symlink():
        raise EnvironmentError("restore requires an absolute unlinked store")
    settings = (root / "environment.json").read_bytes()
    source = root
    binding = library_root(source)
    fork_settings: dict[str, object] | None = None
    if destination is not None:
        if not binding:
            raise EnvironmentError("fork requires a persistent source library")
        if not destination.is_absolute() or any(
            path.is_symlink() or path.is_junction() for path in (destination, *destination.parents)
        ):
            raise EnvironmentError("fork destination must be absolute and unlinked")
        destination = destination.resolve()
        for retained in (source.resolve(), store.resolve(), Path(binding).resolve()):
            if destination.is_relative_to(retained) or retained.is_relative_to(destination):
                raise EnvironmentError("fork destination overlaps retained environment or data")
        if destination.exists() or not destination.parent.is_dir():
            raise EnvironmentError("fork requires a new destination with an existing parent")
        if select_store(destination, store) != store:
            raise EnvironmentError("fork must use the source wheel store")
        if any(Journal.read(path).incomplete for path in journal_paths(source)):
            raise EnvironmentError("incomplete source operation requires recovery before forking")
        configuration = read_document(source / "environment.json")
        fork_settings = {
            key: configuration[key]
            for key in ("format", "channel", "variant", "mirrors", "packageIndex", "port")
            if key in configuration
        }
        fork_settings["libraryRoot"] = str(Path(binding).resolve())
    try:
        selected_plan = plan_restore(root, entry_id)
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"snapshot recorded by tools {recorded_tools}: {exc}") from exc
    if destination is not None:
        selected_plan["fork"] = {
            "source": str(source),
            "destination": str(destination),
            "sourceCurrent": current,
            "settings": fork_settings,
        }
    alternatives = cast("list[dict[str, object]]", selected_plan["alternatives"])
    approximate = not selected_plan["exact"]
    chosen = next(
        (candidate for candidate in alternatives if candidate["approval"] == approval), None
    )
    if approval is not None and chosen is None:
        raise EnvironmentError("restore approval is stale or does not name a proposed alternative")
    if not approximate and alternatives:
        chosen = alternatives[0]
    planned_snapshot = cast("dict[str, object]", chosen["snapshot"]) if chosen else snapshot
    if fork_settings is not None:
        fork_settings["variant"] = cast("dict[str, object]", planned_snapshot["base"])["variant"]
    if expected_plan is not None and expected_plan != selected_plan:
        raise EnvironmentError("restore proposal changed; inspect and approve a new plan")
    result: dict[str, object] = {
        "current": current,
        "status": "approval-required" if approximate else "planned",
        "restoredFrom": entry_id,
        "recordedTools": recorded_tools,
        "currentTools": version("dinkster-supervisor"),
        "restorePlan": selected_plan,
        "plan": [
            f"rebuild software from history entry {entry_id} as a new snapshot",
            "preserve project data, mounts, settings and the current shared tools",
        ],
    }
    steps = cast("list[str]", result["plan"])
    if destination is not None:
        result.update(destination=str(destination), sourceCurrent=current, current=None)
        steps[:] = [
            f"rebuild history entry {entry_id} in a new stopped environment at {destination}",
            f"retain the source library {binding} and shared wheel store {store}",
            "preserve source software and history; copy no user data, credentials or identity",
        ]

    def compatibility_steps(proposed: dict[str, object]) -> list[str]:
        engine = cast("dict[str, object]", proposed["engine"])
        retained = next((entry for entry in entries if entry["snapshot"] == proposed), None)
        python = interpreter(root / "builds" / str(retained["generation"])) if retained else None
        return library_plan(
            root, str(engine["version"]), python=python if python and python.is_file() else None
        )

    steps.extend(compatibility_steps(planned_snapshot))
    for missing in cast("list[dict[str, object]]", selected_plan["missing"]):
        steps.append(f"unavailable {missing['kind']} {missing['identity']}: {missing['artifact']}")
    if approximate:
        steps.append("nothing changes without --approve followed by one complete proposal digest")
        for alternative in alternatives:
            steps.append(f"alternative {alternative['approval']}")
            proposed = cast("dict[str, object]", alternative["snapshot"])
            steps.extend(compatibility_steps(proposed))
            for change in cast("list[dict[str, object]]", alternative["substitutions"]):
                steps.append(
                    f"substitute {change['field']}: {change['before']} -> {change['after']}"
                )
    if not apply or chosen is None:
        return result
    snapshot = cast("dict[str, object]", chosen["snapshot"])
    substitutions = tuple(cast("list[dict[str, object]]", chosen["substitutions"]))

    target_settings: bytes | None = None

    def revalidate() -> None:
        if (
            read_current(source) != current
            or pointer.read_bytes() != pointer_bytes
            or (source / "environment.json").read_bytes() != settings
            or history(source) != entries
            or library_root(source) != binding
        ):
            raise EnvironmentError("environment changed after planning; request a new plan")
        if destination is not None and any(
            Journal.read(path).incomplete for path in journal_paths(source)
        ):
            raise EnvironmentError("incomplete source operation requires recovery before forking")
        if (
            destination is not None
            and target_settings is not None
            and (
                (destination / "environment.json").read_bytes() != target_settings
                or (destination / "store").read_bytes() != pointer_bytes
                or read_current(destination) is not None
            )
        ):
            raise EnvironmentError("fork destination changed during construction")

    with ExitStack() as locks:
        if destination is not None:
            locks.enter_context(writer_lock(registry_path().parent))
        locks.enter_context(writer_lock(store))
        locks.enter_context(writer_lock(source))
        revalidate()
        validate_import(source, selected)
        for entry in entries:
            if entry["snapshot"] == snapshot:
                validate_import(source, entry)
        if destination is not None:
            assert fork_settings is not None
            if select_store(destination, store) != store:
                raise EnvironmentError("fork store selection changed after planning")
            destination.mkdir()
            root = destination
            locks.enter_context(writer_lock(root))
            target = 1
            journal = begin_setup(
                root,
                target,
                cast("int", fork_settings["port"]),
                registration=setup_registration(root, store),
                start=False,
            )
        else:
            target = next_generation(root)
            journal = begin_activation(root, target, reason, restored_from=entry_id)
        try:
            if destination is not None:
                register_project(root, store)
                write_document(root / "environment.json", cast("dict[str, object]", fork_settings))
                target_settings = (root / "environment.json").read_bytes()
                atomic_write(root / "store", pointer_bytes)
                atomic_write(root / "environment-id", secrets.token_hex(16) + "\n")
            replace(Journal.read(journal), stage="building").write(journal)
            with tempfile.TemporaryDirectory(prefix="dinkster-restore-") as temporary:
                inputs = Path(temporary)
                raw_metadata = snapshot.get("metadata")
                if not isinstance(raw_metadata, dict):
                    raise EnvironmentError("snapshot software metadata must be an object")
                metadata = cast("dict[str, object]", raw_metadata)
                for name, value in metadata.items():
                    if not is_software_metadata(name) or not isinstance(value, str):
                        raise EnvironmentError("snapshot contains unsupported software metadata")
                    path = inputs / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value.encode("utf-8"))
                raw_engine, raw_base = snapshot.get("engine"), snapshot.get("base")
                if not isinstance(raw_engine, dict) or not isinstance(raw_base, dict):
                    raise EnvironmentError("snapshot requires engine and base identities")
                recorded_engine = cast("dict[str, object]", raw_engine)
                recorded_base = cast("dict[str, object]", raw_base)
                engine_id = recorded_engine.get("release")
                base_id = recorded_base.get("id")
                variant = recorded_base.get("variant")
                if not all(isinstance(item, str) for item in (engine_id, base_id, variant)):
                    raise EnvironmentError("snapshot requires string software identities")
                engine_id, base_id, variant = cast(
                    "tuple[str, str, str]", (engine_id, base_id, variant)
                )
                engine_bytes = (inputs / "engine-manifest.json").read_bytes()
                git_source = read_document(inputs / "engine-manifest.json").get("source") == "git"
                local = snapshot.get("source") == "local" or git_source
                if git_source:
                    from .environment_sources import restore_source_inputs

                    restore_source_inputs(inputs, store, engine_id)
                    engine = read_document(inputs / "engine-manifest.json")
                    base = validate_manifest(
                        (inputs / "base-manifest.json").read_bytes(), base_id, variant
                    )
                elif local:
                    engine = read_document(inputs / "engine-manifest.json")
                    base = read_document(inputs / "base-manifest.json")
                    for document in (base, engine):
                        for row in artifact_rows(document.get("wheels"), "filename"):
                            filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
                            cached = (
                                object_path(
                                    store / "wheels",
                                    "sha256:" + sha,
                                    label=readable_label(Path(filename).stem),
                                )
                                / filename
                            )
                            url = row.get("url")
                            roots = (
                                (url.rsplit("/", 1)[0],)
                                if isinstance(url, str) and url.startswith("https://")
                                else (str(cached.parent),)
                            )
                            fetch_artifact(roots, filename, cached, sha, cast("int", row["size"]))
                    write_document(object_path(store / "bases", base_id) / "manifest.json", base)
                    write_document(
                        object_path(store / "engines", engine_id) / "release-manifest.json", engine
                    )
                else:
                    engine = validate_manifest(engine_bytes, engine_id, None)
                    base = validate_manifest(
                        (inputs / "base-manifest.json").read_bytes(), base_id, variant
                    )
                provenance = (
                    read_document(inputs / "artifact-sources.json")
                    if not local or git_source
                    else {}
                )
                channel_document = provenance.get("channel")
                if not local and not isinstance(channel_document, dict):
                    raise EnvironmentError("snapshot has no recorded acquisition channel")
                channel_document = cast("dict[str, object]", channel_document)
                channel = Channel.parse(channel_document) if not local else None
                if (channel is not None and channel.engine_release != engine_id) or engine[
                    "version"
                ] != recorded_engine.get("version"):
                    raise EnvironmentError("snapshot engine identities disagree")
                dependency_variant = provenance.get("dependencyVariant", variant)
                dependency_base = provenance.get("dependencyBase", base)
                if not isinstance(dependency_variant, str) or not isinstance(dependency_base, dict):
                    raise EnvironmentError("snapshot has no dependency base identity")
                sources: dict[str, tuple[str, ...]] = {}
                for key in () if local else ("engine", "base"):
                    values = provenance.get(key)
                    if not isinstance(values, list) or any(
                        not isinstance(value, str) or not value
                        for value in cast("list[object]", values)
                    ):
                        raise EnvironmentError("snapshot has no recorded artifact locations")
                    sources[key] = tuple(cast("list[str]", values))
                catalog = (
                    EnvironmentCatalog(
                        "recorded",
                        channel_document,
                        cast("Channel", channel),
                        engine,
                        sources["engine"],
                        {dependency_variant: cast("dict[str, object]", dependency_base)},
                        {dependency_variant: sources["base"]},
                        engine_bytes,
                    )
                    if not local
                    else None
                )
                custom = base.get("source") == "user-defined"
                package_index = provenance.get("packageIndex")
                if package_index is not None and not isinstance(package_index, str):
                    raise EnvironmentError("snapshot package index must be a string or null")
                constraints = (
                    acquire(
                        cast("EnvironmentCatalog", catalog),
                        dependency_variant,
                        store,
                        package_index=package_index,
                        base=base if custom else None,
                        base_sources=sources["base"] if custom else None,
                        inference=cast("dict[str, object] | None", provenance.get("inference")),
                    )
                    if not local
                    else inputs / "constraints.txt"
                )
                if (
                    content_digest(constraints) != snapshot.get("constraints")
                    or constraints.read_bytes() != (inputs / "constraints.txt").read_bytes()
                ):
                    raise EnvironmentError("reacquired constraints differ from the snapshot")
                if catalog is not None:
                    acquire_sources(catalog, store)
                in_process = snapshot.get("inProcessPacks")
                if not isinstance(in_process, list) or any(
                    not isinstance(pack, str) for pack in cast("list[object]", in_process)
                ):
                    raise EnvironmentError("snapshot in-process packs must be strings")
                protocol = snapshot.get("workerProtocol")
                if (
                    type(protocol) is not int
                    or protocol != base["workerProtocol"]
                    or protocol != engine["workerProtocol"]
                ):
                    raise EnvironmentError("snapshot worker protocols disagree")
                build_generation(
                    root=root,
                    number=target,
                    store=store,
                    base=base_id,
                    engine_version=cast("str", engine["version"]),
                    engine_release=engine_id,
                    constraints=inputs / "constraints.txt",
                    variant=variant,
                    worker_protocol=protocol,
                    in_process_packs=tuple(cast("list[str]", in_process)),
                    isolated_inference_packs=tuple(
                        cast("list[str]", snapshot.get("isolatedInferencePacks", []))
                    ),
                    source_generation=inputs
                    if (inputs / "packs").is_dir()
                    or (inputs / "development-sources.json").is_file()
                    else None,
                    pack_locks=tuple(sorted((inputs / "packs").glob("*/pack-lock.json"))),
                    own_environment_size=sum(
                        path.stat().st_size
                        for path in (object_path(store / "bases", base_id) / "python").rglob("*")
                        if path.is_file()
                    ),
                    catalog_lock=inputs / "dinkster.lock" if "dinkster.lock" in metadata else None,
                    artifacts=object_path(
                        store / "engines", digest(provenance.get("packSourceRelease", engine_id))
                    )
                    / "artifacts"
                    if "dinkster.lock" in metadata
                    else None,
                    acquire_pack_wheels=True,
                    acquire_pack_sources=True,
                    source_kind=cast("str", snapshot["source"]),
                    artifact_sources=provenance if not local or git_source else None,
                )
            entry = prepare_entry(
                root,
                root / "builds" / str(target),
                operation=journal.stem,
                reason="setup" if destination is not None else reason,
                restored_from=None if destination is not None else entry_id,
                substitutions=substitutions,
            )
            entry["recordedTools"] = tools
            rebuilt = cast("dict[str, object]", entry["snapshot"])
            if rebuilt.get("isolatedInferencePacks", []) != snapshot.get(
                "isolatedInferencePacks", []
            ):
                raise EnvironmentError(
                    "rebuilt inference exclusions differ from the recorded snapshot"
                )
            if any(
                rebuilt[key] != snapshot.get(key)
                for key in (
                    "engine",
                    "base",
                    "constraints",
                    "workerProtocol",
                    "source",
                    "inProcessPacks",
                    "metadata",
                )
            ):
                raise EnvironmentError("rebuilt software differs from the recorded snapshot")
            replace(Journal.read(journal), history_entry=entry).write(journal)
            activate(root, journal, revalidate=revalidate)
        except Exception as exc:
            if Journal.read(journal).stage == "complete":
                raise EnvironmentError(
                    "restore activated; history publication requires recovery"
                ) from exc
            message = f"restore entry {entry_id} recorded by tools {recorded_tools}: {exc}"
            replace(Journal.read(journal), error=message).write(journal)
            record_failed_restore(root, journal.stem, entry_id, message)
            recover_activation(root, journal)
            raise EnvironmentError(message) from exc
        result.update(current=target, status="applied", exact=not substitutions)
    return result
