"""Rebuild recorded software without restoring project data or shared tools."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import cast

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
)
from .environment_restore_plan import plan_restore
from .environment_state import (
    Channel,
    EnvironmentError,
    Journal,
    content_digest,
    digest,
    read_current,
    read_document,
    write_document,
    writer_lock,
)
from .environment_transactions import (
    activate,
    begin_activation,
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
        raise EnvironmentError("project has no active generation")
    activations = activation_entries(root)
    if reason == "rollback" and (
        len(activations) < 2
        or activations[-1]["generation"] != current
        or activations[-2]["id"] != entry_id
    ):
        raise EnvironmentError("rollback selection changed; inspect a new plan")
    pointer = root / "store"
    pointer_bytes = pointer.read_bytes()
    store = Path(pointer_bytes.decode("utf-8").strip())
    if pointer.is_symlink() or not store.is_absolute() or store.is_symlink():
        raise EnvironmentError("restore requires an absolute unlinked store")
    settings = (root / "project.json").read_bytes()
    try:
        selected_plan = plan_restore(root, entry_id)
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"snapshot recorded by tools {recorded_tools}: {exc}") from exc
    if expected_plan is not None and expected_plan != selected_plan:
        raise EnvironmentError("restore proposal changed; inspect and approve a new plan")
    alternatives = cast("list[dict[str, object]]", selected_plan["alternatives"])
    approximate = not selected_plan["exact"]
    chosen = next(
        (candidate for candidate in alternatives if candidate["approval"] == approval), None
    )
    if approval is not None and chosen is None:
        raise EnvironmentError("restore approval is stale or does not name a proposed alternative")
    if not approximate and alternatives:
        chosen = alternatives[0]
    result: dict[str, object] = {
        "current": current,
        "status": "approval-required" if approximate else "planned",
        "restoredFrom": entry_id,
        "recordedTools": recorded_tools,
        "currentTools": version("dinkster-supervisor"),
        "restorePlan": selected_plan,
        "plan": [
            f"rebuild software from history entry {entry_id} as a new generation",
            "preserve project data, mounts, settings and the current shared tools",
        ],
    }
    steps = cast("list[str]", result["plan"])
    for missing in cast("list[dict[str, object]]", selected_plan["missing"]):
        steps.append(f"unavailable {missing['kind']} {missing['identity']}: {missing['artifact']}")
    if approximate:
        steps.append("nothing changes without --approve followed by one complete proposal digest")
        for alternative in alternatives:
            steps.append(f"alternative {alternative['approval']}")
            for change in cast("list[dict[str, object]]", alternative["substitutions"]):
                steps.append(
                    f"substitute {change['field']}: {change['before']} -> {change['after']}"
                )
    if not apply or chosen is None:
        return result
    snapshot = cast("dict[str, object]", chosen["snapshot"])
    substitutions = tuple(cast("list[dict[str, object]]", chosen["substitutions"]))

    def revalidate() -> None:
        if (
            read_current(root) != current
            or pointer.read_bytes() != pointer_bytes
            or (root / "project.json").read_bytes() != settings
            or history(root) != entries
        ):
            raise EnvironmentError("project changed after planning; request a new plan")

    with writer_lock(store), writer_lock(root):
        revalidate()
        target = next_generation(root)
        journal = begin_activation(root, target, reason, restored_from=entry_id)
        try:
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
                local = snapshot.get("source") == "local"
                if local:
                    engine = read_document(inputs / "engine-manifest.json")
                    base = read_document(inputs / "base-manifest.json")
                    for document in (base, engine):
                        for row in artifact_rows(document.get("wheels"), "filename"):
                            filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
                            cached = store / "wheels" / sha / filename
                            url = row.get("url")
                            roots = (
                                (url.rsplit("/", 1)[0],)
                                if isinstance(url, str) and url.startswith("https://")
                                else (str(cached.parent),)
                            )
                            fetch_artifact(roots, filename, cached, sha, cast("int", row["size"]))
                    write_document(store / "bases" / base_id[7:] / "manifest.json", base)
                    write_document(
                        store / "engines" / engine_id[7:] / "release-manifest.json", engine
                    )
                else:
                    engine = validate_manifest(engine_bytes, engine_id, None)
                    base = validate_manifest(
                        (inputs / "base-manifest.json").read_bytes(), base_id, variant
                    )
                provenance = read_document(inputs / "artifact-sources.json") if not local else {}
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
                    source_generation=inputs if (inputs / "packs").is_dir() else None,
                    pack_locks=tuple(sorted((inputs / "packs").glob("*/pack-lock.json"))),
                    own_environment_size=sum(
                        path.stat().st_size
                        for path in (store / "bases" / base_id[7:] / "python").rglob("*")
                        if path.is_file()
                    ),
                    catalog_lock=inputs / "dinkster.lock" if "dinkster.lock" in metadata else None,
                    artifacts=store
                    / "engines"
                    / digest(provenance.get("packSourceRelease", engine_id))[7:]
                    / "artifacts"
                    if "dinkster.lock" in metadata
                    else None,
                    acquire_pack_wheels=True,
                    acquire_pack_sources=True,
                    source_kind="local" if local else "published",
                    artifact_sources=provenance if not local else None,
                )
            entry = prepare_entry(
                root,
                root / "generations" / str(target),
                operation=journal.stem,
                reason=reason,
                restored_from=entry_id,
                substitutions=substitutions,
            )
            entry["recordedTools"] = tools
            rebuilt = cast("dict[str, object]", entry["snapshot"])
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
