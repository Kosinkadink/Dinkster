"""Prepare explicit checkout code against an unchanged acquired release base."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dinkster_store import object_path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

from .environment_artifacts import (
    EnvironmentCatalog,
    acquire_base,
    artifact_rows,
    engine_label,
    supports_base_protocol,
    validate_manifest,
)
from .environment_constraints import (
    effective_constraints,
    selected_wheel_requirements,
    selected_wheel_rows,
    validate_base_requirements,
    wheel_compatibility,
)
from .environment_development import install_live_sources, source_files, source_identity
from .environment_downloads import artifact_available, fetch_artifact
from .environment_frontend import select_frontend
from .environment_local import resolve_wheels, retain_wheels, run_local_command
from .environment_state import (
    Channel,
    EnvironmentError,
    Generation,
    content_digest,
    interpreter,
    read_document,
    relative_file,
    write_document,
)


def source_approval(root: Path, selection: object, expected: str | None = None) -> str:
    """Bind confirmation to selected inputs and the project being changed."""
    from .environment_state import read_current

    root = root.resolve()
    current = read_current(root)
    paths = [root / "environment.json", root / "store"]
    if current is not None:
        paths.append(root / "builds" / str(current) / "build.json")
    approval = source_identity(
        {
            "root": str(root),
            "current": current,
            "project": [content_digest(path) if path.exists() else None for path in paths],
            "selection": selection,
        }
    )
    if expected is not None and expected != approval:
        raise EnvironmentError("source inputs changed from the approved plan; request a new plan")
    return approval


@contextmanager
def acquired_source_engine(args: Any) -> Generator[Any]:
    """Fetch only an explicitly selected managed engine, without enabling develop mode."""
    from .git_sources import fetched_git_source

    value = getattr(args, "engine_source", None)
    if value is None:
        yield args
        return
    selected = copy.copy(args)
    selected.git_mirror = args.git_mirror or os.environ.get("DINKSTER_GIT_MIRROR")
    with fetched_git_source(value, mirror=selected.git_mirror) as (checkout, origin):
        selected.engine_checkout = checkout
        selected.engine_origin = origin
        yield selected


def select_source_pack(
    root: Path,
    *,
    value: str | None,
    pack: str,
    ref: str | None,
    mirror: str | None,
    package_index: str | None,
    apply: bool,
    approval: str | None = None,
) -> dict[str, Any]:
    """Replace only a selected Git pack through ordinary generation provisioning."""
    from urllib.parse import urlencode

    from .environment_generations import (
        build_generation,
        restore_pack_sources,
    )
    from .environment_pack_install import current_software
    from .environment_runtime import require_idle
    from .environment_state import Journal, writer_lock
    from .environment_transactions import (
        activate,
        begin_activation,
        next_generation,
        recover_activation,
    )
    from .git_sources import fetched_git_source, parse_git_source

    root = root.resolve()
    source, generation, store = current_software(root)
    if generation.source == "local":
        raise EnvironmentError("managed Git packs require an installed release base")
    provenance = read_document(source / "artifact-sources.json")
    previous = cast("dict[str, dict[str, Any]]", provenance.get("packs", {}))
    if value is None:
        if not pack or pack not in previous:
            raise EnvironmentError("source pack update requires a recorded Git pack name")
        origin = previous[pack]
        value = (
            origin["url"]
            + "#"
            + urlencode(
                {
                    **({"ref": origin["requestedRef"]} if origin["requestedRef"] else {}),
                    "subdirectory": origin["subdirectory"],
                }
            )
        )
        mirror = mirror or origin.get("gitMirror")
    url, requested, subdirectory = parse_git_source(cast("str", value))
    specification = (
        url
        + "#"
        + urlencode(
            {
                **({"ref": ref or requested} if ref or requested else {}),
                "subdirectory": subdirectory,
            }
        )
    )
    mirror = mirror or os.environ.get("DINKSTER_GIT_MIRROR")
    software = content_digest(source / "build.json")
    settings = (root / "environment.json").read_bytes()
    store_pointer = (root / "store").read_bytes()

    def revalidate() -> None:
        from .environment_state import read_current

        if (
            read_current(root) != generation.number
            or content_digest(source / "build.json") != software
            or generation.drift(source)
            or (root / "environment.json").read_bytes() != settings
            or (root / "store").read_bytes() != store_pointer
        ):
            raise EnvironmentError("environment changed after source planning; request a new plan")

    with fetched_git_source(specification, mirror=mirror) as (checkout, origin):
        name = tomllib.loads((checkout / "dinkster-pack.toml").read_text())["pack"]["name"]
        if pack and pack != name:
            raise EnvironmentError("selected Git checkout names a different pack")
        identity = source_identity(source_files(checkout))
        selected = {
            **origin,
            "kind": "pack",
            "gitMirror": mirror,
            "contentDigest": identity,
            "source": "git:"
            + cast("str", origin["url"])
            + ("#" + urlencode({"subdirectory": subdirectory}) if subdirectory else "")
            + "@"
            + cast("str", origin["commit"]),
        }
        result = {
            "current": generation.number,
            "sources": {"packs": {**previous, name: selected}},
            "sourceApproval": source_approval(
                root,
                {
                    "pack": selected,
                    "packageIndex": package_index
                    or json.loads(settings).get("packageIndex")
                    or "https://pypi.org/simple",
                },
                approval,
            ),
            "kind": "source",
            "published": False,
            "reproducible": False,
            "status": "planned",
            "plan": [
                f"source: unpublished and non-reproducible; Git pack {name}",
                f"repository {origin['url']}; ref {origin['requestedRef']}; "
                f"commit {origin['commit']}",
                "keep the installed engine, release base, frontend and unselected pack sources",
                "resolve only the selected pack against the target engine constraints",
            ],
        }
        if not apply:
            return result
        with writer_lock(store), writer_lock(root):
            revalidate()
            require_idle(root)
            with tempfile.TemporaryDirectory(dir=store, prefix=".source-pack-") as directory:
                scratch = Path(directory)
                artifacts = scratch / "artifacts"
                prepared = json.loads(
                    run_local_command(
                        [
                            str(interpreter(source)),
                            "-I",
                            "-B",
                            "-m",
                            "dinkster_workers.development_sources",
                        ],
                        cwd=source,
                        input=json.dumps(
                            {
                                "generation": str(source),
                                "store": str(store),
                                "artifacts": str(artifacts),
                                "libraries": [],
                                "packs": [str(checkout)],
                                "packPaths": [str(checkout)],
                                "origins": {str(checkout): selected},
                                "installedLock": str(source / "dinkster.lock"),
                                "variant": generation.variant,
                                "accelerator": "cpu"
                                if generation.variant.endswith("cpu")
                                else "mps"
                                if generation.variant == "mac-arm64"
                                else "cuda",
                                "index": package_index
                                or json.loads(settings).get("packageIndex")
                                or "https://pypi.org/simple",
                            }
                        ),
                    )
                )
                if "sources" not in prepared:
                    raise EnvironmentError(
                        "installed engine does not support managed Git pack inputs"
                    )
                if source_identity(source_files(checkout)) != identity:
                    raise EnvironmentError("selected Git pack changed during preparation")
                catalog = artifacts / "development-packs.lock"
                restore_pack_sources(
                    source, catalog_lock=catalog, artifacts=artifacts, project=root, store=store
                )
                provenance["packs"] = {**previous, **prepared["sources"]}
                revalidate()
                target = next_generation(root)
                journal = begin_activation(root, target, "update")
                try:
                    replace(Journal.read(journal), stage="building").write(journal)
                    build_generation(
                        root=root,
                        number=target,
                        store=store,
                        base=generation.base,
                        engine_release=generation.engine_release,
                        engine_version=generation.engine_version,
                        constraints=source / "constraints.txt",
                        variant=generation.variant,
                        worker_protocol=generation.worker_protocol,
                        source_generation=source,
                        pack_locks=tuple(Path(path) for path in prepared["locks"]),
                        in_process_packs=generation.in_process_packs,
                        own_environment_size=sum(
                            path.stat().st_size
                            for path in (
                                object_path(store / "bases", generation.base) / "python"
                            ).rglob("*")
                            if path.is_file()
                        ),
                        catalog_lock=catalog,
                        artifacts=artifacts,
                        acquire_pack_wheels=True,
                        source_kind="source",
                        artifact_sources=provenance,
                    )
                    activate(root, journal, revalidate=revalidate)
                except Exception as exc:
                    replace(Journal.read(journal), error=str(exc)).write(journal)
                    recover_activation(root, journal)
                    raise
                result.update(
                    current=target, status="applied", sources={"packs": provenance["packs"]}
                )
                return result


def select_source_library(
    root: Path,
    *,
    value: str | None,
    library: str,
    ref: str | None,
    mirror: str | None,
    package_index: str | None,
    apply: bool,
    frontend: bool = False,
    approval: str | None = None,
) -> dict[str, Any]:
    """Replace one shared library while retaining the installed base and all other pins."""
    from urllib.parse import urlencode

    from .environment_generations import build_generation, restore_pack_sources
    from .environment_pack_install import current_software
    from .environment_runtime import require_idle
    from .environment_state import Journal, read_current, writer_lock
    from .environment_transactions import (
        activate,
        begin_activation,
        next_generation,
        recover_activation,
    )
    from .git_sources import fetched_git_source, parse_git_source

    root = root.resolve()
    source, generation, store = current_software(root)
    catalog = installed_release_catalog(source, "main")
    manifest = cast("dict[str, Any]", read_document(source / "engine-manifest.json"))
    sources = copy.deepcopy(cast("dict[str, Any]", manifest.get("sources", {})))
    libraries = sources.setdefault("libraries", {})
    if value is None:
        if frontend:
            if "frontend" not in sources:
                raise EnvironmentError("source frontend update requires a recorded Git frontend")
            origin = sources["frontend"]
        else:
            if not library or library not in libraries:
                raise EnvironmentError("source library update requires a recorded Git library name")
            origin = libraries[library]
        value = (
            origin["url"]
            + "#"
            + urlencode(
                {
                    "ref": origin["requestedRef"] or origin["commit"],
                    "subdirectory": origin["subdirectory"],
                }
            )
        )
        mirror = mirror or origin.get("gitMirror")
    url, requested, subdirectory = parse_git_source(cast("str", value))
    specification = (
        url
        + "#"
        + urlencode(
            {
                **({"ref": ref or requested} if ref or requested else {}),
                "subdirectory": subdirectory,
            }
        )
    )
    mirror = mirror or os.environ.get("DINKSTER_GIT_MIRROR")
    software = content_digest(source / "build.json")
    settings = (root / "environment.json").read_bytes()
    store_pointer = (root / "store").read_bytes()

    def revalidate() -> None:
        if (
            read_current(root) != generation.number
            or content_digest(source / "build.json") != software
            or generation.drift(source)
            or (root / "environment.json").read_bytes() != settings
            or (root / "store").read_bytes() != store_pointer
        ):
            raise EnvironmentError("environment changed after source planning; request a new plan")

    with fetched_git_source(specification, mirror=mirror) as (checkout, origin):
        if (checkout / "dinkster-pack.toml").exists():
            raise EnvironmentError("Git node packs require --pack, not --library")
        artifact = None
        if frontend:
            from .environment_frontend_sources import frontend_ci_artifact

            name = "dinkster-frontend"
            api = json.loads((checkout / "frontend-release.json").read_text())["engineApi"]
            schema = manifest.get("schemaApi")
            if (
                not isinstance(api, dict)
                or type(cast("dict[str, Any]", api).get("min")) is not int
                or type(cast("dict[str, Any]", api).get("max")) is not int
                or type(schema) is not int
                or not 1 <= api["min"] <= schema <= api["max"]
            ):
                raise EnvironmentError(
                    "Git frontend does not support the installed engine schema API"
                )
            artifact = frontend_ci_artifact(origin)
        else:
            project = tomllib.loads((checkout / "pyproject.toml").read_text(encoding="utf-8"))
            name = canonicalize_name(project["project"]["name"])
        if library and canonicalize_name(library) != name:
            raise EnvironmentError("selected Git checkout names a different library")
        protected = set(cast("list[str]", catalog.bases[generation.variant]["replaces"]))
        protected.add("dinkster-frontend")
        protected.update(
            parse_wheel_filename(str(row["name"]))[0]
            for row in artifact_rows(catalog.engine["artifacts"], "name")
            if str(row["name"]).endswith(".whl")
        )
        if "checkout" in manifest:
            with recorded_source_checkout(manifest, store) as engine_checkout:
                for package in manifest["checkout"]["packageRoots"]:
                    metadata = tomllib.loads(
                        (engine_checkout / package / "pyproject.toml").read_text(encoding="utf-8")
                    )
                    protected.add(canonicalize_name(metadata["project"]["name"]))
        if name in protected and not frontend:
            raise EnvironmentError(
                "Git libraries must not replace release-base, engine or frontend packages"
            )
        files = source_files(checkout)
        selected = {
            **origin,
            "kind": "frontend" if frontend else "library",
            "gitMirror": mirror,
            "contentDigest": source_identity(files),
        }
        if frontend:
            sources["frontend"] = selected
        else:
            libraries[name] = selected
        result = {
            "current": generation.number,
            "sources": sources,
            "sourceApproval": source_approval(
                root,
                {
                    "selected": selected,
                    "frontendArtifact": {
                        key: artifact[key]
                        for key in ("id", "run", "name", "digest", "size_in_bytes")
                    }
                    if artifact is not None
                    else None,
                    "packageIndex": package_index
                    or json.loads(settings).get("packageIndex")
                    or "https://pypi.org/simple",
                },
                approval,
            ),
            "kind": "source",
            "published": False,
            "reproducible": False,
            "status": "planned",
            "plan": [
                f"source: unpublished and non-reproducible; Git {selected['kind']} {name}",
                f"repository {origin['url']}; ref {origin['requestedRef']}; "
                f"commit {origin['commit']}",
                "keep the installed release base, engine and unselected libraries and packs",
                (
                    f"use verified CI bundle {artifact['id']} from run {artifact['run']}"
                    if artifact is not None
                    else "build frontend locally with Node and pnpm (no CI bundle for this commit)"
                )
                if frontend
                else "build the selected library against retained exact pins; keep frontend",
            ],
        }
        if not apply:
            return result
        with writer_lock(store), writer_lock(root):
            revalidate()
            require_idle(root)
            with tempfile.TemporaryDirectory(dir=store, prefix=".source-library-") as directory:
                work = Path(directory)
                captured = work / "checkout"
                _retain_checkout(checkout, captured, files)
                if frontend:
                    from zipfile import BadZipFile

                    from .environment_frontend_sources import prepare_frontend_wheel

                    try:
                        built_frontend, build = prepare_frontend_wheel(
                            captured, origin, work / "built", artifact
                        )
                    except (BadZipFile, KeyError, TypeError) as exc:
                        raise EnvironmentError(
                            f"invalid Git frontend build metadata: {exc}"
                        ) from exc
                    selected["build"] = build
                    built = (built_frontend,)
                else:
                    run_local_command(
                        [
                            "uv",
                            "build",
                            "--wheel",
                            "--no-sources",
                            "--out-dir",
                            str(work / "built"),
                            str(captured),
                        ],
                        cwd=captured,
                    )
                    built = tuple((work / "built").glob("*.whl"))
                if len(built) != 1 or parse_wheel_filename(built[0].name)[0] != name:
                    raise EnvironmentError(
                        "selected Git library must build exactly its named wheel"
                    )
                wheels = resolve_library_wheels(
                    source / "constraints.txt",
                    built,
                    base=read_document(source / "base-manifest.json"),
                    store=store,
                    package_index=package_index
                    or json.loads(settings).get("packageIndex")
                    or "https://pypi.org/simple",
                )
                installed = {
                    row["filename"]: row
                    for row in cast(
                        "list[dict[str, Any]]",
                        read_document(source / "installed-wheels.json")["wheels"],
                    )
                }
                retained = {row["filename"]: row for row in manifest.get("wheels", [])}
                for row in wheels:
                    previous = installed.get(row["filename"])
                    if row["name"] == name:
                        row["registry"] = origin["url"]
                    elif previous is not None and row["sha256"] == previous["sha256"]:
                        row["registry"] = previous.get("registry")
                        if row["filename"] in retained:
                            row["url"] = retained[row["filename"]]["url"]
                    else:
                        row["registry"] = row["url"]
                wheel = next(row for row in wheels if row["name"] == name)
                if wheel["sha256"] != content_digest(built[0])[7:]:
                    raise EnvironmentError("resolved library differs from the selected Git build")
                selected["wheel"] = wheel
                release: dict[str, Any] = {
                    **manifest,
                    "source": "git",
                    "kind": "source",
                    "published": False,
                    "reproducible": False,
                    "publishedRelease": catalog.channel.engine_release,
                    "publishedManifest": catalog.engine_bytes.decode("utf-8"),
                    "publishedConstraints": manifest.get(
                        "publishedConstraints", (source / "constraints.txt").read_text()
                    ),
                    "sources": sources,
                    "wheels": wheels,
                }
                # The producer lock is retained in publishedManifest; this selection owns wheels.
                release.pop("dependencies", None)
                if frontend:
                    release.pop("frontend", None)
                release_id = source_identity(release)
                engine_root = object_path(store / "engines", release_id)
                engine_root.mkdir(parents=True, exist_ok=True)
                if "checkout" in manifest:
                    with recorded_source_checkout(manifest, store) as engine_checkout:
                        _retain_checkout(
                            engine_checkout, engine_root / "checkout", manifest["checkout"]["files"]
                        )
                write_document(engine_root / "release-manifest.json", release)
                constraints = engine_root / f"constraints-{generation.variant}.txt"
                constraints.write_text(
                    "\n".join(
                        f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}"
                        for row in wheels
                    )
                    + "\n"
                )
                provenance = read_document(source / "artifact-sources.json")
                if name == "dinkster-inference":
                    provenance.pop("inference", None)
                artifacts = work / "artifacts"
                artifacts.mkdir()
                restore_pack_sources(
                    source,
                    catalog_lock=source / "dinkster.lock",
                    artifacts=artifacts,
                    project=root,
                    store=store,
                )
                revalidate()
                target = next_generation(root)
                journal = begin_activation(root, target, "update")
                try:
                    replace(Journal.read(journal), stage="building").write(journal)
                    build_generation(
                        root=root,
                        number=target,
                        store=store,
                        base=generation.base,
                        engine_release=release_id,
                        engine_version=generation.engine_version,
                        constraints=constraints,
                        variant=generation.variant,
                        worker_protocol=generation.worker_protocol,
                        source_generation=source,
                        in_process_packs=generation.in_process_packs,
                        own_environment_size=sum(
                            path.stat().st_size
                            for path in (
                                object_path(store / "bases", generation.base) / "python"
                            ).rglob("*")
                            if path.is_file()
                        ),
                        catalog_lock=source / "dinkster.lock",
                        artifacts=artifacts,
                        source_kind="source",
                        artifact_sources=provenance,
                    )
                    activate(root, journal, revalidate=revalidate)
                except Exception as exc:
                    replace(Journal.read(journal), error=str(exc)).write(journal)
                    recover_activation(root, journal)
                    raise
                result.update(current=target, status="applied", sources=sources)
                return result


def resolve_library_wheels(
    constraints: Path,
    libraries: tuple[Path, ...],
    *,
    base: dict[str, Any],
    store: Path,
    package_index: str,
) -> list[dict[str, Any]]:
    """Replace selected pins with exact local wheels while fixing every unselected byte."""
    selected = {parse_wheel_filename(path.name)[0]: path for path in libraries}
    if len(selected) != len(libraries):
        raise EnvironmentError("duplicate selected source library")
    if selected.keys() & set(base["replaces"]):
        raise EnvironmentError("Git libraries must not replace release-base packages")
    python = interpreter(object_path(store / "bases", base["id"]))
    compatibility = wheel_compatibility(python)
    locked = (
        "\n".join(
            line
            for line in constraints.read_text(encoding="utf-8").replace("\\\n", "").splitlines()
            if line.strip()
            and canonicalize_name(
                Requirement(re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line)).name
            )
            not in selected
        )
        + "\n"
    )
    cached, _ = selected_wheel_requirements(
        python,
        locked,
        sorted({path.parent for path in (store / "wheels").rglob("*.whl")}),
        compatibility=compatibility,
    )
    with tempfile.TemporaryDirectory(dir=store, prefix=".library-wheels-") as directory:
        work = Path(directory)
        pins, inputs, override = (
            work / filename for filename in ("pins.txt", "requirements.in", "override.txt")
        )
        pins.write_text(re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", locked), encoding="utf-8")
        inputs.write_text(
            cached + "\n".join(path.resolve().as_uri() for path in libraries) + "\n",
            encoding="utf-8",
        )
        override.write_text(
            "\n".join(f"{name} @ {path.resolve().as_uri()}" for name, path in selected.items())
            + "\n",
            encoding="utf-8",
        )
        resolution = resolve_wheels(
            inputs,
            pins,
            work / "pylock.toml",
            python=python,
            index=package_index,
            source=work,
            overrides=override,
        )
        wheels = retain_wheels(resolution, python, store, work)
    selected_wheel_rows(locked, wheels, compatibility)
    for name, path in selected.items():
        if not any(
            row["name"] == name and row["sha256"] == content_digest(path)[7:] for row in wheels
        ):
            raise EnvironmentError("resolved library differs from the retained Git build")
    validate_base_requirements(
        tuple(
            object_path(store / "wheels", "sha256:" + row["sha256"]) / row["filename"]
            for row in wheels
        ),
        base,
    )
    return wheels


def retained_source_libraries(
    generation: Path,
    store: Path,
    *,
    replace_inference: bool = False,
    replace_frontend: bool = False,
) -> dict[str, Any]:
    """Read selected Git library identities without consulting moving refs or rebuilding."""
    record = Generation.read(generation / "build.json")
    path = generation / "engine-manifest.json"
    if not path.is_file():
        return {}
    if content_digest(path) != dict(record.files).get(path.name):
        raise EnvironmentError("retained source identities differ from snapshot")
    manifest = read_document(path)
    sources = cast("dict[str, Any]", manifest.get("sources", {}))
    libraries = copy.deepcopy(cast("dict[str, Any]", sources.get("libraries", {})))
    if not replace_frontend and "frontend" in sources:
        libraries["dinkster-frontend"] = copy.deepcopy(sources["frontend"])
    if replace_inference:
        libraries.pop("dinkster-inference", None)
    for origin in libraries.values():
        row = origin["wheel"]
        wheel = object_path(store / "wheels", "sha256:" + row["sha256"]) / row["filename"]
        if (
            wheel.is_symlink()
            or not wheel.is_file()
            or wheel.stat().st_size != row["size"]
            or content_digest(wheel) != "sha256:" + row["sha256"]
        ):
            raise EnvironmentError(
                f"retained Git library wheel missing or corrupt: {row['filename']}"
            )
    return libraries


def prepare_retained_libraries(
    libraries: dict[str, Any],
    *,
    constraints: Path,
    catalog: EnvironmentCatalog,
    base_id: str,
    variant: str,
    store: Path,
    package_index: str,
    engine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply retained Git library builds to the target release before pack planning."""
    if engine is None:
        manifest = copy.deepcopy(cast("dict[str, Any]", catalog.engine))
    else:
        manifest = cast(
            "dict[str, Any]",
            read_document(
                object_path(store / "engines", engine["engine_release"]) / "release-manifest.json"
            ),
        )
    paths = tuple(
        object_path(store / "wheels", "sha256:" + origin["wheel"]["sha256"])
        / origin["wheel"]["filename"]
        for origin in libraries.values()
    )
    base = read_document(object_path(store / "bases", base_id) / "manifest.json")
    wheels = resolve_library_wheels(
        constraints, paths, base=base, store=store, package_index=package_index
    )
    retained = {row["filename"]: row for row in manifest.get("wheels", [])}
    for row in wheels:
        if row["name"] in libraries:
            row.update(libraries[row["name"]]["wheel"])
        elif row["filename"] in retained and retained[row["filename"]]["sha256"] == row["sha256"]:
            row.update(retained[row["filename"]])
    original = copy.deepcopy(manifest)
    manifest.update(
        source="git",
        kind="source",
        published=False,
        reproducible=False,
        publishedRelease=catalog.channel.engine_release,
        publishedManifest=catalog.engine_bytes.decode("utf-8"),
        publishedConstraints=manifest.get(
            "publishedConstraints", constraints.read_text(encoding="utf-8")
        ),
        sources={
            **manifest.get("sources", {}),
            "libraries": {
                name: origin for name, origin in libraries.items() if name != "dinkster-frontend"
            },
            **(
                {"frontend": libraries["dinkster-frontend"]}
                if "dinkster-frontend" in libraries
                else {}
            ),
        },
        wheels=wheels,
    )
    if "dinkster-frontend" in libraries:
        manifest.pop("frontend", None)
    manifest.pop("dependencies", None)
    identity = source_identity(manifest)
    target = object_path(store / "engines", identity)
    target.mkdir(parents=True, exist_ok=True)
    if "checkout" in original:
        with recorded_source_checkout(original, store) as checkout:
            _retain_checkout(checkout, target / "checkout", original["checkout"]["files"])
    write_document(target / "release-manifest.json", manifest)
    selected = target / f"constraints-{variant}.txt"
    selected.write_text(
        "\n".join(
            f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}" for row in wheels
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "engine_release": identity,
        "engine_version": manifest["version"],
        "worker_protocol": manifest["workerProtocol"],
        "constraints": selected,
        "source_kind": "source",
    }


def installed_release_catalog(generation: Path, channel_name: str) -> EnvironmentCatalog:
    """Keep source operations on the selected generation's release base and code wheelhouse."""
    record = Generation.read(generation / "build.json")
    if record.source == "local":
        raise EnvironmentError("managed Git sources require an installed release base")
    expected = dict(record.files)
    for name in ("engine-manifest.json", "base-manifest.json", "artifact-sources.json"):
        path = generation / name
        if path.is_symlink() or content_digest(path) != expected.get(name):
            raise EnvironmentError(f"source selection requires intact installed metadata: {name}")
    base = validate_manifest(
        (generation / "base-manifest.json").read_bytes(), record.base, record.variant
    )
    manifest = read_document(generation / "engine-manifest.json")
    if manifest.get("source") == "git":
        if source_identity(manifest) != record.engine_release:
            raise EnvironmentError("source engine manifest identity mismatch")
        engine_bytes = cast("str", manifest["publishedManifest"]).encode("utf-8")
        engine_id = cast("str", manifest["publishedRelease"])
    else:
        engine_bytes = (generation / "engine-manifest.json").read_bytes()
        engine_id = record.engine_release
    engine = validate_manifest(engine_bytes, engine_id, None)
    provenance = read_document(generation / "artifact-sources.json")
    channel_document = cast("dict[str, object]", provenance["channel"])
    channel = Channel.parse(channel_document)
    if channel.engine_release != engine_id or channel.base_for(record.variant) != record.base:
        raise EnvironmentError("source selection requires its matching installed release channel")
    return EnvironmentCatalog(
        channel_name,
        channel_document,
        channel,
        engine,
        tuple(cast("list[str]", provenance["engine"])),
        {record.variant: base},
        {record.variant: tuple(cast("list[str]", provenance["base"]))},
        engine_bytes,
    )


def prepare_source_engine(
    source: Path,
    *,
    store: Path,
    base_id: str,
    published_release: str,
    published_constraints: Path,
    variant: str,
    sources: dict[str, Any],
    package_index: str = "https://pypi.org/simple",
    library_wheels: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Resolve checkout code once, reusing exact release wheels without rebuilding its base."""
    base_root = object_path(store / "bases", base_id)
    base = read_document(base_root / "manifest.json")
    python = interpreter(base_root)
    compatibility = wheel_compatibility(python)
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    files = source_files(source)
    checkout_identity = source_identity(files)
    lock = tomllib.loads((source / "uv.lock").read_text(encoding="utf-8"))
    package_roots = sorted(
        {row["source"]["editable"] for row in lock["package"] if "editable" in row["source"]}
    )
    if not package_roots or any(
        not (source / path).resolve().is_relative_to(source.resolve()) for path in package_roots
    ):
        raise EnvironmentError("source workspace packages must stay within the checkout")
    lock_digest = content_digest(source / "uv.lock")
    published = read_document(
        object_path(store / "engines", published_release) / "release-manifest.json"
    )
    protocol = (source / "packages/dinkster-workers/src/dinkster_workers/boundary.py").read_text()
    selected_protocol = re.search(r"(?m)^PROTOCOL_VERSION = ([0-9]+)$", protocol)
    if selected_protocol is None:
        raise EnvironmentError("source must declare its worker protocol")
    worker_protocol = int(selected_protocol[1])
    base_protocols = (
        project.get("tool", {}).get("dinkster", {}).get("base-worker-protocols", [worker_protocol])
    )
    if not supports_base_protocol(
        {"workerProtocol": worker_protocol, "baseWorkerProtocols": base_protocols}, base
    ):
        raise EnvironmentError("source worker protocol does not match the installed release base")
    with tempfile.TemporaryDirectory(dir=store, prefix=".source-engine-") as directory:
        work = Path(directory)
        exported = work / "export.txt"
        run_local_command(
            [
                "uv",
                "export",
                "--python",
                str(python),
                "--locked",
                "--all-packages",
                "--no-dev",
                "--group",
                "engine-base",
                "--no-emit-workspace",
                "--no-header",
                "--no-annotate",
                "--output-file",
                str(exported),
            ],
            cwd=source,
        )
        if content_digest(source / "uv.lock") != lock_digest:
            raise EnvironmentError("source lock changed during preparation")
        overrides = {parse_wheel_filename(path.name)[0]: path for path in library_wheels}
        if len(overrides) != len(library_wheels):
            raise EnvironmentError("duplicate selected source library")
        locked = effective_constraints(exported.read_text(encoding="utf-8"), base)
        lines: list[str] = []
        for line in locked.replace("\\\n", "").splitlines():
            if not line.strip():
                continue
            requirement = Requirement(re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line))
            if canonicalize_name(requirement.name) not in overrides:
                lines.append(line)
        locked = "\n".join(lines) + "\n"
        # These optional selections are independent of the checkout's core dependencies.
        optional_requirements: list[str] = []
        for line in (
            published_constraints.read_text(encoding="utf-8").replace("\\\n", "").splitlines()
        ):
            if not line.strip():
                continue
            name = canonicalize_name(
                Requirement(re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line)).name
            )
            if name in {"dinkster-frontend", "dinkster-inference"} and name not in overrides:
                optional_requirements.append(line)
        locked += "\n".join(optional_requirements) + "\n"
        wheel_directories = [
            base_root / "wheels",
            *sorted({path.parent for path in (store / "wheels").rglob("*.whl")}),
        ]
        cached, _ = selected_wheel_requirements(
            python, locked, wheel_directories, errors=[], compatibility=compatibility
        )
        built = work / "built"
        run_local_command(
            [
                "uv",
                "build",
                "--python",
                str(python),
                "--all-packages",
                "--wheel",
                "--out-dir",
                str(built),
            ],
            cwd=source,
        )
        workspace_wheels = sorted(built.glob("*.whl"))
        if any(
            parse_wheel_filename(path.name)[0] == "dinkster-frontend" for path in workspace_wheels
        ):
            raise EnvironmentError("source engine must not replace the pinned frontend")
        requirements, constraints = work / "engine.in", work / "pins.txt"
        constraints.write_text(
            re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", locked), encoding="utf-8"
        )
        requirements.write_text(
            "\n".join(path.resolve().as_uri() for path in (*workspace_wheels, *library_wheels))
            + "\n"
            + re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", cached)
            + constraints.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        override_file = None
        if overrides:
            override_file = work / "overrides.txt"
            override_file.write_text(
                "\n".join(f"{name} @ {path.resolve().as_uri()}" for name, path in overrides.items())
                + "\n",
                encoding="utf-8",
            )
        resolution = resolve_wheels(
            requirements,
            constraints,
            work / "pylock.toml",
            python=python,
            index=package_index,
            source=source,
            overrides=override_file,
        )
        wheels = retain_wheels(resolution, python, store, source)
        # A matching version alone is not authority to substitute different lock bytes.
        selected_wheel_rows(locked, wheels, compatibility)
        paths = tuple(
            object_path(store / "wheels", "sha256:" + row["sha256"]) / row["filename"]
            for row in wheels
        )
        validate_base_requirements(paths, base)
        release: dict[str, Any] = {
            "format": "dinkster.engine/1",
            "source": "git",
            "version": project["project"]["version"],
            "capabilities": project.get("tool", {})
            .get("dinkster", {})
            .get("engine-capabilities", []),
            "workerProtocol": worker_protocol,
            "baseWorkerProtocols": base_protocols,
            "libraryApi": project.get("tool", {}).get("dinkster", {}).get("library-api"),
            "libraryCapabilities": project.get("tool", {})
            .get("dinkster", {})
            .get("library-capabilities", []),
            "lock": lock_digest,
            "publishedRelease": published_release,
            "publishedManifest": (
                object_path(store / "engines", published_release) / "release-manifest.json"
            ).read_text(encoding="utf-8"),
            "publishedConstraints": published_constraints.read_text(encoding="utf-8"),
            "checkout": {
                "identity": checkout_identity,
                "files": files,
                "packageRoots": package_roots,
            },
            "kind": "source",
            "published": False,
            "reproducible": False,
            "sources": sources,
            "wheels": wheels,
        }
        if "schemaApi" in published:
            release["schemaApi"] = published["schemaApi"]
        frontend_reference = select_frontend(published, sources.get("channel"))
        if frontend_reference is not None:
            release["frontend"] = frontend_reference
        release_id = (
            "sha256:" + hashlib.sha256(json.dumps(release, sort_keys=True).encode()).hexdigest()
        )
        engine_root = object_path(store / "engines", release_id, label=engine_label(release))
        engine_root.mkdir(parents=True, exist_ok=True)
        checkout = engine_root / "checkout"
        if source_files(source) != files:
            raise EnvironmentError("source checkout changed during preparation")
        _retain_checkout(source, checkout, files)
        write_document(engine_root / "release-manifest.json", release)
        retained = engine_root / f"constraints-{variant}.txt"
        retained.write_text(
            "\n".join(
                f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}" for row in wheels
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "base": base_id,
            "engine_release": release_id,
            "constraints": retained,
            "engine_version": project["project"]["version"],
            "variant": variant,
            "worker_protocol": worker_protocol,
            "source_kind": "source",
        }


def _retain_checkout(source: Path, checkout: Path, files: dict[str, str]) -> None:
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checkout.parent, prefix=".checkout-") as directory:
            captured = Path(directory) / "tree"
            captured.mkdir()
            for name in files:
                relative_file(name)
                target = captured / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
            if source_files(captured) != files:
                raise EnvironmentError("source checkout changed during capture")
            shutil.move(str(captured), checkout)
    if source_files(checkout) != files:
        raise EnvironmentError("retained source checkout differs from its recorded identity")


@contextmanager
def recorded_source_checkout(manifest: dict[str, Any], store: Path) -> Generator[Path]:
    """Reuse captured code or fetch only the recorded commit, never its moving branch."""
    from urllib.parse import urlencode

    from .git_sources import fetched_git_source

    checkout = object_path(store / "engines", source_identity(manifest)) / "checkout"
    if checkout.exists():
        if source_files(checkout) != manifest["checkout"]["files"]:
            raise EnvironmentError("retained source checkout differs from its recorded identity")
        yield checkout
        return
    origin = manifest["sources"]["engine"]
    value = (
        origin["url"]
        + "#"
        + urlencode(
            {
                "ref": origin["commit"],
                "subdirectory": origin["subdirectory"],
            }
        )
    )
    with fetched_git_source(value, mirror=manifest["sources"].get("gitMirror")) as (
        source,
        resolved,
    ):
        if (
            resolved["commit"] != origin["commit"]
            or source_files(source) != manifest["checkout"]["files"]
        ):
            raise EnvironmentError("recorded Git commit differs from the captured source identity")
        yield source


def source_missing_inputs(
    snapshot: dict[str, object], store: Path, *, offline: bool = False
) -> list[dict[str, object]]:
    """Preview exact source recovery without changing retained software or following refs."""
    metadata = cast("dict[str, str]", snapshot["metadata"])
    manifest = json.loads(metadata["engine-manifest.json"])
    engine_id = cast("dict[str, str]", snapshot["engine"])["release"]
    base_record = cast("dict[str, str]", snapshot["base"])
    if source_identity(manifest) != engine_id or manifest.get("source") != "git":
        raise EnvironmentError("source snapshot manifest identity mismatch")
    base = validate_manifest(
        metadata["base-manifest.json"].encode("utf-8"), base_record["id"], base_record["variant"]
    )
    if (
        not supports_base_protocol(manifest, base)
        or snapshot["workerProtocol"] != manifest["workerProtocol"]
    ):
        raise EnvironmentError("source snapshot worker protocols disagree")
    provenance = json.loads(metadata["artifact-sources.json"])
    missing: list[dict[str, object]] = []
    base_root = object_path(store / "bases", base_record["id"])
    if not base_root.is_dir():
        for row in artifact_rows(cast("dict[str, object]", base["archive"])["parts"], "filename"):
            filename = cast("str", row["filename"])
            cached = object_path(store / "downloads", base_record["id"]) / filename
            if not artifact_available(
                (str(cached.parent),) if offline else tuple(provenance["base"]),
                filename,
                cached,
                cast("str", row["sha256"]),
                cast("int", row["size"]),
            ):
                missing.append(
                    {"kind": "base", "identity": base_record["id"], "artifact": filename}
                )
    for row in artifact_rows(manifest["wheels"], "filename"):
        filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
        cached = object_path(store / "wheels", "sha256:" + sha) / filename
        url = row.get("url")
        roots = (
            (url.rsplit("/", 1)[0],)
            if not offline and isinstance(url, str) and url.startswith("https://")
            else (str(cached.parent),)
        )
        if not artifact_available(roots, filename, cached, sha, cast("int", row["size"])):
            missing.append(
                {
                    "kind": "source wheel",
                    "identity": engine_id,
                    "artifact": filename,
                    "sha256": sha,
                    "locations": list(roots),
                }
            )
    if "checkout" not in manifest:
        return missing
    try:
        checkout = object_path(store / "engines", engine_id) / "checkout"
        if offline and not checkout.is_dir():
            raise EnvironmentError("recorded checkout is not retained")
        with recorded_source_checkout(manifest, store):
            pass
    except (OSError, ValueError) as error:
        missing.append(
            {
                "kind": "Git source",
                "identity": engine_id,
                "artifact": manifest["sources"]["engine"]["commit"],
                "error": str(error),
            }
        )
    return missing


def restore_source_inputs(inputs: Path, store: Path, engine_id: str) -> None:
    """Acquire recorded base/wheels and captured code without resolving Git branches or indexes."""
    manifest = cast("dict[str, Any]", read_document(inputs / "engine-manifest.json"))
    base = read_document(inputs / "base-manifest.json")
    provenance = read_document(inputs / "artifact-sources.json")
    if source_identity(manifest) != engine_id:
        raise EnvironmentError("source snapshot manifest identity mismatch")
    acquire_base(base, tuple(cast("list[str]", provenance["base"])), store)
    for row in artifact_rows(manifest["wheels"], "filename"):
        filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
        cached = object_path(store / "wheels", "sha256:" + sha) / filename
        url = row.get("url")
        roots = (
            (url.rsplit("/", 1)[0],)
            if isinstance(url, str) and url.startswith("https://")
            else (str(cached.parent),)
        )
        fetch_artifact(roots, filename, cached, sha, cast("int", row["size"]))
    engine_root = object_path(store / "engines", engine_id)
    if "checkout" in manifest:
        with recorded_source_checkout(manifest, store) as checkout:
            _retain_checkout(checkout, engine_root / "checkout", manifest["checkout"]["files"])
    write_document(engine_root / "release-manifest.json", manifest)


def install_source_editables(generation: Path, store: Path, release_id: str) -> None:
    """Install captured workspace code without resolving dependencies a second time."""
    engine_root = object_path(store / "engines", release_id)
    manifest = read_document(generation / "engine-manifest.json")
    if "checkout" not in manifest:
        return
    checkout = engine_root / "checkout"
    selected = cast("dict[str, Any]", manifest["checkout"])
    if source_identity(manifest) != release_id or source_files(checkout) != selected["files"]:
        raise EnvironmentError("retained source checkout differs from its recorded identity")
    package_roots = selected["packageRoots"]
    for path in package_roots:
        if path != ".":
            relative_file(path)
    install_live_sources(
        generation,
        {
            "sources": [
                {
                    "kind": "engine",
                    "live": True,
                    "path": str(checkout),
                    "packageRoots": package_roots,
                }
            ]
        },
    )
