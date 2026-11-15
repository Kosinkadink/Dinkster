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


def installed_release_catalog(generation: Path, channel_name: str) -> EnvironmentCatalog:
    """Keep source operations on the selected generation's release base and code wheelhouse."""
    record = Generation.read(generation / "generation.json")
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
    if record.source == "source":
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
    if selected_protocol is None or int(selected_protocol[1]) != base["workerProtocol"]:
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
        # The frontend remains the selected release artifact, never a checkout workspace.
        frontend: list[str] = []
        for line in (
            published_constraints.read_text(encoding="utf-8").replace("\\\n", "").splitlines()
        ):
            if (
                line.strip()
                and canonicalize_name(
                    Requirement(re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line)).name
                )
                == "dinkster-frontend"
            ):
                frontend.append(line)
        locked += "\n".join(frontend) + "\n"
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
        release = {
            "format": "dinkster.engine/1",
            "source": "git",
            "version": project["project"]["version"],
            "workerProtocol": base["workerProtocol"],
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
        frontend_reference = select_frontend(published, sources.get("channel"))
        if frontend_reference is not None:
            release.update(frontend=frontend_reference, schemaApi=published["schemaApi"])
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
            "worker_protocol": base["workerProtocol"],
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
        manifest["workerProtocol"] != base["workerProtocol"]
        or snapshot["workerProtocol"] != base["workerProtocol"]
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
    with recorded_source_checkout(manifest, store) as checkout:
        _retain_checkout(checkout, engine_root / "checkout", manifest["checkout"]["files"])
    write_document(engine_root / "release-manifest.json", manifest)


def install_source_editables(generation: Path, store: Path, release_id: str) -> None:
    """Install captured workspace code without resolving dependencies a second time."""
    engine_root = object_path(store / "engines", release_id)
    manifest = read_document(generation / "engine-manifest.json")
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
