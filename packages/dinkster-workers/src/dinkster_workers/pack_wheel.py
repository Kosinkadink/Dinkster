"""Deterministic Python wheels derived from pack manifests."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import keyword
import tomllib
import unicodedata
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .manifest import PackManifest, load_manifest

MANIFEST_FILENAME = "dinkster-pack.toml"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = MAX_ARCHIVE_BYTES
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_ENTRIES = 10_000
MAX_PATH_BYTES = 512
MAX_MANIFEST_BYTES = 1024 * 1024
_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", "build", "dist", "node_modules", "tests"}
)


class PackArchiveError(ValueError):
    """The source cannot be published as a pack wheel."""


def pack_import_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def installed_pack_manifests(
    directory: Path | None = None,
) -> Iterator[tuple[importlib.metadata.Distribution, Path]]:
    """Find pack manifests from RECORD, never by importing their packages."""
    distributions = (
        importlib.metadata.distributions(path=[str(directory)])
        if directory is not None
        else importlib.metadata.distributions()
    )
    for distribution in distributions:
        for file in distribution.files or ():
            if file.name == MANIFEST_FILENAME and len(file.parts) == 2:
                yield distribution, Path(str(distribution.locate_file(file))).resolve()


def _enforce_limit(actual: int, maximum: int, subject: str) -> None:
    if actual > maximum:
        raise PackArchiveError(f"{subject} is {actual} bytes; limit is {maximum} bytes")


def pack_source_directory(manifest: PackManifest) -> Path:
    """Locate the one import package in a checkout or an installed wheel."""
    name = pack_import_name(manifest.name)
    if not name.isidentifier() or keyword.iskeyword(name):
        raise PackArchiveError(f"wheel.import-name: {name!r} is not a Python package name")
    root = manifest.root
    raw = tomllib.loads(manifest.path.read_text(encoding="utf-8"))["pack"]
    for entry in raw.get("entry", {}).values():
        if isinstance(entry, str) and entry.partition(":")[0].partition(".")[0] != name:
            raise PackArchiveError(f"wheel.import-name: entry {entry!r} must belong to {name!r}")
    if root.name == name and (root / "__init__.py").is_file():
        return root
    source = root / "src" if (root / "src").is_dir() else root
    modules = {
        path.name if path.is_dir() else path.stem
        for path in source.iterdir()
        if path.name not in _SKIP_DIRS
        and (path.is_dir() and (path / "__init__.py").is_file() or path.suffix == ".py")
    }
    if modules != {name} or not (source / name / "__init__.py").is_file():
        raise PackArchiveError(
            f"wheel.import-name: pack {manifest.name!r} must ship exactly package {name!r}; "
            f"found {sorted(modules)!r}"
        )
    return source / name


def check_pack_project(manifest: PackManifest) -> None:
    """A supplied project may not give the pack a different identity."""
    for distribution, path in installed_pack_manifests(manifest.root.parent):
        if path != manifest.path.resolve():
            continue
        if distribution.metadata["Name"] != manifest.name:
            raise PackArchiveError(
                f"wheel.distribution-name: distribution {distribution.metadata['Name']!r} "
                f"must equal pack {manifest.name!r}"
            )
    path = manifest.root / "pyproject.toml"
    if not path.is_file():
        return
    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    if project.get("name") != manifest.name:
        raise PackArchiveError(
            f"wheel.distribution-name: distribution {project.get('name')!r} "
            f"must equal pack {manifest.name!r}"
        )
    if project.get("scripts") or project.get("gui-scripts") or project.get("entry-points"):
        raise PackArchiveError("wheel.entry-point-forbidden: packs cannot install entry points")


def wheel_filename(manifest: PackManifest) -> str:
    if manifest.compatibility is None:
        raise PackArchiveError("wheel.invalid: a published pack must declare version and engine")
    distribution = canonicalize_name(manifest.name).replace("-", "_")
    return f"{distribution}-{manifest.compatibility.version}-py3-none-any.whl"


def _collect_entries(manifest: PackManifest) -> list[tuple[str, bytes]]:
    root = manifest.root
    package = pack_source_directory(manifest)
    prefix = pack_import_name(manifest.name)
    entries: dict[str, bytes] = {}
    folded_paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise PackArchiveError(f"{path}: symlinks are not allowed in pack wheels")
        if not path.is_file():
            continue
        if relative.as_posix() in {"pyproject.toml", "uv.lock"}:
            continue
        if path.suffix == ".pth":
            raise PackArchiveError(f"wheel.pth-forbidden: {relative}")
        if path.suffix in {".so", ".pyd", ".dll", ".dylib"}:
            raise PackArchiveError("wheel.invalid: native code requires a platform-tagged wheel")
        inside = path.relative_to(package) if path.is_relative_to(package) else relative
        name = unicodedata.normalize("NFC", f"{prefix}/{inside.as_posix()}")
        if "\\" in name or "\0" in name or ".." in PurePosixPath(name).parts:
            raise PackArchiveError(f"{path}: unsafe wheel path {name!r}")
        _enforce_limit(len(name.encode("utf-8")), MAX_PATH_BYTES, f"archive path {name!r}")
        if name.casefold() in folded_paths:
            raise PackArchiveError(f"{path}: wheel path {name!r} collides after normalization")
        folded_paths.add(name.casefold())
        payload = path.read_bytes()
        _enforce_limit(len(payload), MAX_FILE_BYTES, f"file {name!r}")
        if inside.as_posix() == MANIFEST_FILENAME:
            _enforce_limit(len(payload), MAX_MANIFEST_BYTES, "pack manifest")
        entries[name] = payload
    return sorted(entries.items())


def build_pack_archive(pack_root: Path | str, output: Path | str) -> str:
    """Write a wheel (or its canonical filename in a directory), returning SHA-256."""
    root = Path(pack_root)
    path = root / MANIFEST_FILENAME
    if path.is_symlink():
        raise PackArchiveError(f"{path}: symlinks are not allowed in pack wheels")
    if not path.is_file():
        raise PackArchiveError(f"{root}: no {MANIFEST_FILENAME}; not a pack")
    _enforce_limit(path.stat().st_size, MAX_MANIFEST_BYTES, "pack manifest")
    manifest = load_manifest(path)
    check_pack_project(manifest)
    filename = wheel_filename(manifest)
    destination = Path(output)
    if destination.is_dir() or not destination.suffix:
        destination /= filename
    if destination.name != filename:
        raise PackArchiveError(f"wheel.required: output filename must be {filename!r}")
    assert manifest.compatibility is not None
    entries = _collect_entries(manifest)
    dist_info = filename.removesuffix("-py3-none-any.whl") + ".dist-info"
    metadata = [
        "Metadata-Version: 2.3",
        f"Name: {manifest.name}",
        f"Version: {manifest.compatibility.version}",
    ]
    for raw in manifest.requires:
        metadata.append(f"Requires-Dist: {Requirement(raw)}")
    entries.extend(
        [
            (f"{dist_info}/METADATA", ("\n".join(metadata) + "\n\n").encode("utf-8")),
            (f"{dist_info}/WHEEL", b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
        ]
    )
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, payload in sorted(entries):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", str(len(payload))))
    writer.writerow((f"{dist_info}/RECORD", "", ""))
    entries.append((f"{dist_info}/RECORD", record.getvalue().encode("utf-8")))
    if len(entries) > MAX_ENTRIES:
        raise PackArchiveError(f"archive has {len(entries)} entries; limit is {MAX_ENTRIES}")
    _enforce_limit(sum(len(data) for _, data in entries), MAX_EXPANDED_BYTES, "expanded archive")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(entries):
            _enforce_limit(len(name.encode("utf-8")), MAX_PATH_BYTES, f"archive path {name!r}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    data = buffer.getvalue()
    _enforce_limit(len(data), MAX_ARCHIVE_BYTES, "archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
