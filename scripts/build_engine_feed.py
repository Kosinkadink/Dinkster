"""Build the Dinkster engine feed: per-cell base archives and code wheelhouses.

The feed is a directory that a Dinkster installer consumes offline:

    base/<cell>/<base-id>.tar.gz   relocatable interpreter + torch base archive
    store/<sha256>                 content-addressed wheel files
    engine/<commit>/<cell>.json    code-layer manifest for one commit and cell
    channels/<channel>.json        stable or github-live channel pointer

Each cell (OS x accelerator) builds natively on that OS: the script refuses to
cross-build. The base archive contains a python-build-standalone interpreter
installed by uv, exactly the pinned torch/torchvision build for the cell and
the transitive closure that resolution actually installs, and a pinned uv
executable the installer uses with UV_OFFLINE=1. The base id hashes only the
canonical cell/platform/interpreter identity, bundled uv digest and resolved
base closure, so kitchen, aimdo and every other code-layer pin can change
without a new base. The code layer is the hash-locked wheelhouse uv.lock names
for the cell through marker-true runtime edges, minus the distributions the
built base actually installed, plus the workspace release wheels, the git
dependency built to a local wheel, and the pinned frontend bundle wheel from
scripts/build_release.py and scripts/release_sources.json. Each wheel record
carries its exact distribution name, version and sha256, from which the
installer generates hash-locked requirements. Installing a release never
touches GitHub or PyPI: all wheels are resolved before install.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import importlib
import ipaddress
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

from packaging.markers import Marker
from packaging.requirements import Requirement

from scripts.build_release import TAG_PATTERN, release_version, wheel_metadata

MANIFEST_FORMAT = "dinkster.engine/1"
CHANNEL_FORMAT = "dinkster.engine-channel/1"
CELLS_FORMAT = "dinkster.engine-cells/1"
BASE_IDENTITY_SCHEMA = "dinkster.engine-base-id/1"
CONTROL_RUNTIME_FORMAT = "dinkster.control-runtime/1"
MINIMUM_LAUNCHER_VERSION = "0.0.1"

CONTROL_ENVIRONMENT = "control"
EXECUTION_ENVIRONMENT = "execution"
ENVIRONMENTS = (CONTROL_ENVIRONMENT, EXECUTION_ENVIRONMENT)

BASE_DIR = "base"
STORE_DIR = "store"
ENGINE_DIR = "engine"
CHANNELS_DIR = "channels"
RECORDS_PATH = f"{BASE_DIR}/records.json"
CONTROL_DIR = "control"

CONTROL_RUNTIME_ROOTS = frozenset(
    {
        "dinkster",
        "dinkster-assets",
        "dinkster-caches",
        "dinkster-memory",
        "dinkster-protocol",
        "dinkster-registry",
        "dinkster-schema",
        "dinkster-values",
        "dinkster-workers",
    }
)
CONTROL_RUNTIME_FORBIDDEN = frozenset(
    {
        "dinkster-frontend",
        "dinkster-inference-torch",
        "torch",
        "torchvision",
    }
)

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_GZIP_EMPTY_MTIME = 0
_TAR_EPOCH = 0


class FeedError(RuntimeError):
    """A feed cannot be built: the caller sees the message, nothing partial."""


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FeedError(
            f"command failed ({result.returncode}): {command[0]} ...: {result.stderr.strip()}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Cell configuration


@dataclass(frozen=True)
class TorchRecipe:
    torch_requirement: str
    torchvision_requirement: str
    index_url: str


@dataclass(frozen=True)
class CellConfig:
    name: str
    recipe: TorchRecipe
    os_name: str
    arch: str
    python_implementation: str
    python_version: str


def load_cell_config(cells_path: Path, cell_name: str) -> CellConfig:
    document = json.loads(cells_path.read_text(encoding="utf-8"))
    if document.get("format") != CELLS_FORMAT:
        raise FeedError(f"{cells_path} is not format {CELLS_FORMAT}")
    cell = document.get("cells", {}).get(cell_name)
    if cell is None:
        known = ", ".join(sorted(document.get("cells", {})))
        raise FeedError(f"unknown cell {cell_name!r}; known cells: {known}")
    recipe = document["recipes"][cell["recipe"]]
    python = document["python"]
    return CellConfig(
        name=cell_name,
        recipe=TorchRecipe(
            torch_requirement=recipe["torch_requirement"],
            torchvision_requirement=recipe["torchvision_requirement"],
            index_url=recipe["index_url"],
        ),
        os_name=cell["os"],
        arch=cell["arch"],
        python_implementation=python["implementation"],
        python_version=python["version"],
    )


def require_native_cell(config: CellConfig) -> None:
    running = (
        "windows"
        if sys.platform.startswith("win")
        else "macos"
        if sys.platform == "darwin"
        else "linux"
    )
    if config.os_name != running:
        raise FeedError(
            f"cell {config.name} targets {config.os_name}; base archives build "
            f"natively only, and this host is {running}"
        )


# ---------------------------------------------------------------------------
# Base identity


def base_identity_hash(
    cell_name: str,
    os_name: str,
    arch: str,
    python_implementation: str,
    python_version: str,
    python_build: str,
    packages: dict[str, str],
    uv_sha256: str,
) -> str:
    """The base id: cell, platform, interpreter and the resolved closure only.

    The interpreter identity includes the python-build-standalone build
    revision and the bundled uv digest: standalone distributions can be
    revised byte-wise for one CPython version, and the accepted base is
    immutable content, so no byte under one id may ever vary. Kitchen, aimdo
    and every other code-layer pin are absent by construction, so changing
    them cannot change the base id.
    """
    identity = {
        "schema": BASE_IDENTITY_SCHEMA,
        "cell": cell_name,
        "platform": {"os": os_name, "arch": arch},
        "python": {
            "implementation": python_implementation,
            "version": python_version,
            "build": python_build,
        },
        "packages": dict(sorted(packages.items())),
        "uv_sha256": uv_sha256,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _pin_identity(
    config: CellConfig,
    python_build: str,
    uv_sha256: str,
) -> dict[str, object]:
    """What a base build starts from, used to skip unchanged rebuilds.

    Distinct from the base id: the id hashes the closure that resolution
    actually installed, while this records the requested pins so a rerun can
    recognize that the same pins already produced a verified archive. The
    standalone build revision and the bundled uv digest are pins too: either
    changing must produce a new archive, never a reuse.
    """
    return {
        "cell": config.name,
        "platform": {"os": config.os_name, "arch": config.arch},
        "python": {
            "implementation": config.python_implementation,
            "version": config.python_version,
            "build": python_build,
        },
        "torch_requirement": config.recipe.torch_requirement,
        "torchvision_requirement": config.recipe.torchvision_requirement,
        "index_url": config.recipe.index_url,
        "uv_sha256": uv_sha256,
    }


def _pin_identity_hash(config: CellConfig, python_build: str, uv_sha256: str) -> str:
    return hashlib.sha256(
        _canonical_json(_pin_identity(config, python_build, uv_sha256)).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Base archive construction


def _managed_install_dir(python: Path) -> Path:
    """The uv-managed interpreter install behind a found python, validated.

    A plain ``uv python find`` returns the active project venv when its
    version matches, and the standalone layout differs per platform: POSIX
    installs keep the interpreter under ``bin/``, Windows installs put
    ``python.exe`` at the install root. Both the managed shape and the
    executable the base staging expects are checked.
    """
    if python.name == "python.exe":
        install_dir = python.parent
        executable = install_dir / "python.exe"
    else:
        install_dir = python.parents[1]
        executable = install_dir / "bin" / "python3"
    if not install_dir.name.startswith("cpython-"):
        raise FeedError(f"uv python selection is not a managed interpreter install: {install_dir}")
    if install_dir == Path(sys.prefix) or install_dir.is_relative_to(Path(sys.prefix)):
        raise FeedError(f"uv python selection resolved to the active environment: {install_dir}")
    if not executable.is_file():
        raise FeedError(f"managed interpreter install has no executable at {executable}")
    return install_dir


def _uv_source(uv: str) -> Path:
    source = shutil.which(uv)
    if source is None:
        raise FeedError(f"uv executable {uv!r} not found on PATH")
    return Path(source)


def _standalone_build_id(install_dir: Path) -> str:
    """The python-build-standalone build revision of a managed install.

    Standalone distributions can be revised byte-wise for the same CPython
    version, so the BUILD marker at the install root is part of the
    interpreter's identity.
    """
    marker = install_dir / "BUILD"
    if not marker.is_file():
        raise FeedError(f"managed interpreter install has no BUILD marker: {install_dir}")
    build_id = marker.read_text(encoding="utf-8").strip()
    if not build_id:
        raise FeedError(f"managed interpreter install has an empty BUILD marker: {install_dir}")
    return build_id


def _uv_python_install_dir(uv: str, python_version: str) -> Path:
    _run([uv, "python", "install", python_version])
    python = Path(
        _run(
            [
                uv,
                "python",
                "find",
                "--managed-python",
                "--no-project",
                "--no-config",
                "--resolve-links",
                python_version,
            ]
        ).strip()
    )
    return _managed_install_dir(python)


def _staging_python(staging_root: Path, os_name: str) -> Path:
    if os_name == "windows":
        return staging_root / "python.exe"
    return staging_root / "bin" / "python3"


def _installed_packages(staging_python: Path) -> dict[str, str]:
    script = (
        "import json, importlib.metadata as m\n"
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))\n"
    )
    # -I isolates from user site and environment overrides while still
    # reading the interpreter's own site-packages; -S would see none of them.
    output = _run([str(staging_python), "-I", "-c", script])
    return {normalize_name(name): version for name, version in json.loads(output).items()}


def _normalize_installed_environment(staging_root: Path) -> None:
    """Remove install-location metadata and regenerate deterministic RECORD files."""
    for cache in staging_root.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    site_packages_roots = [
        *staging_root.glob("lib/python*/site-packages"),
        staging_root / "Lib" / "site-packages",
    ]
    for site_packages in site_packages_roots:
        if not site_packages.is_dir():
            continue
        for dist_info in site_packages.glob("*.dist-info"):
            for name in ("direct_url.json", "uv_cache.json"):
                (dist_info / name).unlink(missing_ok=True)

    runtime_root = staging_root.resolve()
    script_roots = {(staging_root / "bin").resolve(), (staging_root / "Scripts").resolve()}
    for site_packages in site_packages_roots:
        if not site_packages.is_dir():
            continue
        for record in site_packages.glob("*.dist-info/RECORD"):
            with record.open(newline="", encoding="utf-8") as handle:
                for source in csv.reader(handle):
                    if not source:
                        continue
                    installed = (site_packages / Path(*source[0].split("/"))).resolve()
                    if not installed.is_relative_to(runtime_root):
                        raise FeedError(f"installed RECORD escapes runtime root: {source[0]}")
                    if installed.parent in script_roots and installed.is_file():
                        installed.unlink()

    staging_paths = {
        str(staging_root).encode(),
        staging_root.as_posix().encode(),
    }
    for scripts in (staging_root / "bin", staging_root / "Scripts"):
        if not scripts.is_dir():
            continue
        for path in scripts.iterdir():
            if path.is_file() and not path.is_symlink():
                content = path.read_bytes()
                if any(prefix in content for prefix in staging_paths):
                    path.unlink()

    for site_packages in site_packages_roots:
        if not site_packages.is_dir():
            continue
        for record in site_packages.glob("*.dist-info/RECORD"):
            rows: list[list[str]] = []
            with record.open(newline="", encoding="utf-8") as handle:
                source_rows = list(csv.reader(handle))
            for source in source_rows:
                if not source:
                    continue
                installed = (site_packages / Path(*source[0].split("/"))).resolve()
                if not installed.is_relative_to(runtime_root):
                    raise FeedError(f"installed RECORD escapes runtime root: {source[0]}")
                if installed == record.resolve():
                    rows.append([source[0], "", ""])
                elif installed.is_file():
                    content = installed.read_bytes()
                    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
                    rows.append([source[0], f"sha256={digest.decode()}", str(len(content))])
            with record.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)


def _copy_uv_binary(uv: str, staging_root: Path, os_name: str) -> tuple[str, str]:
    """Bundle uv at the installer's fixed location; returns path and digest."""
    uv_rel = "tools/uv.exe" if os_name == "windows" else "tools/uv"
    source = _uv_source(uv)
    digest = sha256_file(source)
    destination = staging_root / uv_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if os_name != "windows":
        destination.chmod(0o755)
    return uv_rel, digest


def _validated_link_target(member_rel: str, target: str) -> None:
    """Symlinks in the archive must stay inside the extracted base root.

    Absolute targets and escapes would break relocatability or point outside
    the archive, so the build refuses them instead of shipping them.
    """
    if not target or target.startswith("/") or re.match(r"^[A-Za-z]:", target):
        raise FeedError(f"base archive would contain absolute symlink {member_rel} -> {target}")
    if "\\" in target or any(ord(char) < 32 or ord(char) == 127 for char in target):
        raise FeedError(f"base archive symlink target is not portable: {target!r}")
    parent = posixpath.dirname(member_rel)
    resolved = posixpath.normpath(posixpath.join(parent, target))
    if resolved == ".." or resolved.startswith("../"):
        raise FeedError(f"base archive symlink {member_rel} -> {target} escapes the base root")


def _tar_entry_info(path: Path, staging_root: Path) -> tarfile.TarInfo:
    relative = path.relative_to(staging_root).as_posix()
    if "\\" in relative or any(ord(char) < 32 or ord(char) == 127 for char in relative):
        raise FeedError(f"base archive path is not portable: {relative!r}")
    link_target = None
    if path.is_symlink():
        link_target = os.readlink(path)
        _validated_link_target(relative, link_target)
        stat_result = path.lstat()
        mode = 0o777
    elif path.is_dir():
        stat_result = path.lstat()
        mode = 0o755
    else:
        stat_result = path.lstat()
        mode = 0o755 if stat_result.st_mode & stat.S_IXUSR else 0o644
    info = tarfile.TarInfo(relative)
    info.size = 0 if link_target is not None else stat_result.st_size
    info.mode = mode
    info.mtime = _TAR_EPOCH
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if link_target is not None:
        info.type = tarfile.SYMTYPE
        info.linkname = link_target
    elif path.is_dir():
        info.type = tarfile.DIRTYPE
    else:
        info.type = tarfile.REGTYPE
    return info


def create_base_archive(staging_root: Path, destination: Path) -> None:
    """Pack the staging interpreter tree as a relocatable tar.gz base archive.

    Internal symlinks are preserved so the interpreter still finds its
    libraries after extraction anywhere; external links are rejected.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (staging_root / relative for relative in [Path()] + list(_walk_relative(staging_root))),
        key=lambda path: path.relative_to(staging_root).as_posix(),
    )
    with destination.open("wb") as raw:
        gz = gzip.GzipFile(fileobj=raw, mode="wb", mtime=_GZIP_EMPTY_MTIME, filename="")
        with gz as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as tar:
                for path in paths:
                    info = _tar_entry_info(path, staging_root)
                    if info.type == tarfile.REGTYPE:
                        tar.addfile(info, path.open("rb"))
                    else:
                        tar.addfile(info)


def _walk_relative(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in dirnames + filenames:
            path = Path(directory) / name
            found.append(path.relative_to(root))
    return found


@dataclass(frozen=True)
class BaseBuild:
    base_id: str
    archive_path: str
    sha256: str
    size: int
    packages: dict[str, str]
    python_path: str
    reused: bool


def _load_base_records(feed_dir: Path) -> dict[str, Any]:
    records_path = feed_dir / RECORDS_PATH
    if not records_path.is_file():
        return {}
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, dict):
        raise FeedError(f"{records_path} is not a base record mapping")
    return records


def _save_base_records(feed_dir: Path, records: dict[str, Any]) -> None:
    records_path = feed_dir / RECORDS_PATH
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reuse_base_record(
    feed_dir: Path, config: CellConfig, record: dict[str, Any]
) -> BaseBuild | None:
    """An existing record at the matching pin identity, if it still verifies.

    The archive must exist and hash to the recorded digest; anything else
    falls through to a rebuild rather than shipping unverified bytes.
    """
    archive_rel = record.get("archive")
    if not isinstance(archive_rel, str):
        return None
    archive = feed_dir / archive_rel
    if not archive.is_file():
        return None
    if sha256_file(archive) != record.get("sha256"):
        return None
    packages = record.get("packages")
    if not isinstance(packages, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in packages.items()
    ):
        return None
    python_build = record.get("python_build")
    uv_sha256 = record.get("uv_sha256")
    if not isinstance(python_build, str) or not isinstance(uv_sha256, str):
        return None
    expected_id = base_identity_hash(
        config.name,
        config.os_name,
        config.arch,
        config.python_implementation,
        config.python_version,
        python_build,
        packages,
        uv_sha256,
    )
    if record.get("id") != expected_id:
        return None
    return BaseBuild(
        base_id=expected_id,
        archive_path=archive_rel,
        sha256=record["sha256"],
        size=record["size"],
        packages=packages,
        python_path=record["python_path"],
        reused=True,
    )


def build_base(uv: str, config: CellConfig, feed_dir: Path) -> BaseBuild:
    """Materialize, hash and archive the cell's base, reusing a verified one."""
    require_native_cell(config)
    # Interpreter selection and the bundled uv digest are part of the pin
    # identity, so a revised standalone build or updated host uv forces a
    # fresh archive instead of a reuse.
    uv_sha256 = sha256_file(_uv_source(uv))
    interpreter_dir = _uv_python_install_dir(uv, config.python_version)
    python_build = _standalone_build_id(interpreter_dir)
    pin_hash = _pin_identity_hash(config, python_build, uv_sha256)
    records = _load_base_records(feed_dir)
    record = records.get(pin_hash)
    if record is not None:
        reused = _reuse_base_record(feed_dir, config, record)
        if reused is not None:
            print(
                f"base reuse: cell {config.name} id {reused.base_id} already verified, "
                "skipping build"
            )
            return reused
        print(f"base record {pin_hash} did not verify; rebuilding")

    with tempfile.TemporaryDirectory(prefix="dinkster-base-") as work:
        staging_root = Path(work) / "base-root"
        shutil.copytree(interpreter_dir, staging_root, symlinks=True)
        staging_python = _staging_python(staging_root, config.os_name)
        if not staging_python.is_file():
            raise FeedError(
                f"interpreter install has no python at {staging_python} for cell {config.name}"
            )
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(staging_python),
                # The base is the standalone interpreter itself: installing
                # its pinned torch closure into that root is the intent, so
                # the externally-managed marker does not apply here.
                "--no-config",
                "--break-system-packages",
                "--index-url",
                config.recipe.index_url,
                config.recipe.torch_requirement,
                config.recipe.torchvision_requirement,
            ]
        )
        packages = _installed_packages(staging_python)
        for required in ("torch", "torchvision"):
            if required not in packages:
                raise FeedError(
                    f"base for cell {config.name} resolved without {required}: {sorted(packages)}"
                )
        _normalize_installed_environment(staging_root)
        _, bundled_uv_sha256 = _copy_uv_binary(uv, staging_root, config.os_name)
        python_rel = staging_python.relative_to(staging_root).as_posix()
        base_id = base_identity_hash(
            config.name,
            config.os_name,
            config.arch,
            config.python_implementation,
            config.python_version,
            python_build,
            packages,
            bundled_uv_sha256,
        )
        archive_rel = f"{BASE_DIR}/{config.name}/{base_id}.tar.gz"
        archive = feed_dir / archive_rel
        if archive.is_file():
            archive.unlink()
        create_base_archive(staging_root, archive)

    archive_sha = sha256_file(archive)
    build = BaseBuild(
        base_id=base_id,
        archive_path=archive_rel,
        sha256=archive_sha,
        size=archive.stat().st_size,
        packages=packages,
        python_path=python_rel,
        reused=False,
    )
    records[pin_hash] = {
        "id": build.base_id,
        "archive": build.archive_path,
        "sha256": build.sha256,
        "size": build.size,
        "packages": build.packages,
        "python_path": build.python_path,
        "python_build": python_build,
        "uv_sha256": uv_sha256,
        "pin": _pin_identity(config, python_build, uv_sha256),
    }
    _save_base_records(feed_dir, records)
    print(
        f"base built: cell {config.name} id {base_id} "
        f"({build.size} bytes, {len(packages)} distributions)"
    )
    return build


# ---------------------------------------------------------------------------
# Code layer


def load_lock(root: Path) -> dict[str, dict[str, Any]]:
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in lock["package"]}
    if len(entries) != len(lock["package"]):
        raise FeedError("uv.lock contains duplicate package names")
    return entries


def _is_local_source(source: dict[str, Any]) -> bool:
    return any(key in source for key in ("editable", "directory", "virtual"))


def workspace_members(lock: dict[str, dict[str, Any]]) -> set[str]:
    return {name for name, entry in lock.items() if _is_local_source(entry["source"])}


def cell_marker_environment(config: CellConfig) -> dict[str, str]:
    """The PEP 508 marker environment of the cell's native platform.

    Every variable is fixed by the cell definition, never by the host
    interpreter, so lock traversal is deterministic for each cell.
    """
    platforms = {
        "linux": ("linux", "Linux", "posix"),
        "windows": ("win32", "Windows", "nt"),
        "macos": ("darwin", "Darwin", "posix"),
    }
    machines = {"x86_64": "x86_64", "amd64": "AMD64", "arm64": "arm64"}
    platform = platforms.get(config.os_name)
    machine = machines.get(config.arch)
    if platform is None or machine is None or config.python_implementation != "cpython":
        raise FeedError(
            f"cell {config.name} has no marker environment for "
            f"{config.os_name}/{config.arch}/{config.python_implementation}"
        )
    return {
        "implementation_name": "cpython",
        "implementation_version": config.python_version,
        "os_name": platform[2],
        "platform_machine": machine,
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": platform[1],
        "platform_version": "",
        "python_full_version": config.python_version,
        "python_version": ".".join(config.python_version.split(".")[:2]),
        "sys_platform": platform[0],
    }


def _dependency_allowed(dependency: dict[str, Any], environment: dict[str, str]) -> bool:
    marker = dependency.get("marker")
    if marker is None:
        return True
    return bool(Marker(str(marker)).evaluate(environment))


def locked_closure(
    lock: dict[str, dict[str, Any]],
    roots: set[str],
    environment: dict[str, str],
    skip_expansion: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Every distribution reachable from the roots through runtime dependencies.

    Only the ``dependencies`` edge list is followed, so dev dependencies and
    the lock's dev-dependency sections never enter the closure. Edges whose
    marker is false in the cell's environment are not followed. Distributions
    in ``skip_expansion`` (the base's packages) are recorded when reached but
    not expanded: the lock's subtree for such a distribution describes the
    lock's variant of it, not the variant the base actually installed.
    """
    closure: dict[str, dict[str, Any]] = {}
    stack = sorted(roots)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        entry = lock.get(name)
        if entry is None:
            raise FeedError(f"uv.lock has no entry for {name}")
        closure[name] = entry
        if name in skip_expansion:
            continue
        stack.extend(
            dependency["name"]
            for dependency in entry.get("dependencies", [])
            if _dependency_allowed(dependency, environment)
        )
    return closure


def code_lock_closure(
    lock: dict[str, dict[str, Any]], base_packages: dict[str, str], config: CellConfig
) -> dict[str, dict[str, Any]]:
    """The distributions the code layer must supply, with their lock entries.

    The built base's package mapping is the authoritative exclusion set. A
    distribution the base actually installed never ships again, and its lock
    subtree is not expanded, so the code layer does not inherit the lock's
    variant of a base distribution's dependencies. Every other distribution
    reachable from the workspace members through marker-true edges stays in,
    including one shared with a base distribution while absent from the
    built base itself.
    """
    closure = locked_closure(
        lock,
        workspace_members(lock),
        cell_marker_environment(config),
        skip_expansion=frozenset(base_packages),
    )
    return {name: entry for name, entry in closure.items() if name not in base_packages}


def control_runtime_lock_closure(
    lock: dict[str, dict[str, Any]], config: CellConfig
) -> dict[str, dict[str, Any]]:
    """The intentionally small closure used by the packaged bootstrap.

    The umbrella wheel is installed without dependencies because its metadata
    describes the complete execution product. The other roots are the control
    modules imported by engine installation and generation management; their
    ordinary runtime dependencies are followed from the lock.
    """
    closure = locked_closure(
        lock,
        set(CONTROL_RUNTIME_ROOTS - {"dinkster"}),
        cell_marker_environment(config),
    )
    closure["dinkster"] = lock["dinkster"]
    forbidden = sorted(CONTROL_RUNTIME_FORBIDDEN & closure.keys())
    forbidden.extend(sorted(name for name in closure if name.startswith("dinkster-model-")))
    if forbidden:
        raise FeedError(f"control runtime includes forbidden distributions: {forbidden}")
    return closure


def wheel_filename_tags(filename: str) -> tuple[str, str, frozenset[str]]:
    if not filename.endswith(".whl"):
        raise FeedError(f"not a wheel filename: {filename}")
    parts = filename[: -len(".whl")].split("-")
    if len(parts) != 5:
        raise FeedError(f"not a wheel filename: {filename}")
    return parts[2], parts[3], frozenset(parts[4].split("."))


def _interpreter_abi_ok(interpreter: str, abi: str) -> bool:
    if interpreter.startswith("py3"):
        return abi == "none"
    if re.fullmatch(r"cp3([0-9]|1[0-1])", interpreter):
        return abi == "abi3"
    if interpreter == "cp312":
        return abi in ("cp312", "abi3")
    return False


def _native_platform_ok(platform: str, os_name: str, arch: str) -> bool:
    if os_name == "linux":
        return platform == f"linux_{arch}" or (
            platform.startswith("manylinux") and platform.endswith(f"_{arch}")
        )
    if os_name == "windows":
        return platform == f"win_{arch}"
    if os_name == "macos":
        return arch == "arm64" and re.fullmatch(r"macosx_\d+(?:_\d+)?_arm64", platform) is not None
    return False


def wheel_matches_cell(filename: str, os_name: str, arch: str) -> bool:
    """Whether the wheel can install on the cell's interpreter and platform."""
    interpreter, abi, platforms = wheel_filename_tags(filename)
    if not _interpreter_abi_ok(interpreter, abi):
        return False
    if "any" in platforms:
        return abi == "none"
    return any(_native_platform_ok(platform, os_name, arch) for platform in platforms)


_LEGACY_MANYLINUX_FLOORS = {"1": (2, 5), "2010": (2, 12), "2014": (2, 17)}


def _platform_preference(platform: str) -> tuple[int, int, int]:
    """Smaller is broader: the tag installs on more machines.

    Tags are matched whole, architecture included: versioned glibc and macOS
    tags rank by their floor version ascending, legacy ``manylinuxN`` aliases
    resolve to the same floor as their modern spelling (manylinux2014 is
    manylinux_2_17), and a plain ``linux_`` tag is a last resort.
    """
    modern = re.fullmatch(r"manylinux_(\d+)_(\d+)_.+", platform)
    if modern:
        return (1, int(modern.group(1)), int(modern.group(2)))
    legacy = re.fullmatch(r"manylinux(1|2010|2014)_.+", platform)
    if legacy:
        major, minor = _LEGACY_MANYLINUX_FLOORS[legacy.group(1)]
        return (1, major, minor)
    macos = re.fullmatch(r"macosx_(\d+)_(\d+)_.+", platform)
    if macos:
        return (1, int(macos.group(1)), int(macos.group(2)))
    if platform.startswith("linux_"):
        return (2, 0, 0)
    return (3, 0, 0)


def _wheel_semantic_rank(filename: str) -> tuple[int, int, tuple[int, int, int]]:
    """Interpreter ABI closeness, then broadest native platform, no filename.

    For the fixed CPython 3.12 interpreter an exact cp312 wheel outranks
    abi3, and abi3 wheels rank by the highest CPython baseline they require
    (cp311 before cp39): the closer ABI has the more specific code paths.
    """
    interpreter, abi, platforms = wheel_filename_tags(filename)
    if abi == "cp312":
        abi_class, baseline = 0, 0
    elif abi == "abi3":
        abi_class, baseline = 1, -int(interpreter[len("cp3") :])
    else:
        abi_class, baseline = 2, 0
    return (abi_class, baseline, min(_platform_preference(platform) for platform in platforms))


def _wheel_preference(filename: str) -> tuple[int, int, tuple[int, int, int], str]:
    """The semantic rank, with the filename only for deterministic order."""
    return (*_wheel_semantic_rank(filename), filename)


def select_cell_wheel(
    lock_name: str, wheels: list[dict[str, Any]], os_name: str, arch: str
) -> dict[str, Any]:
    """The locked wheel the cell installs, or a refusal naming the distribution.

    A distribution can ship several wheels valid for one cell (abi3 across
    interpreter baselines, layered manylinux policies); the exact
    interpreter ABI and then the broadest native platform tag win. Equal
    semantic ranks stay ambiguous rather than picking by filename.
    """
    filename_of = lambda wheel: wheel["url"].rsplit("/", 1)[-1]  # noqa: E731
    candidates = [
        wheel for wheel in wheels if wheel_matches_cell(filename_of(wheel), os_name, arch)
    ]
    if not candidates:
        raise FeedError(
            f"distribution {lock_name} has no wheel for cell platform {os_name}/{arch} in uv.lock"
        )
    ranked = sorted(candidates, key=lambda wheel: _wheel_preference(filename_of(wheel)))
    best_key = _wheel_semantic_rank(filename_of(ranked[0]))
    if sum(1 for wheel in candidates if _wheel_semantic_rank(filename_of(wheel)) == best_key) > 1:
        raise FeedError(
            f"distribution {lock_name} has {len(candidates)} candidate wheels for "
            f"{os_name}/{arch}; the lock is ambiguous"
        )
    return ranked[0]


def _build_workspace_wheels(uv: str, root: Path, output: Path) -> dict[str, Path]:
    _run([uv, "build", "--all-packages", "--wheel", "--out-dir", str(output)], cwd=root)
    built: dict[str, Path] = {}
    for wheel in sorted(output.glob("*.whl")):
        name, _ = wheel_metadata(wheel)
        previous = built.setdefault(name, wheel)
        if previous is not wheel:
            raise FeedError(f"workspace build produced two wheels for {name}")
    return built


def _build_git_dependency_wheel(uv: str, source_url: str, output: Path) -> Path:
    """Build a wheel from the git dependency pinned in uv.lock.

    The wheel is what gets installed; a release needs no source checkout, so
    the dependency must become a local wheel before the feed is complete. The
    built wheel lands in the caller-owned ``output`` directory so the returned
    path outlives the temporary clone.
    """
    clone_url = source_url.split("?", 1)[0].split("#", 1)[0]
    commit = source_url.rsplit("#", 1)[-1]
    if not COMMIT_PATTERN.fullmatch(commit):
        raise FeedError(f"git dependency source has no pinned commit: {source_url}")
    with tempfile.TemporaryDirectory(prefix="dinkster-git-dep-") as work:
        clone = Path(work) / "src"
        _run(["git", "clone", "--quiet", clone_url, str(clone)])
        _run(["git", "-C", str(clone), "checkout", "--quiet", commit])
        _run([uv, "build", "--wheel", "--out-dir", str(output), str(clone)])
    wheels = sorted(output.glob("*.whl"))
    if len(wheels) != 1:
        raise FeedError(f"git dependency {clone_url} build produced {len(wheels)} wheels")
    return wheels[0]


def _download_locked_wheel(wheel: dict[str, Any], destination: Path) -> str:
    url = wheel["url"]
    expected = wheel["hash"]
    scheme, _, expected_sha = expected.partition(":")
    if scheme != "sha256" or len(expected_sha) != 64:
        raise FeedError(f"locked wheel hash is not a sha256 digest: {expected}")
    # A long build crosses many single-wheel fetches; a transient connection
    # reset should cost a retry, not the build. The download lands at a
    # temporary name first, so a failed attempt leaves no partial wheel.
    for attempt in range(3):
        partial = destination.with_name(destination.name + f".partial{attempt}")
        try:
            with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as target:
                digest = hashlib.sha256()
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    target.write(block)
            partial.replace(destination)
            actual = digest.hexdigest()
            if actual != expected_sha:
                destination.unlink(missing_ok=True)
                raise FeedError(f"downloaded {url} hashes to {actual}, uv.lock says {expected_sha}")
            return actual
        except (urllib.error.URLError, OSError) as error:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise FeedError(f"wheel download failed after retries, {url}: {error}") from None
            time.sleep(5 * (attempt + 1))
    raise AssertionError("download retry loop exhausted without returning")


@dataclass(frozen=True)
class WheelEntry:
    path: str
    sha256: str
    size: int
    filename: str
    name: str
    version: str
    environments: tuple[str, ...]


def _place_wheel(source: Path, feed_dir: Path) -> WheelEntry:
    digest = sha256_file(source)
    store_path = feed_dir / STORE_DIR / digest
    if store_path.exists():
        existing = sha256_file(store_path)
        if existing != digest:
            raise FeedError(
                f"content store entry {STORE_DIR}/{digest} hashes to {existing}: "
                "tampered or corrupted store"
            )
    else:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, store_path)
    name, version = wheel_metadata(store_path)
    return WheelEntry(
        path=f"{STORE_DIR}/{digest}",
        sha256=digest,
        size=store_path.stat().st_size,
        filename=source.name,
        # Wheel metadata names may use underscores; the manifest contract
        # and the installer's requirements both speak normalized names.
        name=normalize_name(name),
        version=version,
        environments=ENVIRONMENTS,
    )


def _register_entry(entries: dict[str, WheelEntry], entry: WheelEntry) -> None:
    previous = entries.get(entry.name)
    if previous is not None:
        raise FeedError(f"duplicate distribution {entry.name} in the code layer")
    entries[entry.name] = entry


def _refuse_base_code_overlap(base_packages: dict[str, str], entries: list[WheelEntry]) -> None:
    # The base already ships its closure; a second copy in the code layer
    # would make the offline install ambiguous.
    duplicated = sorted(base_packages.keys() & {entry.name for entry in entries})
    if duplicated:
        raise FeedError(f"distributions present in both base and code layers: {duplicated}")


def _wheel_requires_dist(wheel_file: Path) -> list[str]:
    with zipfile.ZipFile(wheel_file) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise FeedError(f"wheel has no unique METADATA file: {wheel_file.name}")
        message = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
    return list(message.get_all("Requires-Dist") or [])


def _verify_code_layer_completeness(
    entries: list[WheelEntry],
    feed_dir: Path,
    base_packages: dict[str, str],
    environment: dict[str, str],
) -> None:
    """Every runtime requirement of every code wheel must be satisfied offline.

    Checked against the actual wheel METADATA, not the lock graph: a
    requirement whose marker is false for the cell does not apply, and the
    base satisfies whatever it actually installed. Everything else must ship
    in the code layer, which is what stops a lock-subtree exclusion from
    hiding a dependency real code wheels need.
    """
    supplied = frozenset(base_packages) | {entry.name for entry in entries}
    missing: dict[str, list[str]] = {}
    for entry in entries:
        for requirement_text in _wheel_requires_dist(feed_dir / entry.path):
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate(
                {**environment, "extra": ""}
            ):
                continue
            required_name = normalize_name(requirement.name)
            if required_name not in supplied:
                missing.setdefault(entry.name, []).append(requirement_text)
    if missing:
        details = "; ".join(f"{name}: {reqs}" for name, reqs in sorted(missing.items()))
        raise FeedError(f"code layer is missing required distributions: {details}")


def build_code_layer(
    uv: str,
    root: Path,
    feed_dir: Path,
    config: CellConfig,
    base_packages: dict[str, str],
    frontend_wheel: Path,
) -> list[WheelEntry]:
    """Resolve and stage every wheel the cell's code layer needs.

    Everything comes from uv.lock's runtime graph for the cell's platform,
    except the distributions the base archive already provides and the
    frontend bundle, which arrives prebuilt. No wheel is fetched at install
    time.
    """
    if not frontend_wheel.is_file():
        raise FeedError(
            f"the code layer requires the pinned frontend bundle wheel, missing: {frontend_wheel}"
        )
    lock = load_lock(root)
    code_closure = code_lock_closure(lock, base_packages, config)
    code_names = sorted(code_closure)

    with tempfile.TemporaryDirectory(prefix="dinkster-wheels-") as work:
        built = _build_workspace_wheels(uv, root, Path(work))
        entries: dict[str, WheelEntry] = {}
        for lock_name in code_names:
            entry = code_closure[lock_name]
            source = entry["source"]
            if _is_local_source(source):
                wheel = built.get(normalize_name(lock_name))
                if wheel is None:
                    raise FeedError(f"workspace build produced no wheel for {lock_name}")
                wheel_entry = _place_wheel(wheel, feed_dir)
            elif "git" in source:
                with tempfile.TemporaryDirectory(prefix="dinkster-git-wheel-") as git_work:
                    git_wheel = _build_git_dependency_wheel(uv, source["git"], Path(git_work))
                    wheel_entry = _place_wheel(git_wheel, feed_dir)
            else:
                locked = select_cell_wheel(
                    lock_name, entry.get("wheels", []), config.os_name, config.arch
                )
                with tempfile.TemporaryDirectory(prefix="dinkster-download-") as download_work:
                    downloaded = Path(download_work) / locked["url"].rsplit("/", 1)[-1]
                    _download_locked_wheel(locked, downloaded)
                    wheel_entry = _place_wheel(downloaded, feed_dir)
            _register_entry(entries, wheel_entry)

    frontend_entry = _place_wheel(frontend_wheel, feed_dir)
    _register_entry(entries, frontend_entry)
    result = [entries[name] for name in sorted(entries)]
    _refuse_base_code_overlap(base_packages, result)
    _verify_code_layer_completeness(
        result, feed_dir, base_packages, cell_marker_environment(config)
    )
    return result


# ---------------------------------------------------------------------------
# Packaged control runtime


def control_runtime_descriptor(
    *,
    commit: str,
    platform: str,
    artifact_path: str,
    sha256: str,
    size: int,
    python: str,
) -> dict[str, object]:
    """Create the canonical descriptor consumed by native packaging."""
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise FeedError("control runtime commit must be a full Git commit")
    if re.fullmatch(r"[a-z0-9_]+-[a-z0-9_]+", platform) is None:
        raise FeedError("control runtime platform must be a normalized OS-architecture key")
    for label, value in (("artifact", artifact_path), ("python", python)):
        path = Path(value)
        if path.is_absolute() or not value or ".." in path.parts or path.as_posix() != value:
            raise FeedError(f"control runtime {label} path must be a normalized relative path")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None or size < 1:
        raise FeedError("control runtime artifact identity is invalid")
    if artifact_path != f"{CONTROL_DIR}/{platform}/{sha256}.tar.gz":
        raise FeedError("control runtime artifact path must be content-addressed by its SHA-256")
    return {
        "format": CONTROL_RUNTIME_FORMAT,
        "commit": commit,
        "platform": platform,
        "artifact": {"path": artifact_path, "sha256": sha256, "size": size},
        "python": python,
        "invocation": ["<python>", "-I", "-m", "dinkster.cli"],
    }


def build_control_runtime(
    uv: str,
    root: Path,
    feed_dir: Path,
    config: CellConfig,
    commit: str,
) -> dict[str, object]:
    """Build the immutable bootstrap used before an engine is installed."""
    require_native_cell(config)
    platform_name = f"{config.os_name}-{config.arch}"
    lock = load_lock(root)
    closure = control_runtime_lock_closure(lock, config)
    interpreter_dir = _uv_python_install_dir(uv, config.python_version)

    with tempfile.TemporaryDirectory(prefix="dinkster-control-") as work:
        work_dir = Path(work)
        staging_root = work_dir / "runtime"
        wheels_dir = work_dir / "wheels"
        wheels_dir.mkdir()
        shutil.copytree(interpreter_dir, staging_root, symlinks=True)
        python = _staging_python(staging_root, config.os_name)
        built = _build_workspace_wheels(uv, root, wheels_dir / "workspace")
        wheels: list[Path] = []
        for name, entry in sorted(closure.items()):
            source = entry["source"]
            if _is_local_source(source):
                wheel = built.get(normalize_name(name))
                if wheel is None:
                    raise FeedError(f"workspace build produced no wheel for {name}")
            elif "git" in source:
                output = wheels_dir / f"git-{name}"
                output.mkdir()
                wheel = _build_git_dependency_wheel(uv, source["git"], output)
            else:
                locked = select_cell_wheel(
                    name, entry.get("wheels", []), config.os_name, config.arch
                )
                output = wheels_dir / locked["url"].rsplit("/", 1)[-1]
                _download_locked_wheel(locked, output)
                wheel = output
            wheels.append(wheel)
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--no-config",
                "--break-system-packages",
                "--no-deps",
                *map(str, wheels),
            ]
        )
        installed = _installed_packages(python)
        missing = sorted(CONTROL_RUNTIME_ROOTS - installed.keys())
        forbidden = sorted(CONTROL_RUNTIME_FORBIDDEN & installed.keys())
        forbidden.extend(sorted(name for name in installed if name.startswith("dinkster-model-")))
        if missing or forbidden:
            raise FeedError(
                f"invalid control runtime packages; missing={missing}, forbidden={forbidden}"
            )
        probe_root = work_dir / "probe-root"
        _run(
            [
                str(python),
                "-I",
                "-m",
                "dinkster.cli",
                "generations",
                "--root",
                str(probe_root),
                "--json",
            ]
        )
        shutil.rmtree(probe_root)
        _normalize_installed_environment(staging_root)
        python_rel = python.relative_to(staging_root).as_posix()
        temporary_archive = work_dir / "control-runtime.tar.gz"
        create_base_archive(staging_root, temporary_archive)
        digest = sha256_file(temporary_archive)
        archive_rel = f"{CONTROL_DIR}/{platform_name}/{digest}.tar.gz"
        archive = feed_dir / archive_rel
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.is_file() and sha256_file(archive) != digest:
            raise FeedError(f"existing control runtime does not match its digest: {archive}")
        if not archive.is_file():
            shutil.copyfile(temporary_archive, archive)

    descriptor = control_runtime_descriptor(
        commit=commit,
        platform=platform_name,
        artifact_path=archive_rel,
        sha256=digest,
        size=archive.stat().st_size,
        python=python_rel,
    )
    descriptor_path = feed_dir / CONTROL_DIR / commit / f"{platform_name}.json"
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return descriptor


# ---------------------------------------------------------------------------
# Manifests


def build_manifest(
    commit: str, cell_name: str, base: BaseBuild, entries: list[WheelEntry]
) -> dict[str, Any]:
    """The manifest document exactly as the strict consumer parser defines it.

    The installer generates its hash-locked requirements from the wheel
    records themselves, so the manifest carries no separate requirements
    section, and the bundled uv is fixed at tools/uv inside the base root
    rather than named by a field.
    """
    return {
        "format": MANIFEST_FORMAT,
        "commit": commit,
        "cell": cell_name,
        "base": {
            "id": base.base_id,
            "archive": {
                "path": base.archive_path,
                "sha256": base.sha256,
                "size": base.size,
            },
            "python": base.python_path,
            "packages": dict(sorted(base.packages.items())),
        },
        "wheels": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size": entry.size,
                "filename": entry.filename,
                "name": entry.name,
                "version": entry.version,
                "environments": list(entry.environments),
            }
            for entry in sorted(entries, key=lambda entry: entry.path)
        ],
    }


# ---------------------------------------------------------------------------
# Channels


def channel_entry(feed_dir: Path, commit: str, cell_name: str) -> dict[str, object]:
    manifest_rel = f"{ENGINE_DIR}/{commit}/{cell_name}.json"
    manifest_path = feed_dir / manifest_rel
    if not manifest_path.is_file():
        raise FeedError(f"channel references a manifest that was never built: {manifest_rel}")
    return {
        "path": manifest_rel,
        "sha256": sha256_file(manifest_path),
        "size": manifest_path.stat().st_size,
    }


def upload_feed(feed_dir: Path, endpoint: str, bucket: str) -> int:
    """Upload the feed to a loopback-only S3-compatible endpoint.

    Development uses MinIO on the local machine; the loopback restriction is
    the guard that keeps this helper from ever publishing externally.
    """
    host = urllib.parse.urlparse(endpoint).hostname
    if host is None:
        raise FeedError(f"upload endpoint is not a URL: {endpoint}")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise FeedError(f"refusing non-loopback upload endpoint {endpoint}")
        except ValueError:
            raise FeedError(f"refusing non-loopback upload endpoint {endpoint}") from None
    try:
        boto3 = importlib.import_module("boto3")
    except ImportError as error:  # build-only dependency
        raise FeedError(f"upload requires boto3: {error}") from None
    client = boto3.client("s3", endpoint_url=endpoint)
    uploaded = 0
    for directory, _, filenames in os.walk(feed_dir):
        for filename in sorted(filenames):
            path = Path(directory) / filename
            key = path.relative_to(feed_dir).as_posix()
            client.upload_file(str(path), bucket, key)
            uploaded += 1
    return uploaded


def git_commit(root: Path) -> str:
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    if not COMMIT_PATTERN.fullmatch(commit):
        raise FeedError(f"git produced a non-commit value: {commit!r}")
    return commit


def require_clean_checkout(root: Path) -> None:
    if _run(["git", "-C", str(root), "status", "--porcelain"]):
        raise FeedError("control runtime must be built from a clean checkout")


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, required=True, help="feed output directory")
    parser.add_argument("--cells", default="", help="comma-separated cells to build")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--frontend-wheel", type=Path, help="prebuilt pinned frontend wheel")
    parser.add_argument("--channel", choices=("stable", "github-live"))
    parser.add_argument("--tag", help="stable channel release tag (vX.Y.Z)")
    parser.add_argument("--evidence-file", type=Path, help="github-live validation evidence JSON")
    parser.add_argument("--upload-endpoint", help="loopback-only S3 endpoint for upload")
    parser.add_argument("--upload-bucket", help="S3 bucket for upload")
    parser.add_argument(
        "--control-runtime",
        action="store_true",
        help="build the minimal packaged bootstrap instead of an engine feed cell",
    )
    args = parser.parse_args(argv)
    try:
        return _main(args)
    except FeedError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _main(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    feed_dir = args.feed.resolve()
    commit = git_commit(root)
    if getattr(args, "control_runtime", False):
        require_clean_checkout(root)

    if args.channel is not None:
        _write_channel(root, feed_dir, commit, args)
    else:
        if not args.cells:
            raise FeedError("nothing to do: pass --cells or --channel")
        frontend_wheel = None if args.control_runtime else _resolve_frontend_wheel(root, args)
        for cell_name in [cell.strip() for cell in args.cells.split(",") if cell.strip()]:
            config = load_cell_config(root / "scripts/engine_cells.json", cell_name)
            if args.control_runtime:
                descriptor = build_control_runtime(args.uv, root, feed_dir, config, commit)
                artifact = descriptor["artifact"]
                assert isinstance(artifact, dict)
                print(f"control runtime written: {artifact['path']}")
                continue
            base = build_base(args.uv, config, feed_dir)
            entries = build_code_layer(
                args.uv, root, feed_dir, config, base.packages, frontend_wheel
            )
            manifest = build_manifest(commit, cell_name, base, entries)
            manifest_rel = f"{ENGINE_DIR}/{commit}/{cell_name}.json"
            manifest_path = feed_dir / manifest_rel
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            code_bytes = sum(entry.size for entry in entries)
            print(
                f"manifest written: {manifest_rel} ({len(entries)} wheels, "
                f"{code_bytes} bytes of wheels)"
            )

    if args.upload_endpoint:
        if not args.upload_bucket:
            raise FeedError("--upload-bucket is required with --upload-endpoint")
        uploaded = upload_feed(feed_dir, args.upload_endpoint, args.upload_bucket)
        print(f"uploaded {uploaded} objects to {args.upload_bucket}")
    return 0


def _resolve_frontend_wheel(root: Path, args: argparse.Namespace) -> Path:
    if args.frontend_wheel is not None:
        wheel = args.frontend_wheel.resolve()
        if not wheel.is_file():
            raise FeedError(f"frontend wheel does not exist: {wheel}")
        return wheel
    raise FeedError(
        "the code layer requires the pinned frontend bundle wheel from "
        f"{(root / 'scripts/release_sources.json')}: pass --frontend-wheel"
    )


def _write_channel(root: Path, feed_dir: Path, commit: str, args: argparse.Namespace) -> int:
    if not args.cells:
        raise FeedError("--channel requires the cells to list, e.g. --cells linux-cu128")
    cells = [cell.strip() for cell in args.cells.split(",") if cell.strip()]
    if args.channel == "stable":
        if not args.tag:
            raise FeedError("the stable channel requires --tag")
        tag = args.tag
        if TAG_PATTERN.fullmatch(tag) is None:
            raise FeedError(f"stable channel tag must match vX.Y.Z, got {tag!r}")
        try:
            release_version(root, tag)
        except ValueError as error:
            raise FeedError(str(error)) from None
    else:
        if args.tag:
            raise FeedError("github-live takes --evidence-file, not --tag")
        if args.evidence_file is None:
            raise FeedError(
                "github-live requires explicit successful-validation evidence via --evidence-file"
            )
        evidence = json.loads(args.evidence_file.read_text(encoding="utf-8"))
        if evidence.get("commit") != commit:
            raise FeedError(
                "validation evidence names a different commit: "
                f"{evidence.get('commit')!r} vs feed commit {commit}"
            )
        validation = evidence.get("validation")
        if not isinstance(validation, str) or not validation.strip():
            raise FeedError("validation evidence must name the successful validation run")
    channel_rel = f"{CHANNELS_DIR}/{args.channel}.json"
    document = {
        "format": CHANNEL_FORMAT,
        "channel": args.channel,
        "commit": commit,
        "minimumLauncherVersion": MINIMUM_LAUNCHER_VERSION,
        "cells": {cell: channel_entry(feed_dir, commit, cell) for cell in sorted(cells)},
    }
    channel_path = feed_dir / channel_rel
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    channel_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"channel written: {channel_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
