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
canonical cell/platform/interpreter identity plus the resolved base closure,
so kitchen, aimdo and every other code-layer pin can change without a new
base. The code layer is the hash-locked wheelhouse uv.lock names for the cell
minus every base distribution, plus the workspace release wheels, the git
dependency built to a local wheel, and the pinned frontend bundle wheel from
scripts/build_release.py and scripts/release_sources.json. Installing a
release never touches GitHub or PyPI: all wheels are resolved before install
and the generated requirements carry only hashes, no URLs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_release import TAG_PATTERN, release_version, wheel_metadata

MANIFEST_FORMAT = "dinkster.engine/1"
CHANNEL_FORMAT = "dinkster.engine-channel/1"
CELLS_FORMAT = "dinkster.engine-cells/1"
BASE_IDENTITY_SCHEMA = "dinkster.engine-base-id/1"
MINIMUM_LAUNCHER_VERSION = "0.0.1"

CONTROL_ENVIRONMENT = "control"
EXECUTION_ENVIRONMENT = "execution"
ENVIRONMENTS = (CONTROL_ENVIRONMENT, EXECUTION_ENVIRONMENT)

BASE_DIR = "base"
STORE_DIR = "store"
ENGINE_DIR = "engine"
CHANNELS_DIR = "channels"
RECORDS_PATH = f"{BASE_DIR}/records.json"

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
    packages: dict[str, str],
) -> str:
    """The base id: cell, platform, interpreter and the resolved closure only.

    Kitchen, aimdo and every other code-layer pin are absent by construction,
    so changing them cannot change the base id.
    """
    identity = {
        "schema": BASE_IDENTITY_SCHEMA,
        "cell": cell_name,
        "platform": {"os": os_name, "arch": arch},
        "python": {"implementation": python_implementation, "version": python_version},
        "packages": dict(sorted(packages.items())),
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _pin_identity(
    config: CellConfig,
) -> dict[str, object]:
    """What a base build starts from, used to skip unchanged rebuilds.

    Distinct from the base id: the id hashes the closure that resolution
    actually installed, while this records the requested pins so a rerun can
    recognize that the same pins already produced a verified archive.
    """
    return {
        "cell": config.name,
        "platform": {"os": config.os_name, "arch": config.arch},
        "python": {
            "implementation": config.python_implementation,
            "version": config.python_version,
        },
        "torch_requirement": config.recipe.torch_requirement,
        "torchvision_requirement": config.recipe.torchvision_requirement,
        "index_url": config.recipe.index_url,
    }


def _pin_identity_hash(config: CellConfig) -> str:
    return hashlib.sha256(_canonical_json(_pin_identity(config)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Base archive construction


def _uv_python_install_dir(uv: str, python_version: str) -> Path:
    _run([uv, "python", "install", python_version])
    python = Path(_run([uv, "python", "find", python_version]).strip())
    return python.parents[1]


def _staging_python(staging_root: Path, os_name: str) -> Path:
    if os_name == "windows":
        return staging_root / "python.exe"
    return staging_root / "bin" / "python3"


def _installed_packages(staging_python: Path) -> dict[str, str]:
    script = (
        "import json, importlib.metadata as m\n"
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))\n"
    )
    output = _run([str(staging_python), "-S", "-c", script])
    return {normalize_name(name): version for name, version in json.loads(output).items()}


def _clean_site_caches(staging_root: Path) -> None:
    for site_packages in staging_root.glob("lib/python*/site-packages"):
        for cache in site_packages.rglob("__pycache__"):
            if cache.is_dir():
                shutil.rmtree(cache, ignore_errors=True)


def _copy_uv_binary(uv: str, staging_root: Path, os_name: str) -> str:
    uv_rel = "tools/uv/uv.exe" if os_name == "windows" else "tools/uv/uv"
    source = shutil.which(uv)
    if source is None:
        raise FeedError(f"uv executable {uv!r} not found on PATH")
    destination = staging_root / uv_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if os_name != "windows":
        destination.chmod(0o755)
    return uv_rel


def _validated_link_target(member_rel: str, target: str) -> None:
    """Symlinks in the archive must stay inside the extracted base root.

    Absolute targets and escapes would break relocatability or point outside
    the archive, so the build refuses them instead of shipping them.
    """
    if target.startswith("/"):
        raise FeedError(f"base archive would contain absolute symlink {member_rel} -> {target}")
    parent = os.path.dirname(member_rel)
    resolved = os.path.normpath(os.path.join(parent, target))
    if resolved.startswith(".."):
        raise FeedError(f"base archive symlink {member_rel} -> {target} escapes the base root")


def _tar_entry_info(path: Path, staging_root: Path) -> tarfile.TarInfo:
    relative = path.relative_to(staging_root).as_posix()
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
    uv_path: str
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
    expected_id = base_identity_hash(
        config.name,
        config.os_name,
        config.arch,
        config.python_implementation,
        config.python_version,
        packages,
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
        uv_path=record["uv_path"],
        reused=True,
    )


def build_base(uv: str, config: CellConfig, feed_dir: Path) -> BaseBuild:
    """Materialize, hash and archive the cell's base, reusing a verified one."""
    require_native_cell(config)
    pin_hash = _pin_identity_hash(config)
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
        interpreter_dir = _uv_python_install_dir(uv, config.python_version)
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
        _clean_site_caches(staging_root)
        uv_rel = _copy_uv_binary(uv, staging_root, config.os_name)
        python_rel = staging_python.relative_to(staging_root).as_posix()
        base_id = base_identity_hash(
            config.name,
            config.os_name,
            config.arch,
            config.python_implementation,
            config.python_version,
            packages,
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
        uv_path=uv_rel,
        reused=False,
    )
    records[pin_hash] = {
        "id": build.base_id,
        "archive": build.archive_path,
        "sha256": build.sha256,
        "size": build.size,
        "packages": build.packages,
        "python_path": build.python_path,
        "uv_path": build.uv_path,
        "pin": _pin_identity(config),
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


def locked_closure(lock: dict[str, dict[str, Any]], roots: set[str]) -> dict[str, dict[str, Any]]:
    """Every distribution reachable from the roots through runtime dependencies.

    Only the ``dependencies`` edge list is followed, so dev dependencies and
    the lock's dev-dependency sections never enter the code layer.
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
        stack.extend(dependency["name"] for dependency in entry.get("dependencies", []))
    return closure


def lock_descendants(lock: dict[str, dict[str, Any]], seeds: set[str]) -> set[str]:
    """The seed distributions and everything reachable from them."""
    seen: set[str] = set()
    stack = sorted(seeds)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        entry = lock.get(name)
        if entry is not None:
            stack.extend(dependency["name"] for dependency in entry.get("dependencies", []))
    return seen


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


def select_cell_wheel(
    lock_name: str, wheels: list[dict[str, Any]], os_name: str, arch: str
) -> dict[str, Any]:
    """The one locked wheel for the cell, or a refusal naming the distribution."""
    candidates = [
        wheel
        for wheel in wheels
        if wheel_matches_cell(wheel["url"].rsplit("/", 1)[-1], os_name, arch)
    ]
    if not candidates:
        raise FeedError(
            f"distribution {lock_name} has no wheel for cell platform {os_name}/{arch} in uv.lock"
        )
    if len(candidates) > 1:
        raise FeedError(
            f"distribution {lock_name} has {len(candidates)} candidate wheels for "
            f"{os_name}/{arch}; the lock is ambiguous"
        )
    return candidates[0]


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
    the dependency must become a local wheel before the feed is complete.
    """
    clone_url = source_url.split("?", 1)[0].split("#", 1)[0]
    commit = source_url.rsplit("#", 1)[-1]
    if not COMMIT_PATTERN.fullmatch(commit):
        raise FeedError(f"git dependency source has no pinned commit: {source_url}")
    with tempfile.TemporaryDirectory(prefix="dinkster-git-dep-") as work:
        clone = Path(work) / "src"
        _run(["git", "clone", "--quiet", clone_url, str(clone)])
        _run(["git", "-C", str(clone), "checkout", "--quiet", commit])
        build_dir = Path(work) / "build"
        _run([uv, "build", "--wheel", "--out-dir", str(build_dir), str(clone)])
        wheels = sorted(build_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise FeedError(f"git dependency {clone_url} build produced {len(wheels)} wheels")
    return wheels[0]


def _download_locked_wheel(wheel: dict[str, Any], destination: Path) -> str:
    url = wheel["url"]
    expected = wheel["hash"]
    scheme, _, expected_sha = expected.partition(":")
    if scheme != "sha256" or len(expected_sha) != 64:
        raise FeedError(f"locked wheel hash is not a sha256 digest: {expected}")
    with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as target:
        digest = hashlib.sha256()
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            target.write(block)
    actual = digest.hexdigest()
    if actual != expected_sha:
        destination.unlink(missing_ok=True)
        raise FeedError(f"downloaded {url} hashes to {actual}, uv.lock says {expected_sha}")
    return actual


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
        name=name,
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
    closure = locked_closure(lock, workspace_members(lock))
    # Every base distribution is a descendant of torch or torchvision in the
    # lock graph, so excluding their subtree keeps the base and code layers
    # disjoint even when a base package is also a code dependency.
    base_names = lock_descendants(lock, {"torch", "torchvision"})
    code_names = sorted(set(closure) - base_names)

    with tempfile.TemporaryDirectory(prefix="dinkster-wheels-") as work:
        built = _build_workspace_wheels(uv, root, Path(work))
        entries: dict[str, WheelEntry] = {}
        for lock_name in code_names:
            entry = closure[lock_name]
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
    return result


# ---------------------------------------------------------------------------
# Manifests and requirements


@dataclass(frozen=True)
class RequirementsFile:
    path: str
    sha256: str
    size: int

    def as_json(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


def write_requirements(
    feed_dir: Path, commit: str, cell_name: str, entries: list[WheelEntry]
) -> RequirementsFile:
    """Hash-locked requirements generated from the wheels' own metadata."""
    lines = "".join(
        f"{entry.name}=={entry.version} --hash=sha256:{entry.sha256}\n"
        for entry in sorted(entries, key=lambda entry: entry.name)
    )
    relative = f"{ENGINE_DIR}/{commit}/{cell_name}.requirements.txt"
    path = feed_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lines, encoding="utf-8")
    return RequirementsFile(relative, sha256_file(path), path.stat().st_size)


def build_manifest(
    commit: str,
    cell_name: str,
    base: BaseBuild,
    entries: list[WheelEntry],
    requirements: RequirementsFile,
) -> dict[str, Any]:
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
            "uv": base.uv_path,
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
        "requirements": requirements.as_json(),
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
        import boto3
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

    if args.channel is not None:
        return _write_channel(root, feed_dir, commit, args)

    if not args.cells:
        raise FeedError("nothing to do: pass --cells or --channel")
    frontend_wheel = _resolve_frontend_wheel(root, args)
    for cell_name in [cell.strip() for cell in args.cells.split(",") if cell.strip()]:
        config = load_cell_config(root / "scripts/engine_cells.json", cell_name)
        base = build_base(args.uv, config, feed_dir)
        entries = build_code_layer(args.uv, root, feed_dir, config, base.packages, frontend_wheel)
        requirements = write_requirements(feed_dir, commit, cell_name, entries)
        manifest = build_manifest(commit, cell_name, base, entries, requirements)
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
    main()
