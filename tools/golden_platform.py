from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__:
    from .evidence_paths import DINKSTER_ROOT, EVIDENCE_ROOT
else:
    from evidence_paths import DINKSTER_ROOT, EVIDENCE_ROOT

GOLDEN_OFFLINE_ENV = "DINKSTER_GOLDENS_OFFLINE"
_MANIFEST_PATH = Path("platform-goldens/manifest.json")
_EVIDENCE_FILES = Path("platform-goldens/files")
_CACHE_ROOT = DINKSTER_ROOT / ".golden-cache"


class GoldenUnavailableError(RuntimeError):
    pass


class GoldenVariantNotFoundError(GoldenUnavailableError):
    pass


class GoldenIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_variant_path(path: Path, key: str, dinkster_root: Path) -> Path:
    selected = path.with_name(f"{path.stem}.{key}{path.suffix}").resolve()
    try:
        relative = selected.relative_to(dinkster_root.resolve())
    except ValueError as error:
        raise ValueError(f"golden path is outside Dinkster: {path}") from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid golden path: {relative}")
    return relative


def _manifest_entry(manifest: dict[str, Any], relative: Path, key: str) -> dict[str, str]:
    if manifest.get("format") != "dinkster-platform-goldens/1":
        raise GoldenIntegrityError("unsupported platform golden manifest format")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise GoldenIntegrityError("platform golden manifest files must be a list")
    matches = [
        entry
        for entry in files
        if isinstance(entry, dict)
        and entry.get("path") == relative.as_posix()
        and entry.get("platform") == key
    ]
    if not matches:
        raise GoldenVariantNotFoundError(f"no evidence golden for {relative.as_posix()} on {key}")
    if len(matches) != 1:
        raise GoldenIntegrityError(f"duplicate evidence golden for {relative.as_posix()} on {key}")
    entry = matches[0]
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise GoldenIntegrityError(f"invalid sha256 for {relative.as_posix()}")
    return {"path": relative.as_posix(), "platform": key, "sha256": sha256}


def fetch_platform_golden(
    path: Path,
    key: str,
    *,
    dinkster_root: Path = DINKSTER_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
    cache_root: Path = _CACHE_ROOT,
) -> Path:
    """Copy one verified platform fixture from evidence into the local cache."""
    relative = _relative_variant_path(path, key, dinkster_root)
    cached = cache_root / relative
    offline = os.environ.get(GOLDEN_OFFLINE_ENV)
    if offline is not None:
        if offline != "1":
            raise ValueError(f"{GOLDEN_OFFLINE_ENV} must be unset or 1")
        raise GoldenUnavailableError(
            f"{GOLDEN_OFFLINE_ENV}=1; platform comparison uses baseline only and is skipped"
        )

    manifest_path = evidence_root / _MANIFEST_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GoldenUnavailableError(f"evidence manifest not found: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise GoldenIntegrityError(f"invalid evidence manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise GoldenIntegrityError("platform golden manifest must be an object")
    entry = _manifest_entry(manifest, relative, key)

    if cached.is_file() and _sha256(cached) == entry["sha256"]:
        return cached

    source = evidence_root / _EVIDENCE_FILES / relative
    if not source.is_file():
        raise GoldenIntegrityError(f"manifest golden not found: {source}")
    if _sha256(source) != entry["sha256"]:
        raise GoldenIntegrityError(f"evidence golden sha256 mismatch: {source}")

    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        if _sha256(temporary_path) != entry["sha256"]:
            raise GoldenIntegrityError(f"cached golden sha256 mismatch: {relative}")
        temporary_path.replace(cached)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cached


def platform_variant_output_path(
    path: Path,
    key: str,
    *,
    dinkster_root: Path = DINKSTER_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
) -> Path:
    files_root = evidence_root / _EVIDENCE_FILES
    if not files_root.is_dir():
        raise RuntimeError(f"dinkster-evidence platform-goldens checkout not found: {files_root}")
    return files_root / _relative_variant_path(path, key, dinkster_root)


def runtime_variant_output_path(
    path: Path,
    runtime: str,
    *,
    dinkster_root: Path = DINKSTER_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
) -> Path:
    if sys.platform.startswith("linux"):
        return dinkster_root / _relative_variant_path(path, runtime, dinkster_root)
    return platform_variant_output_path(
        path,
        f"{sys.platform}-{runtime}",
        dinkster_root=dinkster_root,
        evidence_root=evidence_root,
    )


def platform_golden_path(
    path: Path,
    torch_version: str,
    *,
    dinkster_root: Path = DINKSTER_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
) -> Path:
    if sys.platform.startswith("linux"):
        return path
    key = f"{sys.platform}-py{platform.python_version()}-torch{torch_version}"
    return platform_variant_output_path(
        path, key, dinkster_root=dinkster_root, evidence_root=evidence_root
    )


def cpu_identity() -> str:
    """CPU model of the host executing the generator.

    Executed-reference goldens that are asserted bit-exactly are pinned to
    the mint host's CPU class: conv, FFT, and vectorized libm kernels
    dispatch on microarchitecture and drift by ULPs across CPUs (same class
    as the GPU host drift in Dinkster issue #636). The test loader skips
    exact-equality suites when the live CPU differs from the recorded one.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown-cpu"


def tuple_provenance(torch_version: str, *, pin_cpu: bool = False) -> dict[str, str]:
    """Provenance keys the platform-tuple loader validates. The Linux base
    fixture stays free of provenance unless the generator opts into CPU
    pinning (pin_cpu=True, for fixtures whose executed kernels drift across
    CPU microarchitectures); other platforms record the full platform
    tuple."""
    if sys.platform.startswith("linux"):
        return {"cpu": cpu_identity()} if pin_cpu else {}
    return platform_provenance(torch_version, pin_cpu=pin_cpu)


def platform_provenance(torch_version: str, *, pin_cpu: bool = False) -> dict[str, str]:
    provenance: dict[str, str] = {}
    if pin_cpu:
        provenance["cpu"] = cpu_identity()
    provenance.update(
        {
            "python": f"Python {platform.python_version()}",
            "torch": torch_version,
        }
    )
    if not sys.platform.startswith("linux"):
        provenance.update({"os": platform.platform(), "platform": sys.platform})
    return provenance
