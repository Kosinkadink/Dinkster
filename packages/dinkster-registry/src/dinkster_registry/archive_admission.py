"""Archive admission shared by engine validation and registry publication."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unicodedata
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from io import BufferedReader
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, cast

from blake3 import blake3
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    canonical_name,
    reserved_root,
    schema_from_wire,
    schema_signature,
    validate_claim,
    validate_name,
)
from dinkster_workers import MEBIBYTE
from dinkster_workers import doctor as dinkster_doctor
from dinkster_workers.doctor import (
    _check_assets,  # pyright: ignore[reportPrivateUsage]
    _check_blueprints,  # pyright: ignore[reportPrivateUsage]
    _check_environment,  # pyright: ignore[reportPrivateUsage]
    _check_icon,  # pyright: ignore[reportPrivateUsage]
    _check_namespace_coverage,  # pyright: ignore[reportPrivateUsage]
    _check_requires,  # pyright: ignore[reportPrivateUsage]
    _check_reserved_namespaces,  # pyright: ignore[reportPrivateUsage]
    _check_sandbox_needs,  # pyright: ignore[reportPrivateUsage]
    _check_templates,  # pyright: ignore[reportPrivateUsage]
    _probe_findings,  # pyright: ignore[reportPrivateUsage]
    _scan_sources,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_workers.manifest import ManifestError, PackManifest, load_manifest
from dinkster_workers.sandbox import SandboxError, detect_probe_jail
from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from .artifact import MANIFEST_FILENAME, build_artifact
from .model import artifact_digest

SCAN_POLICY_VERSION = "registry-scan/1"
ARCHIVE_LIMIT = 64 * MEBIBYTE
EXPANDED_LIMIT = 256 * MEBIBYTE
FILE_LIMIT = 64 * MEBIBYTE
ENTRY_LIMIT = 10_000
PATH_BYTE_LIMIT = 512
MANIFEST_LIMIT = MEBIBYTE
_SKIP_DIRECTORIES = frozenset(
    {".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", "build", "dist", "node_modules"}
)

FindingEffect = Literal["block", "review", "signal"]
FindingSeverity = Literal["info", "warning", "error", "critical"]


@dataclass(frozen=True)
class AdmissionFinding:
    code: str
    severity: FindingSeverity
    effect: FindingEffect
    message: str


@dataclass(frozen=True)
class StageResult:
    findings: tuple[AdmissionFinding, ...] = ()
    data: dict[str, Any] | None = None
    operator_details: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return any(finding.effect == "block" for finding in self.findings)

    @property
    def review(self) -> bool:
        return any(finding.effect == "review" for finding in self.findings)


def _block(code: str, message: str) -> AdmissionFinding:
    return AdmissionFinding(code, "error", "block", message)


def _review(code: str, message: str) -> AdmissionFinding:
    return AdmissionFinding(code, "warning", "review", message)


def _signal(code: str, message: str, *, severity: FindingSeverity = "warning") -> AdmissionFinding:
    return AdmissionFinding(code, severity, "signal", message)


def _safe_archive_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > ENTRY_LIMIT:
        raise ValueError(f"archive must contain between 1 and {ENTRY_LIMIT} entries")
    names: set[str] = set()
    folded: set[str] = set()
    expanded = 0
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or "\\" in name
            or "\x00" in name
            or path.is_absolute()
            or PureWindowsPath(name).is_absolute()
            or ".." in path.parts
            or name != unicodedata.normalize("NFC", name)
        ):
            raise ValueError(f"archive path is unsafe: {name!r}")
        if name.endswith("/") or info.is_dir():
            raise ValueError(f"archive directories must be implicit: {name!r}")
        if len(name.encode("utf-8")) > PATH_BYTE_LIMIT:
            raise ValueError(f"archive path exceeds {PATH_BYTE_LIMIT} UTF-8 bytes: {name!r}")
        if name in names:
            raise ValueError(f"archive path is duplicated: {name!r}")
        names.add(name)
        folded_name = name.casefold()
        if folded_name in folded:
            raise ValueError(f"archive paths collide case-insensitively: {name!r}")
        folded.add(folded_name)
        if info.create_system != 3 or mode != 0o644:
            raise ValueError(f"archive entry is not a regular 0644 file: {name!r}")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"archive entry uses unsupported compression: {name!r}")
        if info.flag_bits & 0x1:
            raise ValueError(f"archive entry is encrypted: {name!r}")
        if info.date_time != (1980, 1, 1, 0, 0, 0) or info.extra or info.comment:
            raise ValueError(f"archive entry metadata is not deterministic: {name!r}")
        if info.file_size > FILE_LIMIT:
            raise ValueError(f"archive file exceeds {FILE_LIMIT} bytes: {name!r}")
        if name == MANIFEST_FILENAME and info.file_size > MANIFEST_LIMIT:
            raise ValueError(f"archive manifest exceeds {MANIFEST_LIMIT} bytes")
        expanded += info.file_size
        if expanded > EXPANDED_LIMIT:
            raise ValueError(f"archive expands beyond {EXPANDED_LIMIT} bytes")
    if MANIFEST_FILENAME not in names:
        raise ValueError(f"archive has no root {MANIFEST_FILENAME}")
    return infos


def run_ingest_integrity(data: bytes, declared_size: int, declared_digest: str) -> StageResult:
    if len(data) != declared_size:
        return StageResult(
            (_block("integrity.size-mismatch", "Object length differs from submission."),)
        )
    actual_digest = artifact_digest(data)
    if actual_digest != declared_digest:
        return StageResult(
            (_block("integrity.digest-mismatch", "Object BLAKE3 differs from submission."),)
        )
    if not 1 <= len(data) <= ARCHIVE_LIMIT:
        return StageResult(
            (_block("integrity.archive-size", f"Archive exceeds the {ARCHIVE_LIMIT}-byte limit."),)
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.comment:
                raise ValueError("archive comment must be empty")
            infos = _safe_archive_infos(archive)
            if archive.testzip() is not None:
                raise ValueError("archive member failed its CRC check")
            if any(
                part in _SKIP_DIRECTORIES
                for info in infos
                for part in PurePosixPath(info.filename).parts
            ):
                raise ValueError("archive bytes are not the deterministic serialization")
            with tempfile.TemporaryDirectory(prefix="dinkster-registry-integrity-") as temporary:
                root = Path(temporary) / "sources"
                root.mkdir()
                for info in infos:
                    destination = root / info.filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(info))
                rebuilt = Path(temporary) / "rebuilt.zip"
                build_artifact(root, rebuilt)
                rebuilt_digest = artifact_digest(rebuilt.read_bytes())
        if rebuilt_digest != actual_digest:
            raise ValueError("archive bytes are not the deterministic serialization")
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return StageResult((_block("integrity.invalid-archive", str(exc)),))
    return StageResult(data={"artifactDigest": actual_digest, "size": len(data)})


@contextmanager
def extracted_archive(data: bytes) -> Generator[Path]:
    with tempfile.TemporaryDirectory(prefix="dinkster-registry-pack-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in _safe_archive_infos(archive):
                destination = root / info.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
        yield root


def _pack_metadata(manifest_path: Path) -> dict[str, object]:
    with manifest_path.open("rb") as source:
        document = tomllib.load(source)
    pack = document.get("pack")
    if not isinstance(pack, dict):
        raise ValueError("manifest is missing [pack]")
    return cast("dict[str, object]", pack)


def _manifest_payload(manifest: PackManifest, release: dict[str, object]) -> dict[str, Any]:
    contracts = manifest.contracts
    assert contracts is not None
    return {
        "dinkster": manifest.compatibility.engine
        if manifest.compatibility is not None
        else release.get("dinkster", ""),
        "license": release.get("license", ""),
        "contracts": {
            "host": contracts.host,
            "api": contracts.api,
            "inference": contracts.inference,
        },
        "platforms": list(manifest.platforms),
        "sandbox": {
            "gpu": manifest.sandbox.gpu,
            "network": manifest.sandbox.network,
            "writableMounts": manifest.sandbox.writable_mounts,
        },
        "pythonRequirements": list(manifest.requires),
        "nodeSource": manifest.node_source,
        "dependencies": [
            {"packName": dependency.pack, "versionRange": dependency.version}
            for dependency in manifest.dependencies
        ],
        "repository": release.get("repository"),
        "namespaces": list(manifest.namespaces),
        "localeCatalogs": [
            {
                "locale": catalog.locale,
                "digest": catalog.digest,
                "path": f"locales/{catalog.locale}.json",
            }
            for catalog in manifest.locale_catalogs
        ],
        "templates": [
            {
                "id": template.id,
                "name": template.name,
                "digest": template.digest,
                "path": template.path.as_posix(),
                "description": template.description,
                "tags": list(template.tags),
                "family": template.family,
                "models": list(template.models),
                "assets": list(template.assets),
                "thumbnailDigest": template.thumbnail.digest if template.thumbnail else "",
                "thumbnailMediaType": template.thumbnail.media_type if template.thumbnail else "",
                "thumbnailPath": template.thumbnail.path.as_posix() if template.thumbnail else "",
            }
            for template in manifest.templates
        ],
    }


def run_manifest_admission(
    root: Path,
    pack_name: str,
    release_version: str,
    *,
    publisher: str | None = None,
    engine_version: str | None = None,
) -> StageResult:
    """Validate archive declarations; catalog collisions remain caller-owned."""
    try:
        manifest = load_manifest(root / MANIFEST_FILENAME)
        release = _pack_metadata(root / MANIFEST_FILENAME)
        problem = validate_name(manifest.name)
        if problem is not None:
            raise ValueError(f"invalid pack name {manifest.name!r}: {problem}")
        canonical = canonical_name(manifest.name)
        if canonical != manifest.name:
            raise ValueError(f"pack name must use canonical hyphen spelling: {canonical!r}")
        try:
            version = Version(release_version)
        except ValueError as exc:
            raise ValueError(f"invalid public PEP 440 version: {release_version!r}") from exc
        if version.local is not None:
            raise ValueError("local PEP 440 versions are not supported by the registry")
        if str(version) != release_version:
            raise ValueError(f"version must use normalized PEP 440 form: {str(version)!r}")
        if manifest.name != pack_name:
            raise ValueError(
                f"manifest pack name {manifest.name!r} does not match submission {pack_name!r}"
            )
        if manifest.compatibility is not None:
            if manifest.compatibility.version != release_version:
                raise ValueError(
                    f"manifest pack version {manifest.compatibility.version!r} "
                    f"does not match submission {release_version!r}"
                )
            if engine_version is not None:
                manifest.compatibility.require_engine(pack_name, engine_version)
        dinkster_range = release.get("dinkster", "")
        if not isinstance(dinkster_range, str):
            raise ValueError("[pack] dinkster must be a version range")
        SpecifierSet(dinkster_range)
        license_expression = release.get("license")
        if license_expression is not None:
            if not isinstance(license_expression, str):
                raise ValueError("[pack] license must be an SPDX expression")
            normalized_license = canonicalize_license_expression(license_expression)
            if "LicenseRef-" in normalized_license and not (root / "LICENSE").is_file():
                raise ValueError("a LicenseRef license requires a root LICENSE file")
        repository = release.get("repository")
        if repository is not None and (
            not isinstance(repository, str) or not repository.startswith("https://")
        ):
            raise ValueError("[pack] repository must be an HTTPS URL")
        if manifest.contracts is None:
            raise ValueError("manifest is missing [pack.contracts]")
        for claim in (manifest.name, *manifest.namespaces):
            problem = validate_claim(claim)
            if problem is not None:
                raise ValueError(f"invalid namespace claim {claim!r}: {problem}")
            if reserved_root(claim) is not None and publisher != "dinkster":
                raise ValueError(f"namespace claim is reserved: {claim!r}")
        if any(dependency.pack == pack_name for dependency in manifest.dependencies):
            raise ValueError("a pack cannot depend on itself")
    except (
        AssertionError,
        InvalidLicenseExpression,
        InvalidSpecifier,
        ManifestError,
        OSError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as exc:
        return StageResult((_block("manifest.invalid", str(exc)),))
    return StageResult(data=_manifest_payload(manifest, release))


def run_doctor_static(root: Path) -> StageResult:
    manifest_path = root / MANIFEST_FILENAME
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        return StageResult((_block("doctor.manifest-invalid", str(exc)),))
    findings = [
        *_check_requires(manifest),
        *_check_environment(manifest),
        *_check_sandbox_needs(manifest),
        *_check_reserved_namespaces(manifest),
        *_check_icon(manifest),
        *_check_blueprints(manifest),
        *_check_assets(manifest),
        *_check_templates(manifest),
        *_scan_sources(manifest),
    ]
    mapped = tuple(
        AdmissionFinding(
            finding.code,
            finding.severity,
            "block" if finding.severity == "error" else "signal",
            finding.message,
        )
        for finding in findings
    )
    errors = sum(finding.severity == "error" for finding in findings)
    warnings = sum(finding.severity == "warning" for finding in findings)
    return StageResult(
        mapped,
        {
            "status": "failed" if errors else "warning" if warnings else "passed",
            "doctorVersion": distribution_version("dinkster-api"),
            "reportVersion": dinkster_doctor.DOCTOR_REPORT_VERSION,
            "errorCount": errors,
            "warningCount": warnings,
        },
    )


@dataclass(frozen=True)
class ProbeSettings:
    probe_timeout_seconds: int = 30
    probe_dependency_timeout_seconds: int = 300
    probe_package_index_url: str = "https://pypi.org/simple"
    probe_output_limit_bytes: int = 2 * MEBIBYTE
    probe_operator_tail_bytes: int = 8 * 1024

    def __post_init__(self) -> None:
        bounds = (
            ("probe_timeout_seconds", self.probe_timeout_seconds, 1, 600),
            ("probe_dependency_timeout_seconds", self.probe_dependency_timeout_seconds, 1, 1800),
            ("probe_output_limit_bytes", self.probe_output_limit_bytes, 4096, 4 * MEBIBYTE),
            (
                "probe_operator_tail_bytes",
                self.probe_operator_tail_bytes,
                512,
                self.probe_output_limit_bytes,
            ),
        )
        for name, value, minimum, maximum in bounds:
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")


@dataclass(frozen=True)
class _ProbeProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    output_limit: str | None
    timed_out: bool


def _probe_environment() -> dict[str, str]:
    return {
        "PATH": os.pathsep.join(
            (str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin")
        ),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "UV_CONCURRENT_DOWNLOADS": "1",
        "UV_CONCURRENT_INSTALLS": "1",
    }


def _read_probe_pipe(
    pipe: BufferedReader,
    limit: int,
    output: bytearray,
    exceeded: threading.Event,
) -> None:
    try:
        while chunk := pipe.read(64 * 1024):
            remaining = limit - len(output)
            output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                exceeded.set()
                return
    finally:
        pipe.close()


def _kill_probe_group(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        raise OSError("Executable admission requires POSIX process groups")
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_probe_process(
    command: list[str],
    *,
    timeout_seconds: int,
    output_limit_bytes: int,
    pass_fds: tuple[int, ...] = (),
) -> _ProbeProcessResult:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=_probe_environment(),
        pass_fds=pass_fds,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    stdout_exceeded = threading.Event()
    stderr_exceeded = threading.Event()
    threads = (
        threading.Thread(
            target=_read_probe_pipe,
            args=(process.stdout, output_limit_bytes, stdout, stdout_exceeded),
            daemon=True,
        ),
        threading.Thread(
            target=_read_probe_pipe,
            args=(process.stderr, output_limit_bytes, stderr, stderr_exceeded),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    timed_out = False
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if stdout_exceeded.is_set() or stderr_exceeded.is_set():
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.01)
    _kill_probe_group(process)
    for thread in threads:
        thread.join(timeout=5)
    output_limit = (
        "stdout" if stdout_exceeded.is_set() else "stderr" if stderr_exceeded.is_set() else None
    )
    return _ProbeProcessResult(
        process.returncode, bytes(stdout), bytes(stderr), output_limit, timed_out
    )


def _probe_operator_details(
    result: _ProbeProcessResult, jail: str, tail_limit_bytes: int
) -> dict[str, object]:
    return {
        "jail": jail,
        "returnCode": result.returncode,
        "stdoutTail": result.stdout[-tail_limit_bytes:].decode("utf-8", errors="replace"),
        "stderrTail": result.stderr[-tail_limit_bytes:].decode("utf-8", errors="replace"),
    }


_PUBLIC_PROBE_EXCEPTION_TYPES = frozenset(
    {
        "AssertionError",
        "ImportError",
        "MemoryError",
        "ModuleNotFoundError",
        "OSError",
        "RuntimeError",
        "SystemExit",
        "TypeError",
        "ValueError",
    }
)
_PROBE_FINDING_MESSAGES = {
    "entry.unresolvable": "The pack entry point could not be imported.",
    "entry.not-node-classes": "The pack entry point did not return only Node subclasses.",
    "schema.invalid": "A pack node returned an invalid schema.",
    "schema.replacement-invalid": "A pack node has an invalid replacement rule.",
    "schema.duplicate-node-type": "The pack returned duplicate node types.",
    "entry.types-failed": "The pack type-registration entry point failed.",
    "import.side-effect-output": (
        "The pack wrote to standard output or standard error during import."
    ),
    "import.log": "The pack emitted structured logs during import.",
    "import.log-foreign-origin": "The pack logged under another pack's logger origin.",
    "import.side-effect-cuda": "The pack initialized a CUDA context during import.",
    "import.side-effect-threads": "The pack started threads during import.",
    "import.slow": "The pack import exceeded the doctor's slow-import threshold.",
    "types.fallback-codec": "A schema type uses the fallback boundary codec.",
    "types.unregistered": "A schema references an unregistered type.",
    "namespace.uncovered": "A node type is outside the pack's declared namespaces.",
}


def _probe_failure_message(stdout: bytes) -> str:
    try:
        payload = json.loads(stdout.decode("utf-8"))
        error = payload["error"]
        error_type = error["type"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return "Executable probe failed before producing a valid structured error."
    if isinstance(error_type, str) and error_type in _PUBLIC_PROBE_EXCEPTION_TYPES:
        return f"Executable probe failed with {error_type}."
    return "Executable probe failed with a pack error."


def run_executable_probe(
    root: Path,
    *,
    settings: ProbeSettings | None = None,
    require_isolation: bool = True,
) -> StageResult:
    configured = settings or ProbeSettings()
    try:
        manifest = load_manifest(root / MANIFEST_FILENAME)
        requirements = [Requirement(value) for value in manifest.requires_for("cpu")]
    except (ManifestError, InvalidRequirement):
        return StageResult(
            (_block("probe.dependencies-invalid", "Pack requirements are invalid."),)
        )
    if any(requirement.url is not None for requirement in requirements):
        return StageResult(
            (
                _block(
                    "probe.dependencies-invalid", "Probe dependencies must use the package index."
                ),
            )
        )
    if not requirements:
        return _run_executable_probe(root, configured, require_isolation)
    installer = shutil.which("uv")
    if installer is None:
        return StageResult(
            (_block("probe.dependencies-unavailable", "The dependency installer is unavailable."),)
        )
    # The existing probe mount exposes dependencies read-only, without mounting
    # another host directory or changing the submitted archive bytes.
    with tempfile.TemporaryDirectory(prefix=".registry-probe-", dir=root) as directory:
        dependencies = Path(directory)
        common = [installer, "--no-config", "--no-cache", "--no-python-downloads", "--no-progress"]
        command = [
            *common,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(dependencies),
            "--only-binary",
            ":all:",
            "--keyring-provider",
            "disabled",
            "--index-url",
            configured.probe_package_index_url,
            "--no-sources",
            "--torch-backend",
            "cpu",
            *[str(requirement) for requirement in requirements],
        ]
        try:
            completed = _run_probe_process(
                command,
                timeout_seconds=configured.probe_dependency_timeout_seconds,
                output_limit_bytes=configured.probe_output_limit_bytes,
            )
        except OSError:
            return StageResult(
                (_block("probe.dependencies-unavailable", "Dependency provisioning failed."),)
            )
        if completed.returncode or completed.timed_out or completed.output_limit:
            operator_details = _probe_operator_details(
                completed, "dependency-installer", configured.probe_operator_tail_bytes
            )
            operator_details["timedOut"] = completed.timed_out
            operator_details["limitedStream"] = completed.output_limit
            return StageResult(
                (
                    _block(
                        "probe.dependencies-unresolved",
                        "Declared dependencies could not be provisioned as wheels "
                        "within the limits.",
                    ),
                ),
                operator_details=operator_details,
            )
        # Dependency .pth files must not run before the trusted harness.
        return _run_executable_probe(root, configured, require_isolation, dependencies)


def _run_executable_probe(
    root: Path,
    configured: ProbeSettings,
    require_isolation: bool,
    dependencies: Path | None = None,
) -> StageResult:
    manifest_path = root / MANIFEST_FILENAME
    jail = None
    if require_isolation:
        try:
            jail = detect_probe_jail()
        except SandboxError as exc:
            return StageResult(
                (_block("probe.isolation-unavailable", "A full Linux probe jail is unavailable."),),
                operator_details={
                    "isolationError": str(exc)[-configured.probe_operator_tail_bytes :]
                },
            )
        if not jail.user_namespaces:
            return StageResult(
                (_block("probe.isolation-unavailable", "A full Linux probe jail is unavailable."),),
                operator_details={"jail": "degraded"},
            )
    probe_module = (
        "dinkster_registry._archive_probe" if jail is None else "dinkster_workers._doctor_probe"
    )
    command = [sys.executable, "-m", probe_module, str(manifest_path)]
    environment = _probe_environment()
    if dependencies is not None:
        # Trusted bootstrap imports the harness before adding dependency paths.
        paths = [path for path in sys.path if path and Path(path).is_absolute()]
        environment["PYTHONPATH"] = os.pathsep.join([*paths, str(dependencies)])
        if jail is None:
            command = [
                sys.executable,
                "-c",
                "import runpy, sys; "
                f"sys.path.extend([{str(dependencies)!r}]); "
                "sys.argv = sys.argv[1:]; "
                f"runpy.run_module({probe_module!r}, run_name='__main__')",
                probe_module,
                str(manifest_path),
            ]
    try:
        if jail is None:
            completed = _run_probe_process(
                command,
                timeout_seconds=configured.probe_timeout_seconds,
                output_limit_bytes=configured.probe_output_limit_bytes,
            )
        else:
            with tempfile.TemporaryFile() as environment_file:
                environment_file.write(json.dumps(environment).encode("utf-8"))
                environment_file.flush()
                environment_file.seek(0)
                env_fd = environment_file.fileno()
                completed = _run_probe_process(
                    jail.wrap(command, root, env_fd=env_fd),
                    timeout_seconds=configured.probe_timeout_seconds,
                    output_limit_bytes=configured.probe_output_limit_bytes,
                    pass_fds=(env_fd,),
                )
    except OSError as exc:
        return StageResult(
            (_block("probe.launch-failed", "The executable probe could not be started."),),
            operator_details={"exceptionType": type(exc).__name__},
        )
    jail_name = "user-pid-namespace" if jail is not None else "disabled-for-test"
    operator_details = _probe_operator_details(
        completed, jail_name, configured.probe_operator_tail_bytes
    )
    if completed.timed_out:
        return StageResult(
            (
                _block(
                    "probe.timeout",
                    f"Executable probe exceeded {configured.probe_timeout_seconds}s.",
                ),
            ),
            operator_details=operator_details,
        )
    if completed.output_limit is not None:
        operator_details["limitedStream"] = completed.output_limit
        return StageResult(
            (
                _block(
                    "probe.output-limit",
                    f"Executable probe exceeded the {completed.output_limit} output limit.",
                ),
            ),
            operator_details=operator_details,
        )
    if completed.returncode != 0:
        return StageResult(
            (_block("probe.failed", _probe_failure_message(completed.stdout)),),
            operator_details=operator_details,
        )
    if completed.stderr:
        return StageResult(
            (_block("probe.failed", "Executable probe wrote unexpected process diagnostics."),),
            operator_details=operator_details,
        )
    try:
        payload = cast("dict[str, Any]", json.loads(completed.stdout.decode("utf-8")))
        manifest = load_manifest(manifest_path)
        schemas: list[dict[str, Any]]
        if jail is None:
            doctor_report = cast("dict[str, Any]", payload["doctorReport"])
            schemas = cast("list[dict[str, Any]]", payload["schemas"])
        else:
            doctor_report = payload
            catalog = cast("dict[str, Any]", doctor_report["catalog"])
            catalog_schemas = cast("dict[str, dict[str, Any]]", catalog["schemas"])
            schemas = []
            for node_type, wire_schema in catalog_schemas.items():
                schema = schema_from_wire(wire_schema)
                schemas.append(
                    {
                        "nodeType": node_type,
                        "schemaVersion": schema.version,
                        "schemaWireVersion": SCHEMA_WIRE_VERSION,
                        "schemaSignature": schema_signature(schema),
                        "role": "schema-only" if node_type in manifest.schema_only else "executing",
                        "schema": wire_schema,
                    }
                )
        probe_findings = _probe_findings(doctor_report, manifest)
        probe_findings.extend(
            _check_namespace_coverage(
                manifest, tuple(cast("str", schema["nodeType"]) for schema in schemas)
            )
        )
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        ManifestError,
    ):
        return StageResult(
            (_block("probe.invalid-report", "Executable probe returned an invalid report."),),
            operator_details=operator_details,
        )
    findings = [
        AdmissionFinding(
            finding.code,
            finding.severity,
            "block" if finding.severity == "error" else "signal",
            _PROBE_FINDING_MESSAGES.get(
                finding.code, "The executable probe reported a pack policy finding."
            ),
        )
        for finding in probe_findings
    ]
    observed_gpu = bool(doctor_report.get("cuda_initialized_on_import"))
    findings.append(
        _signal(
            "probe.declared-observed-needs",
            "Declared needs "
            f"gpu={manifest.sandbox.gpu}, network={manifest.sandbox.network}, "
            f"writable={manifest.sandbox.writable_mounts}; observed import needs "
            f"gpu={observed_gpu}, network=false, writable=false.",
            severity="info",
        )
    )
    if observed_gpu and not manifest.sandbox.gpu:
        findings.append(_review("probe.undeclared-gpu", "Import initialized an undeclared GPU."))
    if (
        not schemas
        and manifest.node_source != "remote-gateway-catalog"
        and not any(finding.effect == "block" for finding in findings)
    ):
        findings.append(_block("probe.no-schemas", "Pack probe returned no node schemas."))
    return StageResult(tuple(findings), {"schemas": schemas}, operator_details)


_SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_NATIVE_SUFFIXES = (".dll", ".dylib", ".exe", ".pyd", ".so")
_KNOWN_BAD_HASHES = frozenset({blake3(b"dinkster-known-bad-test-payload").hexdigest()})


def _nested_archive_format(payload: bytes) -> str | None:
    signatures = (
        (b"PK\x03\x04", "ZIP"),
        (b"\x1f\x8b", "gzip"),
        (b"\x28\xb5\x2f\xfd", "Zstandard"),
        (b"\xfd7zXZ\x00", "XZ"),
    )
    for signature, name in signatures:
        if payload.startswith(signature):
            return name
    if len(payload) >= 262 and payload[257:262] == b"ustar":
        return "tar"
    return None


def run_fast_security(data: bytes) -> StageResult:
    findings: list[AdmissionFinding] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = _safe_archive_infos(archive)
            for info in infos:
                payload = archive.read(info)
                if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
                    findings.append(
                        _block(
                            "security.confirmed-secret", f"Secret pattern found in {info.filename}."
                        )
                    )
                if blake3(payload).hexdigest() in _KNOWN_BAD_HASHES:
                    findings.append(
                        _block("security.known-bad-file", f"Known-bad file hash: {info.filename}.")
                    )
                lower = info.filename.lower()
                native_magic = payload.startswith((b"\x7fELF", b"MZ", b"\xcf\xfa\xed\xfe"))
                if lower.endswith(_NATIVE_SUFFIXES) or native_magic:
                    findings.append(
                        _review(
                            "security.native-binary",
                            f"Native or executable file requires review: {info.filename}.",
                        )
                    )
                nested_format = _nested_archive_format(payload)
                if nested_format is not None:
                    findings.append(
                        _review(
                            "security.nested-archive",
                            f"Nested {nested_format} archive requires review: {info.filename}.",
                        )
                    )
                if lower.endswith((".pyc", ".pyo")) or (
                    len(payload) > 512
                    and re.search(
                        rb"(?:exec|eval)\s*\(.*(?:base64|marshal|zlib)", payload, re.DOTALL
                    )
                ):
                    findings.append(
                        _review(
                            "security.obfuscated-code",
                            "Obfuscated or bytecode-only content requires review: "
                            f"{info.filename}.",
                        )
                    )
    except (ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        findings.append(_block("security.archive-policy", str(exc)))
    return StageResult(
        tuple(findings),
        {
            "status": "failed"
            if any(finding.effect == "block" for finding in findings)
            else "warning"
            if findings
            else "passed",
            "policyVersion": SCAN_POLICY_VERSION,
        },
    )
