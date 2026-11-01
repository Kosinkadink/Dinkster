"""Archive admission shared by engine validation and registry publication."""

from __future__ import annotations

import io
import re
import tempfile
import unicodedata
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from blake3 import blake3

from .artifact import MANIFEST_FILENAME, build_artifact
from .model import artifact_digest

SCAN_POLICY_VERSION = "registry-scan/1"
ARCHIVE_LIMIT = 64 * 1024 * 1024
EXPANDED_LIMIT = 256 * 1024 * 1024
FILE_LIMIT = 64 * 1024 * 1024
ENTRY_LIMIT = 10_000
PATH_BYTE_LIMIT = 512
MANIFEST_LIMIT = 1024 * 1024
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
