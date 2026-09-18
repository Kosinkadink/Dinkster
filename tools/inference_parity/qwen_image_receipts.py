"""Pure verification for the official base Qwen Image artifact receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from dinkster_inference.qwen_image_layout import qwen_image_dit_layout
from dinkster_inference.qwen_image_text import qwen_image_text_layout

_PROVIDER_REPOSITORY = "Comfy-Org/Qwen-Image_ComfyUI"
_PROVIDER_REVISION = "46839d338df81ce625d5fae27d7e370314c0fbc9"
_LICENSE_REPOSITORY = "Qwen/Qwen-Image"
_LICENSE_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
_LICENSE_SHA256 = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
_PROVIDER_PATHS = {
    "qwen-image-dit": "split_files/diffusion_models/qwen_image_bf16.safetensors",
    "qwen2.5-vl-7b-text": ("split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    "wan21-vae": "split_files/vae/qwen_image_vae.safetensors",
}
_DIT_HEADER_CONTRACT = {
    "dtype_counts": {"BF16": 1933},
    "public_contract": "dinkster_inference.qwen_image_layout.qwen_image_dit_layout",
    "tensor_count": 1933,
}
_TEXT_HEADER_CONTRACT = {
    "dtype_counts": {"BF16": 371, "F32": 716, "F8_E4M3": 359},
    "logical_contract": "dinkster_inference.qwen_image_text.qwen_image_text_layout",
    "logical_contract_tensors": 728,
    "provider_extras": ["lm_head.weight", "scaled_fp8"],
    "tensor_count": 1446,
}
_VAE_REQUIRED_TENSORS = {
    "conv1.weight": [32, 32, 1, 1, 1],
    "conv2.weight": [16, 16, 1, 1, 1],
    "decoder.conv1.weight": [384, 16, 3, 3, 3],
    "decoder.head.2.weight": [3, 96, 3, 3, 3],
    "decoder.middle.1.proj.weight": [384, 384, 1, 1],
    "decoder.upsamples.11.resample.1.weight": [96, 192, 3, 3],
    "decoder.upsamples.3.time_conv.weight": [768, 384, 3, 1, 1],
    "encoder.conv1.weight": [96, 3, 3, 3, 3],
    "encoder.downsamples.5.resample.1.weight": [192, 192, 3, 3],
    "encoder.downsamples.5.time_conv.weight": [192, 192, 3, 1, 1],
    "encoder.head.2.weight": [32, 384, 3, 3, 3],
    "encoder.middle.1.to_qkv.weight": [1152, 384, 1, 1],
}
_VAE_HEADER_CONTRACT = {
    "dtype_counts": {"BF16": 194},
    "public_contract": "dinkster_inference_torch.wan21_vae.WanVAEConfig",
    "required_tensors": _VAE_REQUIRED_TENSORS,
    "tensor_count": 194,
}
_HEADER_CONTRACTS = {
    "qwen-image-dit": _DIT_HEADER_CONTRACT,
    "qwen2.5-vl-7b-text": _TEXT_HEADER_CONTRACT,
    "wan21-vae": _VAE_HEADER_CONTRACT,
}
_MANIFEST_FIELDS = {
    "artifact_root",
    "artifacts",
    "license_receipt",
    "provider",
    "schema",
    "storage_receipt",
}
_ARTIFACT_FIELDS = {
    "acquisition",
    "blake3",
    "bytes",
    "header_contract",
    "header_sha256",
    "local_path",
    "provider_path",
    "role",
    "sha256",
    "source_url",
}
_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
_MAX_HEADER_BYTES = 16 * 1024 * 1024


class ArtifactReceiptError(ValueError):
    """An artifact receipt or local artifact does not match its immutable claim."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactReceiptError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    """Load one JSON receipt manifest."""
    try:
        value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ArtifactReceiptError) as error:
        raise ArtifactReceiptError(f"cannot read receipt manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactReceiptError("receipt manifest must be an object")
    return value


def _hex(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactReceiptError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _exact_json(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_json(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on mutable, incomplete, or ambiguous receipt claims."""
    if (
        set(manifest) != _MANIFEST_FIELDS
        or type(manifest.get("schema")) is not int
        or manifest["schema"] != 1
    ):
        raise ArtifactReceiptError("unsupported Qwen Image artifact receipt schema")
    root = manifest.get("artifact_root")
    if not isinstance(root, str) or not PurePosixPath(root).is_absolute():
        raise ArtifactReceiptError("artifact_root must be an absolute path")

    provider = manifest.get("provider")
    expected_provider = {
        "api_url": (
            f"https://huggingface.co/api/models/{_PROVIDER_REPOSITORY}/"
            f"revision/{_PROVIDER_REVISION}"
        ),
        "gated": False,
        "license": "Apache-2.0",
        "model_card_url": (
            f"https://huggingface.co/{_PROVIDER_REPOSITORY}/raw/{_PROVIDER_REVISION}/README.md"
        ),
        "repository": _PROVIDER_REPOSITORY,
        "revision": _PROVIDER_REVISION,
    }
    if not _exact_json(provider, expected_provider):
        raise ArtifactReceiptError("provider receipt is not the exact official immutable source")

    license_receipt = manifest.get("license_receipt")
    expected_license = {
        "bytes": 11343,
        "name": "Apache License 2.0",
        "repository": _LICENSE_REPOSITORY,
        "revision": _LICENSE_REVISION,
        "sha256": _LICENSE_SHA256,
        "source_url": (
            f"https://huggingface.co/{_LICENSE_REPOSITORY}/raw/{_LICENSE_REVISION}/LICENSE"
        ),
    }
    if not _exact_json(license_receipt, expected_license):
        raise ArtifactReceiptError("license receipt is not the exact official license text")

    storage = manifest.get("storage_receipt")
    if not isinstance(storage, dict) or set(storage) != {
        "after_free_bytes",
        "before_free_bytes",
        "minimum_free_bytes",
        "required_download_bytes",
        "reused_existing_bytes",
    }:
        raise ArtifactReceiptError("storage_receipt must be an object")
    for field in (
        "before_free_bytes",
        "after_free_bytes",
        "minimum_free_bytes",
        "required_download_bytes",
        "reused_existing_bytes",
    ):
        if type(storage.get(field)) is not int:
            raise ArtifactReceiptError(f"storage {field} must be an integer")
    for field in ("before_free_bytes", "after_free_bytes", "minimum_free_bytes"):
        if type(storage.get(field)) is not int or storage[field] <= 0:
            raise ArtifactReceiptError(f"storage {field} must be a positive integer")
    if (
        type(storage["required_download_bytes"]) is not int
        or storage["required_download_bytes"] != 0
    ):
        raise ArtifactReceiptError("this receipt must reuse all existing artifacts")
    if (
        min(storage["before_free_bytes"], storage["after_free_bytes"])
        < storage["minimum_free_bytes"]
    ):
        raise ArtifactReceiptError("recorded free storage is below the safe margin")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ArtifactReceiptError("exactly three Qwen Image artifacts are required")
    roles: set[str] = set()
    paths: set[str] = set()
    expected_roles = {"qwen-image-dit", "qwen2.5-vl-7b-text", "wan21-vae"}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ArtifactReceiptError("artifact receipt must be an object")
        role = artifact.get("role")
        if not isinstance(role, str) or role not in expected_roles or role in roles:
            raise ArtifactReceiptError("artifact roles must be unique strings")
        roles.add(role)
        local_path = artifact.get("local_path")
        receipt_path = PurePosixPath(local_path) if isinstance(local_path, str) else None
        if (
            not isinstance(local_path, str)
            or receipt_path is None
            or receipt_path.is_absolute()
            or ".." in receipt_path.parts
            or local_path in paths
        ):
            raise ArtifactReceiptError("artifact local paths must be unique contained paths")
        paths.add(local_path)
        if type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0:
            raise ArtifactReceiptError(f"artifact {role} bytes must be positive")
        _hex(artifact.get("sha256"), f"artifact {role} sha256")
        _hex(artifact.get("blake3"), f"artifact {role} blake3")
        _hex(artifact.get("header_sha256"), f"artifact {role} header_sha256")
        if artifact.get("acquisition") != "reused-existing":
            raise ArtifactReceiptError(f"artifact {role} must record reuse without duplication")
        provider_path = _PROVIDER_PATHS[role]
        expected_url = (
            f"https://huggingface.co/{_PROVIDER_REPOSITORY}/resolve/"
            f"{_PROVIDER_REVISION}/{provider_path}"
        )
        if (
            artifact.get("provider_path") != provider_path
            or artifact.get("source_url") != expected_url
        ):
            raise ArtifactReceiptError(f"artifact {role} source is not the exact official path")
        if not _exact_json(artifact.get("header_contract"), _HEADER_CONTRACTS[role]):
            raise ArtifactReceiptError(f"artifact {role} header contract drifted")
    if roles != expected_roles:
        raise ArtifactReceiptError("artifact roles do not form the exact Qwen Image graph")
    if storage["reused_existing_bytes"] != sum(item["bytes"] for item in artifacts):
        raise ArtifactReceiptError("reused artifact bytes do not match the artifact graph")


def _relative_parts(relative: str) -> tuple[str, ...]:
    receipt_path = PurePosixPath(relative)
    if receipt_path.is_absolute() or ".." in receipt_path.parts:
        raise ArtifactReceiptError(f"artifact path is not contained: {relative}")
    parts = receipt_path.parts
    if not parts or any(part in {"", "."} for part in parts):
        raise ArtifactReceiptError(f"artifact path is not contained: {relative}")
    return parts


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _namespace_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _recheck_artifact_namespace(
    root: Path,
    parts: tuple[str, ...],
    directory_identities: list[tuple[int, int]],
    artifact_identity: tuple[int, int],
) -> None:
    recheck_fds: list[int] = []
    artifact_fd: int | None = None
    try:
        current_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        recheck_fds.append(current_fd)
        if _namespace_identity(os.fstat(current_fd)) != directory_identities[0]:
            raise ArtifactReceiptError("artifact root changed during verification")
        for index, component in enumerate(parts[:-1], start=1):
            current_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current_fd,
            )
            recheck_fds.append(current_fd)
            if _namespace_identity(os.fstat(current_fd)) != directory_identities[index]:
                raise ArtifactReceiptError("artifact directory changed during verification")
        artifact_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=current_fd,
        )
        if _namespace_identity(os.fstat(artifact_fd)) != artifact_identity:
            raise ArtifactReceiptError("artifact path changed during verification")
    except ArtifactReceiptError:
        raise
    except OSError as error:
        raise ArtifactReceiptError("artifact namespace changed during verification") from error
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        for directory_fd in reversed(recheck_fds):
            os.close(directory_fd)


@contextmanager
def _open_artifact(root: Path, relative: str, expected_bytes: int) -> Iterator[tuple[int, Path]]:
    """Open one ordinary artifact below root without following any relative symlink."""
    parts = _relative_parts(relative)
    root_mode = root.lstat().st_mode
    if not stat.S_ISDIR(root_mode) or root.is_symlink():
        raise ArtifactReceiptError(f"artifact root is not an ordinary directory: {root}")
    directory_fds: list[int] = []
    directory_identities: list[tuple[int, int]] = []
    artifact_fd: int | None = None
    try:
        current_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        directory_fds.append(current_fd)
        directory_identities.append(_namespace_identity(os.fstat(current_fd)))
        for component in parts[:-1]:
            current_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current_fd,
            )
            directory_fds.append(current_fd)
            directory_identities.append(_namespace_identity(os.fstat(current_fd)))
        artifact_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=current_fd,
        )
        before = os.fstat(artifact_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactReceiptError(f"artifact is not an ordinary file: {root / relative}")
        if before.st_size != expected_bytes:
            raise ArtifactReceiptError(
                f"artifact size mismatch for {root / relative}: "
                f"{before.st_size} != {expected_bytes}"
            )
        yield artifact_fd, root / relative
        after = os.fstat(artifact_fd)
        if _stable_identity(after) != _stable_identity(before):
            raise ArtifactReceiptError(f"artifact changed during verification: {root / relative}")
        _recheck_artifact_namespace(
            root,
            parts,
            directory_identities,
            _namespace_identity(before),
        )
    except OSError as error:
        raise ArtifactReceiptError(
            f"artifact is not an ordinary file contained by root: {root / relative}"
        ) from error
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _parse_safetensors_header(encoded: bytes, payload_bytes: int) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_json_object)
    except ArtifactReceiptError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactReceiptError(f"invalid safetensors header: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactReceiptError("safetensors header must be an object")
    if "__metadata__" in value:
        raise ArtifactReceiptError("safetensors metadata is not permitted")
    ranges: list[tuple[int, int, str]] = []
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ArtifactReceiptError("safetensors tensor entries must be named objects")
        if set(entry) != {"data_offsets", "dtype", "shape"}:
            raise ArtifactReceiptError(f"malformed safetensors entry: {key}")
        offsets = entry["data_offsets"]
        shape = entry["shape"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_bytes
            or not isinstance(shape, list)
            or any(type(item) is not int or item < 0 for item in shape)
            or entry["dtype"] not in _DTYPE_BYTES
        ):
            raise ArtifactReceiptError(f"malformed safetensors geometry: {key}")
        elements = math.prod(shape)
        if offsets[1] - offsets[0] != elements * _DTYPE_BYTES[entry["dtype"]]:
            raise ArtifactReceiptError(f"safetensors byte span mismatch: {key}")
        ranges.append((offsets[0], offsets[1], key))
    previous_end = 0
    for start, end, key in sorted(ranges):
        if start != previous_end:
            raise ArtifactReceiptError(f"safetensors payload has a gap or overlap at {key}")
        previous_end = end
    if previous_end != payload_bytes:
        raise ArtifactReceiptError("safetensors payload has unclaimed trailing bytes")
    return value


def _read_safetensors_header_fd(fd: int, file_size: int) -> dict[str, dict[str, Any]]:
    try:
        length_bytes = os.pread(fd, 8, 0)
        if len(length_bytes) != 8:
            raise ArtifactReceiptError("truncated safetensors prefix")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length == 0 or header_length > _MAX_HEADER_BYTES or header_length > file_size - 8:
            raise ArtifactReceiptError("truncated safetensors header")
        encoded = os.pread(fd, header_length, 8)
        if len(encoded) != header_length:
            raise ArtifactReceiptError("truncated safetensors header")
    except OSError as error:
        raise ArtifactReceiptError(f"cannot read safetensors header: {error}") from error
    return _parse_safetensors_header(encoded, file_size - 8 - header_length)


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read and structurally validate an ordinary safetensors file header."""
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        stat_result = os.fstat(fd)
        if not stat.S_ISREG(stat_result.st_mode):
            raise ArtifactReceiptError(f"safetensors path is not an ordinary file: {path}")
        return _read_safetensors_header_fd(fd, stat_result.st_size)
    except ArtifactReceiptError:
        raise
    except OSError as error:
        raise ArtifactReceiptError(f"cannot open safetensors file {path}: {error}") from error
    finally:
        if fd is not None:
            os.close(fd)


def canonical_header_digest(header: dict[str, dict[str, Any]]) -> str:
    """Digest the sorted tensor key, dtype, and shape role surface."""
    canonical = [(key, header[key]["dtype"], header[key]["shape"]) for key in sorted(header)]
    encoded = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_tensor(
    header: dict[str, dict[str, Any]], key: str, dtype: str, shape: tuple[int, ...]
) -> None:
    entry = header.get(key)
    if entry is None:
        raise ArtifactReceiptError(f"missing required tensor {key}")
    if entry.get("dtype") != dtype or entry.get("shape") != list(shape):
        raise ArtifactReceiptError(
            f"header role mismatch for {key}: expected {dtype} {shape}, "
            f"found {entry.get('dtype')} {entry.get('shape')}"
        )


def validate_dit_header(header: dict[str, dict[str, Any]]) -> None:
    """Require the exact public S2 base DiT role in BF16 storage."""
    layout = qwen_image_dit_layout().keys
    if set(header) != set(layout):
        raise ArtifactReceiptError("DiT header keys do not exactly match the public S2 layout")
    for key, shape in layout.items():
        _require_tensor(header, key, "BF16", shape)


def validate_text_header(header: dict[str, dict[str, Any]]) -> None:
    """Map the scaled-FP8 package exactly onto the public S4A text role."""
    layout = qwen_image_text_layout()
    extras = {"lm_head.weight", "scaled_fp8"}
    logical = {key for key in header if not key.endswith((".scale_input", ".scale_weight"))}
    if logical != set(layout) | extras:
        raise ArtifactReceiptError("text header logical keys do not match the public S4A role")
    expected_scales: set[str] = set()
    for key, shape in layout.items():
        entry = header[key]
        dtype = entry.get("dtype")
        if dtype not in {"BF16", "F8_E4M3"} or entry.get("shape") != list(shape):
            raise ArtifactReceiptError(f"text header role mismatch for {key}")
        if dtype == "F8_E4M3":
            stem = key.removesuffix(".weight")
            expected_scales.update({f"{stem}.scale_input", f"{stem}.scale_weight"})
    actual_scales = set(header) - logical
    if actual_scales != expected_scales:
        raise ArtifactReceiptError("text scaled-FP8 scale companions are incomplete or foreign")
    for key in expected_scales:
        _require_tensor(header, key, "F32", ())
    _require_tensor(header, "lm_head.weight", "BF16", (152064, 3584))
    _require_tensor(header, "scaled_fp8", "F8_E4M3", (0,))
    dtype_counts: dict[str, int] = {}
    for entry in header.values():
        dtype = entry["dtype"]
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if (
        len(header) != _TEXT_HEADER_CONTRACT["tensor_count"]
        or dtype_counts != _TEXT_HEADER_CONTRACT["dtype_counts"]
    ):
        raise ArtifactReceiptError("text header storage does not match the official package")


def validate_vae_header(header: dict[str, dict[str, Any]]) -> None:
    """Require the public S5A Wan21 role anchors and exact provider surface."""
    if len(header) != _VAE_HEADER_CONTRACT["tensor_count"]:
        raise ArtifactReceiptError("VAE tensor count does not match the public S5A role")
    dtype_counts: dict[str, int] = {}
    for entry in header.values():
        dtype = entry["dtype"]
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if dtype_counts != _VAE_HEADER_CONTRACT["dtype_counts"]:
        raise ArtifactReceiptError("VAE storage does not match the public S5A role")
    for key, shape in _VAE_HEADER_CONTRACT["required_tensors"].items():
        _require_tensor(header, key, "BF16", tuple(shape))


def _digests_fd(fd: int) -> tuple[str, str]:
    from blake3 import blake3

    sha256 = hashlib.sha256()
    blake = blake3()
    offset = 0
    while chunk := os.pread(fd, 16 * 1024 * 1024, offset):
        sha256.update(chunk)
        blake.update(chunk)
        offset += len(chunk)
    return sha256.hexdigest(), blake.hexdigest()


def verify_receipts(manifest_path: Path, artifact_root: Path) -> dict[str, Any]:
    """Verify local paths, bytes, digests, and header roles without model imports."""
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    expected_root = Path(manifest["artifact_root"])
    actual_root = artifact_root.absolute()
    if actual_root != expected_root:
        raise ArtifactReceiptError(f"artifact root mismatch: {actual_root} != {expected_root}")
    try:
        if not stat.S_ISDIR(actual_root.lstat().st_mode):
            raise ArtifactReceiptError("artifact root is not an ordinary directory")
    except OSError as error:
        raise ArtifactReceiptError(f"cannot inspect artifact root: {error}") from error
    free_bytes = os.statvfs(actual_root).f_bavail * os.statvfs(actual_root).f_frsize
    if free_bytes < manifest["storage_receipt"]["minimum_free_bytes"]:
        raise ArtifactReceiptError("current free storage is below the safe margin")

    verified: list[dict[str, Any]] = []
    validators = {
        "qwen-image-dit": validate_dit_header,
        "qwen2.5-vl-7b-text": validate_text_header,
        "wan21-vae": validate_vae_header,
    }
    for artifact in manifest["artifacts"]:
        with _open_artifact(actual_root, artifact["local_path"], artifact["bytes"]) as (fd, path):
            initial_identity = _stable_identity(os.fstat(fd))
            sha256, blake = _digests_fd(fd)
            if sha256 != artifact["sha256"]:
                raise ArtifactReceiptError(f"SHA256 mismatch for {artifact['role']}")
            if blake != artifact["blake3"]:
                raise ArtifactReceiptError(f"BLAKE3 mismatch for {artifact['role']}")
            header = _read_safetensors_header_fd(fd, artifact["bytes"])
            if _stable_identity(os.fstat(fd)) != initial_identity:
                raise ArtifactReceiptError(f"artifact changed while reading: {artifact['role']}")
            if canonical_header_digest(header) != artifact["header_sha256"]:
                raise ArtifactReceiptError(f"header digest mismatch for {artifact['role']}")
            validators[artifact["role"]](header)
        verified.append(
            {
                "blake3": blake,
                "bytes": artifact["bytes"],
                "header_sha256": artifact["header_sha256"],
                "path": str(path),
                "role": artifact["role"],
                "sha256": sha256,
            }
        )
    return {
        "artifact_root": str(actual_root),
        "free_bytes": free_bytes,
        "provider_revision": manifest["provider"]["revision"],
        "verified": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("qwen_image_artifacts.json"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_receipts(args.manifest, args.artifact_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
