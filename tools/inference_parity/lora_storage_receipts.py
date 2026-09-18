from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO

from blake3 import blake3


class ReceiptError(ValueError):
    pass


@dataclass(frozen=True)
class FileReceipt:
    size: int
    sha256: str
    blake3: str


@dataclass(frozen=True)
class SafetensorsHeader:
    raw_sha256: str
    length: int
    tensors: Mapping[str, Mapping[str, object]]
    metadata: Mapping[str, str]


EXPECTED_ROOT = Path("/home/kosin/ComfyUI-Shared/models")
HEX_DIGEST = frozenset("0123456789abcdef")
DTYPE_BYTES = MappingProxyType(
    {
        "BOOL": 1,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "U8": 1,
    }
)
EXPECTED_ARTIFACTS = MappingProxyType(
    {
        "combined-sd15-checkpoint": {
            "repository": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            "file": "v1-5-pruned-emaonly.safetensors",
            "source_url": (
                "https://huggingface.co/stable-diffusion-v1-5/"
                "stable-diffusion-v1-5/resolve/"
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14/"
                "v1-5-pruned-emaonly.safetensors"
            ),
            "local_path": "checkpoints/v1-5-pruned-emaonly.safetensors",
            "size": 4265146304,
            "sha256": "6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa",
            "blake3": "2da04f4a2d5a205c9422c871eca1e93670e843d48752b4d35ab782d002e5dcae",
            "header_length": 156444,
            "header_sha256": "ed6e1a1f33ba3a02193e599f6441e213f973f2949dedcfafefd570da536eae9e",
            "tensor_count": 1145,
            "license": "creativeml-openrail-m",
        },
        "sd15-unet-lcm-lora": {
            "repository": "latent-consistency/lcm-lora-sdv1-5",
            "revision": "cf2fced511dbe7e26c8d1d397e728fbab875db4b",
            "file": "pytorch_lora_weights.safetensors",
            "source_url": (
                "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/"
                "cf2fced511dbe7e26c8d1d397e728fbab875db4b/"
                "pytorch_lora_weights.safetensors"
            ),
            "local_path": (
                "loras/latent-consistency_lcm-lora-sdv1-5_cf2fced5_pytorch_lora_weights.safetensors"
            ),
            "size": 134621556,
            "sha256": "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4",
            "blake3": "d07d31701c4d354c6910228dd0fb785869aff4a4c70e1d677ee459859f36a6fb",
            "header_length": 116544,
            "header_sha256": "3ca15dde66470e73c8d68dbb9ab64d6e7ad014f525b58193a9a3b161eb6cc263",
            "tensor_count": 834,
            "license": "openrail++",
        },
    }
)
EXPECTED_GRAPHS = MappingProxyType(
    {
        "no_lora": {
            "artifact_roles": ["combined-sd15-checkpoint"],
            "nodes": {
                "checkpoint": "dinkster.load_checkpoint",
                "positive": "dinkster.clip_text_encode",
                "negative": "dinkster.clip_text_encode",
                "latent": "dinkster.empty_latent_image",
                "sampler": "dinkster.ksampler",
                "decode": "dinkster.vae_decode",
            },
            "edges": [
                ["checkpoint.model", "sampler.model"],
                ["checkpoint.clip", "positive.clip"],
                ["checkpoint.clip", "negative.clip"],
                ["checkpoint.vae", "decode.vae"],
                ["positive.conditioning", "sampler.positive"],
                ["negative.conditioning", "sampler.negative"],
                ["latent.latent", "sampler.latent_image"],
                ["sampler.latent", "decode.samples"],
            ],
            "inputs": {
                "checkpoint.checkpoint": "combined-sd15-checkpoint",
                "positive.text": (
                    "beautiful scenery nature glass bottle landscape, purple galaxy bottle,"
                ),
                "negative.text": "text, watermark",
                "latent.width": 512,
                "latent.height": 512,
                "latent.batch_size": 1,
                "sampler.seed": 685468484323813,
                "sampler.steps": 4,
                "sampler.cfg": 1.0,
                "sampler.sampler_name": "lcm",
                "sampler.scheduler": "normal",
                "sampler.denoise": 1.0,
            },
            "output": "decode.image",
        },
        "with_lora": {
            "artifact_roles": ["combined-sd15-checkpoint", "sd15-unet-lcm-lora"],
            "nodes": {
                "checkpoint": "dinkster.load_checkpoint",
                "lora": "dinkster.load_lora_model_only",
                "positive": "dinkster.clip_text_encode",
                "negative": "dinkster.clip_text_encode",
                "latent": "dinkster.empty_latent_image",
                "sampler": "dinkster.ksampler",
                "decode": "dinkster.vae_decode",
            },
            "edges": [
                ["checkpoint.model", "lora.model"],
                ["lora.model", "sampler.model"],
                ["checkpoint.clip", "positive.clip"],
                ["checkpoint.clip", "negative.clip"],
                ["checkpoint.vae", "decode.vae"],
                ["positive.conditioning", "sampler.positive"],
                ["negative.conditioning", "sampler.negative"],
                ["latent.latent", "sampler.latent_image"],
                ["sampler.latent", "decode.samples"],
            ],
            "inputs": {
                "checkpoint.checkpoint": "combined-sd15-checkpoint",
                "lora.lora": "sd15-unet-lcm-lora",
                "lora.strength_model": 1.0,
                "positive.text": (
                    "beautiful scenery nature glass bottle landscape, purple galaxy bottle,"
                ),
                "negative.text": "text, watermark",
                "latent.width": 512,
                "latent.height": 512,
                "latent.batch_size": 1,
                "sampler.seed": 685468484323813,
                "sampler.steps": 4,
                "sampler.cfg": 1.0,
                "sampler.sampler_name": "lcm",
                "sampler.scheduler": "normal",
                "sampler.denoise": 1.0,
            },
            "output": "decode.image",
        },
    }
)
EXPECTED_OPERATION_ORDER = (
    "load owned fp32 base storage",
    "decode immutable ordered PatchSet",
    "apply every patch entry in declaration order in fp32",
    "cast exactly once to authoritative fp16 storage",
    "enroll converted storage with no deferred PatchSet",
)


def blake3_hex(value: bytes) -> str:
    return blake3(value).hexdigest()


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ReceiptError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _exact_json(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return actual.keys() == expected.keys() and all(
            _exact_json(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return len(actual) == len(expected) and all(
            _exact_json(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _closed(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must be an object")
    if set(value) != keys:
        raise ReceiptError(f"{label} fields must be exactly {sorted(keys)}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptError(f"{label} must be a non-negative integer")
    return value


def _digest(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in HEX_DIGEST for character in text):
        raise ReceiptError(f"{label} must be lowercase 64-hex")
    return text


def _relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReceiptError(f"{label} must be a contained normalized relative path")
    return text


def load_packet(path: Path) -> dict[str, Any]:
    try:
        packet = json.loads(path.read_text("utf-8"), object_pairs_hook=_object_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"packet cannot be read: {exc}") from exc
    validate_packet(packet)
    return packet


def validate_packet(value: object) -> None:
    packet = _closed(
        value,
        {
            "schema",
            "commission",
            "artifact_root",
            "artifacts",
            "licenses",
            "storage_receipt",
            "graphs",
            "storage_oracle",
            "physical_execution",
        },
        "packet",
    )
    if _integer(packet["schema"], "schema") != 1:
        raise ReceiptError("schema must be 1")
    if packet["commission"] != "W0-LORA-STORAGE-A1-ARTIFACT-PACKET":
        raise ReceiptError("commission is not authoritative")
    if packet["artifact_root"] != str(EXPECTED_ROOT):
        raise ReceiptError("artifact_root is not authoritative")
    artifacts = packet["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ReceiptError("artifacts must contain exactly two entries")
    roles: list[str] = []
    for index, artifact_value in enumerate(artifacts):
        artifact = _closed(
            artifact_value,
            {"role", *next(iter(EXPECTED_ARTIFACTS.values())).keys()},
            f"artifact {index}",
        )
        role = _string(artifact["role"], f"artifact {index} role")
        if role in roles or role not in EXPECTED_ARTIFACTS:
            raise ReceiptError(f"artifact role is duplicate or unknown: {role}")
        roles.append(role)
        expected = EXPECTED_ARTIFACTS[role]
        for key, expected_value in expected.items():
            value = artifact[key]
            if key in {"size", "header_length", "tensor_count"}:
                _integer(value, f"{role} {key}")
            elif key in {"sha256", "blake3", "header_sha256"}:
                _digest(value, f"{role} {key}")
            elif key == "local_path":
                _relative_path(value, f"{role} local_path")
            else:
                _string(value, f"{role} {key}")
            if value != expected_value:
                raise ReceiptError(f"{role} {key} is not authoritative")
    if roles != list(EXPECTED_ARTIFACTS):
        raise ReceiptError("artifact role order is not authoritative")
    _validate_licenses(packet["licenses"])
    _validate_storage_receipt(packet["storage_receipt"])
    if not _exact_json(packet["graphs"], dict(EXPECTED_GRAPHS)):
        raise ReceiptError("graph authority is not exact")
    oracle = _closed(
        packet["storage_oracle"],
        {"wiring_epoch", "destination_dtype", "operation_order", "proofs"},
        "storage_oracle",
    )
    if type(oracle["wiring_epoch"]) is not int or oracle["wiring_epoch"] != 17:
        raise ReceiptError("storage oracle identity is not exact")
    if oracle["destination_dtype"] != "fp16":
        raise ReceiptError("storage oracle identity is not exact")
    if not _exact_json(oracle["operation_order"], list(EXPECTED_OPERATION_ORDER)):
        raise ReceiptError("storage oracle operation order is not exact")
    proofs = oracle["proofs"]
    if (
        not isinstance(proofs, list)
        or not proofs
        or not all(isinstance(item, str) for item in proofs)
    ):
        raise ReceiptError("storage oracle proofs must be non-empty strings")
    if packet["physical_execution"] != "blocked_pending_independent_packet_review":
        raise ReceiptError("physical execution hold is not exact")


def _validate_licenses(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ReceiptError("licenses must contain exactly two entries")
    expected = (
        (
            "creativeml-openrail-m",
            "CreativeML Open RAIL-M dated August 22, 2022",
            "https://huggingface.co/spaces/CompVis/stable-diffusion-license/resolve/"
            "14d42d09bffd871b1666a084fc954a50cff72ac0/license.txt",
            14385,
            "be351ebe7ac01bcdbb018639aadcfd38f136b7dc3f2a3d4d3a24db51d1b210ef",
        ),
        (
            "openrail++",
            "CreativeML Open RAIL++-M License dated July 26, 2023",
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/"
            "462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md",
            14109,
            "19b6998b569b53ac1fc2158a8a3202c8699a9a4605b47075715d9c96be7fb6d0",
        ),
    )
    for index, (entry_value, expected_values) in enumerate(zip(value, expected, strict=True)):
        entry = _closed(
            entry_value,
            {
                "identifier",
                "text_name",
                "text_url",
                "text_size",
                "text_sha256",
                "declaration_url",
                "accepted",
                "terms",
            },
            f"license {index}",
        )
        actual = (
            entry["identifier"],
            entry["text_name"],
            entry["text_url"],
            entry["text_size"],
            entry["text_sha256"],
        )
        if actual != expected_values:
            raise ReceiptError(f"license {index} text authority is not exact")
        _integer(entry["text_size"], f"license {index} text_size")
        _digest(entry["text_sha256"], f"license {index} text_sha256")
        declaration = _string(entry["declaration_url"], f"license {index} declaration_url")
        if "/resolve/" not in declaration:
            raise ReceiptError(f"license {index} declaration is mutable")
        if entry["accepted"] is not True:
            raise ReceiptError(f"license {index} is not accepted")
        terms = entry["terms"]
        if (
            not isinstance(terms, list)
            or not terms
            or not all(isinstance(item, str) for item in terms)
        ):
            raise ReceiptError(f"license {index} terms are not exact strings")


def _validate_storage_receipt(value: object) -> None:
    receipt = _closed(
        value,
        {
            "free_before_bytes",
            "free_after_bytes",
            "base_action",
            "lora_action",
            "official_lora_size_matches_before",
            "rejected_lookalike",
            "redistribution",
        },
        "storage_receipt",
    )
    before = _integer(receipt["free_before_bytes"], "free_before_bytes")
    after = _integer(receipt["free_after_bytes"], "free_after_bytes")
    if before < after or before - after < EXPECTED_ARTIFACTS["sd15-unet-lcm-lora"]["size"]:
        raise ReceiptError("free-space receipt does not cover the acquired LoRA")
    if receipt["base_action"] != "reused_exact_existing_file":
        raise ReceiptError("base acquisition receipt is not exact")
    if receipt["lora_action"] != "acquired_missing_official_file_once":
        raise ReceiptError("LoRA acquisition receipt is not exact")
    if receipt["official_lora_size_matches_before"] != 0:
        raise ReceiptError("duplicate search receipt is not exact")
    if receipt["redistribution"] != "none":
        raise ReceiptError("redistribution receipt is not exact")
    rejected = _closed(
        receipt["rejected_lookalike"], {"path", "size", "sha256"}, "rejected_lookalike"
    )
    if rejected != {
        "path": "/home/kosin/ComfyUI-Shared/models/loras/lcm-lora-sdv1-5.safetensors",
        "size": 2132625432,
        "sha256": "463d6a9fe8a4b56a4d69ef3692074c0617428dfd8e8f12f9efe3b1e9a71717ce",
    }:
        raise ReceiptError("rejected lookalike receipt is not exact")


def read_safetensors_header(handle: BinaryIO, file_size: int) -> SafetensorsHeader:
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ReceiptError("safetensors header prefix is truncated")
    length = struct.unpack("<Q", prefix)[0]
    if length == 0 or length > file_size - 8:
        raise ReceiptError("safetensors header length is invalid")
    raw = handle.read(length)
    if len(raw) != length:
        raise ReceiptError("safetensors header is truncated")
    try:
        value = json.loads(raw, object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"safetensors header is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptError("safetensors header must be an object")
    metadata_value = value.pop("__metadata__", {})
    if not isinstance(metadata_value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in metadata_value.items()
    ):
        raise ReceiptError("safetensors metadata must map strings to strings")
    tensors: dict[str, Mapping[str, object]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor_value in value.items():
        descriptor = _closed(descriptor_value, {"dtype", "shape", "data_offsets"}, name)
        dtype = _string(descriptor["dtype"], f"{name} dtype")
        if dtype not in DTYPE_BYTES:
            raise ReceiptError(f"{name} dtype is unsupported")
        shape = descriptor["shape"]
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
        ):
            raise ReceiptError(f"{name} shape is invalid")
        offsets = descriptor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ReceiptError(f"{name} data_offsets are invalid")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if offsets[1] - offsets[0] != elements * DTYPE_BYTES[dtype]:
            raise ReceiptError(f"{name} byte span does not match dtype and shape")
        ranges.append((offsets[0], offsets[1], name))
        tensors[name] = MappingProxyType(dict(descriptor))
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ReceiptError(f"safetensors payload is not contiguous before {name}")
        cursor = end
    if cursor != file_size - 8 - length:
        raise ReceiptError("safetensors payload does not cover the file exactly")
    return SafetensorsHeader(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        length=length,
        tensors=MappingProxyType(tensors),
        metadata=MappingProxyType(dict(metadata_value)),
    )


def validate_role(role: str, tensors: Mapping[str, Mapping[str, object]]) -> None:
    keys = tuple(tensors)
    if role == "combined-sd15-checkpoint":
        required = ("model.diffusion_model.", "cond_stage_model.", "first_stage_model.")
        if any(not any(key.startswith(prefix) for key in keys) for prefix in required):
            raise ReceiptError("combined SD1.5 checkpoint is missing a required component role")
        if any(
            descriptor.get("dtype") not in {"F32", "I32", "I64"} for descriptor in tensors.values()
        ):
            raise ReceiptError("combined SD1.5 checkpoint storage dtype is foreign")
        return
    if role != "sd15-unet-lcm-lora":
        raise ReceiptError(f"unknown artifact role: {role}")
    if any(key.startswith("lora_te") or key.startswith("lora_text") for key in keys):
        raise ReceiptError("LCM LoRA must be UNet-only")
    if not keys or any(not key.startswith("lora_unet_") for key in keys):
        raise ReceiptError("LCM LoRA contains a foreign role")
    if any(descriptor.get("dtype") != "F16" for descriptor in tensors.values()):
        raise ReceiptError("LCM LoRA must contain only F16 tensors")
    groups: dict[str, set[str]] = {}
    suffixes = (".alpha", ".lora_down.weight", ".lora_up.weight")
    for key in keys:
        suffix = next((candidate for candidate in suffixes if key.endswith(candidate)), None)
        if suffix is None:
            raise ReceiptError("LCM LoRA contains a foreign tensor suffix")
        groups.setdefault(key[: -len(suffix)], set()).add(suffix)
    required_suffixes = set(suffixes)
    if any(value != required_suffixes for value in groups.values()):
        raise ReceiptError("LCM LoRA tensor triplet is incomplete")


def _namespace_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _recheck_artifact_namespace(
    artifact_root: Path,
    parts: tuple[str, ...],
    directory_identities: list[tuple[int, int]],
    artifact_identity: tuple[int, int],
) -> None:
    directory_descriptors: list[int] = []
    artifact_descriptor: int | None = None
    try:
        current = os.open(
            artifact_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        directory_descriptors.append(current)
        if _namespace_identity(os.fstat(current)) != directory_identities[0]:
            raise ReceiptError("artifact root changed during verification")
        for index, part in enumerate(parts[:-1], start=1):
            current = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            directory_descriptors.append(current)
            if _namespace_identity(os.fstat(current)) != directory_identities[index]:
                raise ReceiptError("artifact directory changed during verification")
        artifact_descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current
        )
        if _namespace_identity(os.fstat(artifact_descriptor)) != artifact_identity:
            raise ReceiptError("artifact path changed during verification")
    except ReceiptError:
        raise
    except OSError as exc:
        raise ReceiptError("artifact namespace changed during verification") from exc
    finally:
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


@contextmanager
def _open_artifact(artifact_root: Path, relative: str, expected_size: int) -> Iterator[int]:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts or "." in parts:
        raise ReceiptError(f"artifact path is not contained: {relative}")
    root_mode = artifact_root.lstat().st_mode
    if artifact_root.is_symlink() or not stat.S_ISDIR(root_mode):
        raise ReceiptError(f"artifact root is not an ordinary directory: {artifact_root}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory_descriptors: list[int] = []
    directory_identities: list[tuple[int, int]] = []
    artifact_descriptor: int | None = None
    try:
        current = os.open(artifact_root, directory_flags)
        directory_descriptors.append(current)
        directory_identities.append(_namespace_identity(os.fstat(current)))
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            directory_descriptors.append(current)
            directory_identities.append(_namespace_identity(os.fstat(current)))
        artifact_descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        before = os.fstat(artifact_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReceiptError(f"artifact is not a regular file: {artifact_root / relative}")
        if before.st_size != expected_size:
            raise ReceiptError(f"artifact size mismatch: {artifact_root / relative}")
        yield artifact_descriptor
        after = os.fstat(artifact_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise ReceiptError(f"artifact changed while verifying: {artifact_root / relative}")
        _recheck_artifact_namespace(
            artifact_root,
            parts,
            directory_identities,
            _namespace_identity(before),
        )
    except ReceiptError:
        raise
    except OSError as exc:
        raise ReceiptError(
            f"artifact path cannot be opened without symlinks: {relative}: {exc}"
        ) from exc
    finally:
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _verify_artifact_descriptor(
    descriptor: int, path: Path, expected: FileReceipt
) -> SafetensorsHeader:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ReceiptError(f"artifact is not a regular file: {path}")
    if before.st_size != expected.size:
        raise ReceiptError(f"artifact size mismatch: {path}")
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        header = read_safetensors_header(handle, expected.size)
        handle.seek(0)
        sha = hashlib.sha256()
        b3 = blake3()
        while chunk := handle.read(8 * 1024 * 1024):
            sha.update(chunk)
            b3.update(chunk)
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    ):
        raise ReceiptError(f"artifact changed while verifying: {path}")
    if sha.hexdigest() != expected.sha256:
        raise ReceiptError(f"artifact sha256 mismatch: {path}")
    if b3.hexdigest() != expected.blake3:
        raise ReceiptError(f"artifact blake3 mismatch: {path}")
    return header


def verify_packet(packet_path: Path, artifact_root: Path) -> dict[str, object]:
    packet = load_packet(packet_path)
    if artifact_root != EXPECTED_ROOT or artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ReceiptError("artifact root must be the exact ordinary commissioned directory")
    verified: list[dict[str, object]] = []
    for artifact_value in packet["artifacts"]:  # type: ignore[union-attr]
        artifact = artifact_value
        role = artifact["role"]
        path = artifact_root / artifact["local_path"]
        expected = FileReceipt(artifact["size"], artifact["sha256"], artifact["blake3"])
        with _open_artifact(artifact_root, artifact["local_path"], expected.size) as descriptor:
            header = _verify_artifact_descriptor(descriptor, path, expected)
        if header.length != artifact["header_length"]:
            raise ReceiptError(f"{role} header length mismatch")
        if header.raw_sha256 != artifact["header_sha256"]:
            raise ReceiptError(f"{role} header sha256 mismatch")
        if len(header.tensors) != artifact["tensor_count"]:
            raise ReceiptError(f"{role} tensor count mismatch")
        if dict(header.metadata) != {"format": "pt"}:
            raise ReceiptError(f"{role} metadata mismatch")
        validate_role(role, header.tensors)
        verified.append(
            {
                "role": role,
                "path": str(path),
                "size": expected.size,
                "sha256": expected.sha256,
                "blake3": expected.blake3,
                "header_sha256": header.raw_sha256,
                "tensor_count": len(header.tensors),
            }
        )
    return {
        "packet": str(packet_path),
        "artifact_root": str(artifact_root),
        "artifacts": verified,
        "physical_execution": packet["physical_execution"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the W0 SD1.5 LoRA artifact packet")
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path(__file__).with_name("lora_storage_artifacts.json"),
    )
    parser.add_argument("--artifact-root", type=Path, default=EXPECTED_ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_packet(args.packet, args.artifact_root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
