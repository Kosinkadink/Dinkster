"""Convert legacy torch checkpoints at the compat boundary.

Dinkster's core inference packages remain safetensors-only. This module mirrors
the safe-loading intent of ComfyUI ``comfy/utils.py:110-155`` at pinned commit
``f4b99bc62389af315013dda85f24f2bbd262b686``: its torch load is weights-only
at line 143 and its ``state_dict`` extraction is at lines 145-146. Dinkster
deliberately diverges by refusing unsafe fallback instead of retrying pickle
loading, and by converting once to a safetensors sidecar instead of retaining
an in-memory pickle-loaded state dict.

This module is the canonical conversion implementation consumed by both the
compat serving path and ``tools/convert_ckpt_to_safetensors.py``. Imports stay
torch-free until conversion actually runs.

Classification is deliberately asymmetric. A clean safetensors header passes
through by content regardless of the physical filename, preserving
extensionless content-addressed vault assets. Conversion remains
extension-gated: only .ckpt, .pt, and .pth paths may reach ``torch.load``, so
arbitrary malformed or extensionless bytes are never treated as pickle.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from dinkster_inference import load_safetensors_header
from dinkster_values import MEBIBYTE

__all__ = [
    "LegacyCheckpointError",
    "LegacyConversionResult",
    "classify_conversion_error",
    "classify_weight_source",
    "convert_legacy_checkpoint",
    "discover_converted_sidecar",
    "resolve_weight_source",
]

log = logging.getLogger("dinkster.compat_comfy.legacy_sources")
_SAFE_EXTENSIONS = frozenset({".safetensors", ".sft"})
_LEGACY_EXTENSIONS = frozenset({".ckpt", ".pt", ".pth"})
_SUPPORTED_EXTENSIONS = tuple(sorted(_SAFE_EXTENSIONS | _LEGACY_EXTENSIONS))
_RETRYABLE_ERROR_PREFIXES = (
    "legacy-checkpoint-converter-unavailable",
    "legacy-checkpoint-sidecar-directory-",
    "legacy-checkpoint-sidecar-write-failed",
    "legacy-checkpoint-sidecar-cleanup-failed",
    "legacy-checkpoint-source-read-failed",
    "legacy-checkpoint-source-reverification-failed",
    "legacy-checkpoint-source-changed",
)


class LegacyCheckpointError(ValueError):
    """A legacy source cannot be converted safely at the compat boundary."""


@dataclass(frozen=True)
class LegacyConversionResult:
    """Stable conversion facts used by the CLI and sidecar diagnostics."""

    dropped_keys: tuple[str, ...]
    output_bytes: int
    output_sha256: str


def classify_conversion_error(
    error: BaseException,
) -> Literal["retryable", "refused"]:
    """Classify canonical converter failures for worker retry discipline."""
    if not isinstance(error, LegacyCheckpointError):
        return "retryable"
    if str(error).startswith(_RETRYABLE_ERROR_PREFIXES):
        return "retryable"
    return "refused"


def classify_weight_source(path: Path, logical_name: str | None = None) -> str:
    """Classify one weight source without importing torch or trusting its path."""
    try:
        load_safetensors_header(path)
    except (OSError, ValueError):
        name = logical_name or ""
        if Path(name).suffix.lower() in _LEGACY_EXTENSIONS:
            return "legacy-convertible"
        return "unsupported"
    return "safetensors"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MEBIBYTE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_file(tensors: Mapping[str, Any], path: Path) -> None:
    safetensors_torch = importlib.import_module("safetensors.torch")
    safetensors_torch.save_file(tensors, path, metadata=None)


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _load_torch_module() -> Any:
    return importlib.import_module("torch")


def convert_legacy_checkpoint(source: Path, output: Path) -> LegacyConversionResult:
    """Safely convert one torch-pickle checkpoint to deterministic safetensors."""
    try:
        torch = _load_torch_module()
    except Exception as exc:  # noqa: BLE001 - normalize broken native installs
        raise LegacyCheckpointError(
            "legacy-checkpoint-converter-unavailable: torch is not installed in "
            f"the compat environment needed to convert {source}"
        ) from exc
    try:
        loaded = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - normalize safe-loader refusals
        raise LegacyCheckpointError(f"legacy-checkpoint-safe-load-failed: {source}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise LegacyCheckpointError(
            "legacy-checkpoint-root-not-mapping: "
            f"{source}: expected a mapping, got {type(loaded).__name__}"
        )

    dropped: list[str] = []
    root = cast("Mapping[object, object]", loaded)
    for key in root:
        if not isinstance(key, str):
            raise LegacyCheckpointError(
                "legacy-checkpoint-key-not-string: "
                f"{source}: expected str, got {type(key).__name__}"
            )
    string_root = cast("Mapping[str, object]", root)
    if "state_dict" in root:
        state_dict = root["state_dict"]
        dropped.extend(key for key in string_root if key != "state_dict")
        if not isinstance(state_dict, Mapping):
            raise LegacyCheckpointError(
                "legacy-checkpoint-state-dict-not-mapping: "
                f"{source}: expected a mapping, got {type(state_dict).__name__}"
            )
        values = cast("Mapping[object, object]", state_dict)
    else:
        values = root

    tensors: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise LegacyCheckpointError(
                "legacy-checkpoint-key-not-string: "
                f"{source}: expected str, got {type(key).__name__}"
            )
        if not isinstance(value, torch.Tensor):
            dropped.append(key)
            continue
        tensor = cast("Any", value)
        try:
            tensors[key] = tensor.detach().cpu().contiguous()
        except Exception as exc:  # noqa: BLE001 - normalize tensor failures
            raise LegacyCheckpointError(
                "legacy-checkpoint-conversion-failed: "
                f"{source}: tensor {key!r} could not be normalized: {exc}"
            ) from exc

    ordered = {key: tensors[key] for key in sorted(tensors)}
    try:
        _save_file(ordered, output)
    except Exception as exc:  # noqa: BLE001 - normalize serializer/IO failures
        raise LegacyCheckpointError(
            f"legacy-checkpoint-conversion-failed: {source} -> {output}: {exc}"
        ) from exc
    try:
        output_bytes = output.stat().st_size
        output_sha256 = _sha256_file(output)
    except OSError as exc:
        raise LegacyCheckpointError(
            f"legacy-checkpoint-conversion-failed: {source} -> {output}: {exc}"
        ) from exc
    return LegacyConversionResult(tuple(dropped), output_bytes, output_sha256)


def _require_writable_directory(source: Path) -> None:
    directory = source.parent
    writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    try:
        mode = directory.stat().st_mode
    except OSError as exc:
        raise LegacyCheckpointError(
            "legacy-checkpoint-sidecar-directory-unavailable: "
            f"{directory}: {exc}; convert manually with "
            "tools/convert_ckpt_to_safetensors.py"
        ) from exc
    if mode & writable_bits and os.access(directory, os.W_OK):
        return
    raise LegacyCheckpointError(
        "legacy-checkpoint-sidecar-directory-not-writable: "
        f"{directory}; convert manually with tools/convert_ckpt_to_safetensors.py"
    )


def _sidecar_path(path: Path, source_digest: str) -> Path:
    return path.with_name(f"{path.stem}.{source_digest[:16]}.dinkster.safetensors")


def discover_converted_sidecar(path: Path) -> Path | None:
    """Return an adjacent valid converted sidecar without hashing the source."""
    prefix = f"{path.stem}."
    suffix = ".dinkster.safetensors"
    for candidate in sorted(path.parent.glob(f"{path.stem}.*{suffix}")):
        identity = candidate.name[len(prefix) : -len(suffix)]
        if len(identity) != 16 or any(
            character not in "0123456789abcdef" for character in identity
        ):
            continue
        try:
            load_safetensors_header(candidate)
        except (OSError, ValueError):
            continue
        return candidate
    return None


def resolve_weight_source(path: Path, logical_name: str | None = None) -> Path:
    """Return a safetensors source, converting a supported legacy file once."""
    header_failure: BaseException | None = None
    try:
        load_safetensors_header(path)
    except (OSError, ValueError) as exc:
        header_failure = exc
    else:
        return path

    suffix = Path(logical_name or path.name).suffix.lower()
    if suffix not in _LEGACY_EXTENSIONS:
        supported = ", ".join(_SUPPORTED_EXTENSIONS)
        raise LegacyCheckpointError(
            f"unsupported-weight-source-format: {path}; supported formats: {supported}; "
            "a legacy pickle checkpoint must use .ckpt, .pt, or .pth, or be "
            "converted manually with tools/convert_ckpt_to_safetensors.py"
        ) from header_failure

    try:
        source_bytes = path.stat().st_size
        source_digest = _sha256_file(path)
    except OSError as exc:
        raise LegacyCheckpointError(f"legacy-checkpoint-source-read-failed: {path}: {exc}") from exc
    sidecar = _sidecar_path(path, source_digest)
    if sidecar.exists():
        try:
            load_safetensors_header(sidecar)
        except (OSError, ValueError):
            pass
        else:
            return sidecar

    _require_writable_directory(path)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        result = convert_legacy_checkpoint(path, temporary)
        try:
            current_digest = _sha256_file(path)
        except OSError as exc:
            raise LegacyCheckpointError(
                f"legacy-checkpoint-source-reverification-failed: {path}: {exc}"
            ) from exc
        if current_digest != source_digest:
            raise LegacyCheckpointError(
                f"legacy-checkpoint-source-changed: {path} changed during conversion"
            )
        try:
            _replace_file(temporary, sidecar)
        except OSError as exc:
            try:
                load_safetensors_header(sidecar)
            except (OSError, ValueError):
                raise exc from None
            _remove_file(temporary)
        temporary = None
    except LegacyCheckpointError:
        raise
    except OSError as exc:
        raise LegacyCheckpointError(
            "legacy-checkpoint-sidecar-write-failed: "
            f"{path} -> {sidecar}: {exc}; convert manually with "
            "tools/convert_ckpt_to_safetensors.py"
        ) from exc
    finally:
        if temporary is not None:
            try:
                _remove_file(temporary)
            except OSError as exc:
                raise LegacyCheckpointError(
                    f"legacy-checkpoint-sidecar-cleanup-failed: {temporary}: {exc}"
                ) from exc

    log.warning(
        "converted legacy checkpoint source=%s sidecar=%s source_sha256=%s "
        "source_bytes=%d output_bytes=%d",
        path,
        sidecar,
        source_digest,
        source_bytes,
        result.output_bytes,
    )
    return sidecar
