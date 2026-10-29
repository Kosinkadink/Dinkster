"""Reading weight-source payloads into torch tensors.

The torch-free layer parses and validates safetensors headers
(dinkster_inference.sources: keys -> geometry + absolute byte ranges)
without touching payload bytes. This module is the executing half:
map the file once and wrap each selected range as a torch tensor -
the stage-4 "file-slice reads" surface (docs/native-inference-plan.md
1.4), with no dependency on the safetensors package (the header
parser is Dinkster's own; the payload is plain packed bytes).

Every returned nonempty tensor owns a reference to one copy-on-write
mapping through its torch storage. The mapping therefore remains
valid after this loader returns, is released after the last tensor
using it is released, and lets callers mutate tensors without writing
the checkpoint. Its storage also retains the original file range so
residency backends can transfer immutable weights without faulting the
mapped pages. Payload bytes are little-endian per the safetensors format;
this reader requires a little-endian host and refuses otherwise rather
than byte-swapping silently.
"""

from __future__ import annotations

import mmap
import os
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, cast

import torch
from dinkster_inference.gguf import (
    GGUFEncodedStorage,
    GGUFWeightSource,
    admit_gguf_encoded_storage,
    decode_gguf_encoded_storage,
)
from dinkster_inference.sources import (
    SafetensorsSource,
    load_safetensors_header,
)

from .gguf_linear import GGUF_BLOCK_DECODERS

__all__ = [
    "SourceReadError",
    "TensorFileSlice",
    "load_tensors",
    "load_tensors_from_file",
    "tensor_file_slice",
]


class SourceReadError(ValueError):
    """A payload read that cannot produce the requested tensors: an
    unknown key, a dtype torch cannot represent, or a payload shorter
    than its header promised."""


@dataclass(frozen=True)
class TensorFileSlice:
    """The open checkpoint range backing one mapped tensor."""

    file: BinaryIO
    lock: threading.Lock
    offset: int
    size: int


def tensor_file_slice(tensor: torch.Tensor) -> TensorFileSlice | None:
    """Return direct-read metadata for an unchanged mapped tensor."""

    info = getattr(tensor.untyped_storage(), "_dinkster_tensor_file_slice", None)
    if (
        not isinstance(info, TensorFileSlice)
        or tensor.device.type != "cpu"
        or tensor.storage_offset() != 0
        or not tensor.is_contiguous()
        or tensor.nbytes != info.size
    ):
        return None
    return info


#: Dinkster dtype names (devices.py) -> torch storage dtypes. Newer torch
#: releases expose packed F4 and E8M0; F6 has no torch representation.
_TORCH_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e5m2fnuz": torch.float8_e5m2fnuz,
        "complex64": torch.complex64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "uint64": torch.uint64,
        "bool": torch.bool,
        **({"float4_e2m1": torch.float4_e2m1fn_x2} if hasattr(torch, "float4_e2m1fn_x2") else {}),
        **({"float8_e8m0": torch.float8_e8m0fnu} if hasattr(torch, "float8_e8m0fnu") else {}),
    }
)

#: The scaled-FP8 formats implemented by the quantized linear executor.
#: Keep dtype-name ownership beside the complete torch dtype table.
FP8_QUANT_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {name: _TORCH_DTYPES[name] for name in ("float8_e4m3fn", "float8_e5m2")}
)


_GGUF_PLAIN_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
    }
)


def _read_gguf_slice(storage: GGUFEncodedStorage) -> bytes:
    try:
        return storage.file_slice.read()
    except OSError as error:
        raise SourceReadError(str(error)) from error


def _decode_quantized(storage: GGUFEncodedStorage) -> torch.Tensor:
    """Decode admitted quantized storage, preferring the vectorized kernels.

    Every layout with a vectorized decoder is bit-identical to the pure
    reference decode; a layout without one falls back to the reference.
    """

    decoder = GGUF_BLOCK_DECODERS.get(storage.descriptor.ggml_type.name)
    if decoder is None:
        return torch.tensor(decode_gguf_encoded_storage(storage), dtype=torch.float32).reshape(
            storage.logical_shape
        )
    raw = torch.frombuffer(bytearray(_read_gguf_slice(storage)), dtype=torch.uint8)
    blocks = raw.reshape(-1, storage.descriptor.block_bytes)
    return decoder(blocks, storage.logical_shape)


def load_gguf_encoded_blocks(
    authority: GGUFWeightSource,
    key: str,
    *,
    expected_runtime_facts: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Read one mapped quantized tensor's encoded blocks as a
    (blocks, block_bytes) uint8 tensor. The tensor's layout must have
    a vectorized block decoder so an encoded-resident linear can
    decode it on use."""

    if expected_runtime_facts is not None and authority.runtime_facts != expected_runtime_facts:
        raise SourceReadError(
            f"{authority.path}: GGUF artifact or execution identity changed after planning"
        )
    mapped = authority.component_map.public_tensors().get(key)
    if mapped is None:
        raise SourceReadError(f"{authority.path}: no mapped GGUF tensor named {key!r}")
    storage = admit_gguf_encoded_storage(authority, mapped.model_key)
    if storage.descriptor.ggml_type.name not in GGUF_BLOCK_DECODERS:
        raise SourceReadError(
            f"{authority.path}: GGUF tensor {key!r} is"
            f" {storage.descriptor.ggml_type.name}, which has no"
            f" encoded-resident block decoder"
        )
    raw = torch.frombuffer(bytearray(_read_gguf_slice(storage)), dtype=torch.uint8)
    return raw.reshape(-1, storage.descriptor.block_bytes)


def load_gguf_tensors(
    authority: GGUFWeightSource,
    keys: Iterable[str] | None = None,
    *,
    expected_runtime_facts: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Decode selected mapped GGUF tensors through the admitted CPU route."""

    if expected_runtime_facts is not None and authority.runtime_facts != expected_runtime_facts:
        raise SourceReadError(
            f"{authority.path}: GGUF artifact or execution identity changed after planning"
        )
    source = authority.source
    component = authority.component_map
    # Requested keys use the source's public spelling (matching
    # GGUFWeightSource.keys()); text maps read a different underlying
    # source_name.
    by_public = component.public_tensors()
    selected = tuple(by_public) if keys is None else tuple(keys)
    out: dict[str, torch.Tensor] = {}
    for key in selected:
        mapped = by_public.get(key)
        if mapped is None:
            raise SourceReadError(f"{authority.path}: no mapped GGUF tensor named {key!r}")
        tensor = source.tensor(mapped.source_name)
        if tensor.ggml_type.quantized:
            storage = admit_gguf_encoded_storage(authority, mapped.model_key)
            out[key] = _decode_quantized(storage)
            continue
        dtype = _GGUF_PLAIN_DTYPES.get(tensor.ggml_type.name)
        if dtype is None:
            raise SourceReadError(
                f"{authority.path}: GGUF tensor {key!r} has unsupported plain type"
                f" {tensor.ggml_type.name}"
            )
        try:
            data = bytearray(authority.file_slice(tensor.offset, tensor.nbytes).read())
        except OSError as error:
            raise SourceReadError(str(error)) from error
        out[key] = torch.frombuffer(data, dtype=dtype).reshape(mapped.logical_shape)
    return out


def load_tensors(path: Path, keys: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """Read ``keys`` (default: every tensor) from a safetensors file.

    Header validation is the torch-free parser's
    (``MalformedSafetensors`` on format violations); this function
    adds :class:`SourceReadError` for unknown keys, torch-unmappable
    dtypes, and truncated payloads. For header-only inspection or
    metadata, use ``load_safetensors_header`` directly.
    """
    source = load_safetensors_header(path)
    with path.open("rb") as file:
        return load_tensors_from_file(file, source, keys)


def load_tensors_from_file(
    file: BinaryIO,
    source: SafetensorsSource,
    keys: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Map tensors from the same caller-owned file used to parse ``source``."""

    path = source.path
    if sys.byteorder != "little":
        raise SourceReadError(
            "safetensors payloads are little-endian; big-endian hosts are not supported"
        )
    selected = tuple(source.keys()) if keys is None else tuple(keys)
    out: dict[str, torch.Tensor] = {}
    planned = []
    needs_mapping = False
    for key in selected:
        if key not in source.entries:
            raise SourceReadError(f"{path}: no tensor named {key!r}")
        entry = source.entry(key)
        geometry = entry.geometry
        dtype = _TORCH_DTYPES.get(geometry.dtype.name)
        if dtype is None:
            raise SourceReadError(
                f"{path}: tensor {key!r} has dtype"
                f" {geometry.dtype.name}, which has no packed torch"
                " representation"
            )
        shape, count = geometry.shape, geometry.numel
        if geometry.dtype.name == "float4_e2m1":
            # safetensors stores logical F4 geometry; torch packs two values
            # per element along the last axis, not across arbitrary axes.
            if not shape or shape[-1] % 2:
                raise SourceReadError(f"{path}: tensor {key!r} requires an even final F4 dimension")
            shape = (*shape[:-1], shape[-1] // 2)
            count //= 2
        planned.append((key, entry, dtype, shape, count))
        needs_mapping = needs_mapping or geometry.numel != 0

    file_size = os.fstat(file.fileno()).st_size
    for key, entry, _dtype, _shape, _count in planned:
        geometry = entry.geometry
        if geometry.numel != 0 and entry.offset + entry.nbytes > file_size:
            available = max(0, file_size - entry.offset)
            raise SourceReadError(
                f"{path}: tensor {key!r} payload truncated ({available} of {entry.nbytes} bytes)"
            )

    if not needs_mapping:
        for key, _entry, dtype, shape, _count in planned:
            out[key] = torch.empty(shape, dtype=dtype, device="cpu")
        return out

    # ACCESS_COPY is MAP_PRIVATE with a writable buffer: torch can
    # mutate its views, but dirty pages can never reach the file.
    payload = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_COPY)
    direct_file = cast(BinaryIO, os.fdopen(os.dup(file.fileno()), "rb", buffering=0))
    file_lock = threading.Lock()
    for key, entry, dtype, shape, count in planned:
        geometry = entry.geometry
        if geometry.numel == 0:
            out[key] = torch.empty(shape, dtype=dtype, device="cpu")
            continue
        tensor = torch.frombuffer(
            payload,
            dtype=dtype,
            count=count,
            offset=entry.offset,
        )
        value = tensor.reshape(shape)
        cast(Any, value.untyped_storage())._dinkster_tensor_file_slice = TensorFileSlice(
            direct_file,
            file_lock,
            entry.offset,
            entry.nbytes,
        )
        out[key] = value
    return out
