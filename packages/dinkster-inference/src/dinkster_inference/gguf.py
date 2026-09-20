"""Strict GGUF v3 inspection, component mapping, and scalar GGML decode."""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Literal, TypeAlias, cast

from .catalog import builtin_families, builtin_family_registry
from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .registry import Registry
from .t5_text import T5_XXL_CONFIG, T5TextDetectError, detect_t5_config
from .weights import TensorGeometry, WeightEntry

_MAGIC = b"GGUF"
_VERSION = 3
_DEFAULT_ALIGNMENT = 32
_MAX_HEADER_BYTES = 100_000_000
_MAX_ITEMS = 1_000_000
_MAX_STRING_BYTES = 1 << 30
_MAX_ARRAY_ITEMS = 1_000_000
_MAX_RANK = 4
_MAX_TENSOR_NAME_BYTES = 127
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1
_MAPPER_ID = "dinkster.gguf.diffusion.v1"
_TEXT_MAPPER_ID = "dinkster.gguf.text.v1"


class MalformedGGUF(ValueError):
    """The file violates Dinkster's bounded little-endian GGUF v3 contract."""


class GGUFMappingError(ValueError):
    """A valid GGUF container has no unique admitted Dinkster component mapping."""


class GGUFDecodeError(ValueError):
    """Encoded blocks cannot be decoded under an admitted GGML layout."""


class GGUFStorageRefusalCode(StrEnum):
    UNSUPPORTED_QUANT_TYPE = "unsupported-quant-type"
    LAYOUT_MISMATCH = "layout-mismatch"
    SOURCE_IDENTITY_MISMATCH = "source-identity-mismatch"
    SOURCE_RANGE_MISMATCH = "source-range-mismatch"


class GGUFStorageRefusal(ValueError):
    def __init__(self, code: GGUFStorageRefusalCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


GGUFResidencyMode: TypeAlias = Literal["speed", "memory", "balanced"]
GGUF_RESIDENCY_MODES: tuple[GGUFResidencyMode, ...] = ("speed", "memory", "balanced")
GGUFResidencySelection: TypeAlias = Literal["auto", "speed", "memory", "balanced"]
GGUF_RESIDENCY_SELECTIONS: tuple[GGUFResidencySelection, ...] = ("auto", *GGUF_RESIDENCY_MODES)


class GGUFResidencyRefusal(ValueError):
    """The artifact cannot be admitted under the requested residency mode."""


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


GGUFScalar: TypeAlias = int | float | bool | str


@dataclass(frozen=True)
class GGUFMetadataValue:
    value_type: GGUFValueType
    value: GGUFScalar | tuple[GGUFScalar, ...]
    element_type: GGUFValueType | None = None


@dataclass(frozen=True)
class GGMLType:
    code: int
    name: str
    block_elements: int
    block_bytes: int
    quantized: bool


_F32 = GGMLType(0, "F32", 1, 4, False)
_F16 = GGMLType(1, "F16", 1, 2, False)
Q4_0 = GGMLType(2, "Q4_0", 32, 18, True)
Q8_0 = GGMLType(8, "Q8_0", 32, 34, True)
Q4_K = GGMLType(12, "Q4_K", 256, 144, True)
Q5_K = GGMLType(13, "Q5_K", 256, 176, True)
Q6_K = GGMLType(14, "Q6_K", 256, 210, True)
_BF16 = GGMLType(30, "BF16", 1, 2, False)

ADMITTED_GGML_TYPES: Mapping[int, GGMLType] = MappingProxyType(
    {item.code: item for item in (_F32, _F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, _BF16)}
)


@dataclass(frozen=True)
class GGUFEncodedLayout:
    id: str
    ggml_type: GGMLType
    aliases: tuple[str, ...] = ()


_ENCODED_LAYOUTS = tuple(
    GGUFEncodedLayout(f"dinkster.gguf.{ggml_type.name.lower()}", ggml_type)
    for ggml_type in (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)
)
_ENCODED_LAYOUT_BY_CODE = MappingProxyType(
    {layout.ggml_type.code: layout for layout in _ENCODED_LAYOUTS}
)


def builtin_gguf_storage_registry() -> Registry[GGUFEncodedLayout]:
    """Return the exact encoded GGML layouts accepted as first-class storage."""

    registry: Registry[GGUFEncodedLayout] = Registry()
    for layout in _ENCODED_LAYOUTS:
        registry.register(layout)
    return registry


_GGML_TYPE_NAMES = (
    "F32",
    "F16",
    "Q4_0",
    "Q4_1",
    "Q4_2",
    "Q4_3",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q8_1",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_K",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ3_XXS",
    "IQ1_S",
    "IQ4_NL",
    "IQ3_S",
    "IQ2_S",
    "IQ4_XS",
    "I8",
    "I16",
    "I32",
    "I64",
    "F64",
    "IQ1_M",
    "BF16",
    "Q4_0_4_4",
    "Q4_0_4_8",
    "Q4_0_8_8",
    "TQ1_0",
    "TQ2_0",
    "IQ4_NL_4_4",
    "IQ4_NL_4_8",
    "IQ4_NL_8_8",
    "MXFP4",
    "NVFP4",
    "Q1_0",
    "Q2_0",
)


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    wire_shape: tuple[int, ...]
    ggml_type: GGMLType
    offset: int
    relative_offset: int
    nbytes: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class GGUFSource:
    path: Path
    version: int
    alignment: int
    data_offset: int
    tensors: Mapping[str, GGUFTensor]
    metadata_values: Mapping[str, GGUFMetadataValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(self, "metadata_values", MappingProxyType(dict(self.metadata_values)))

    def keys(self) -> Sequence[str]:
        return tuple(self.tensors)

    def tensor(self, name: str) -> GGUFTensor:
        return self.tensors[name]

    def metadata(self) -> Mapping[str, GGUFMetadataValue]:
        return self.metadata_values


@dataclass(frozen=True)
class GGUFComponentTensor:
    model_key: str
    source_name: str
    logical_shape: tuple[int, ...]
    ggml_type: GGMLType
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUFComponentMap:
    mapper_id: str
    architecture: str
    family_id: str
    component: str
    tensor_prefix: str
    tensors: Mapping[str, GGUFComponentTensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    def public_tensors(self) -> dict[str, GGUFComponentTensor]:
        """Public planning key (``tensor_prefix + model_key``) -> mapped
        tensor. This is the only key spelling a mapped GGUF presents to
        planning and loading; raw source names stay internal (they equal
        the public keys for diffusion maps and differ for text maps)."""
        return {self.tensor_prefix + tensor.model_key: tensor for tensor in self.tensors.values()}


@dataclass(frozen=True)
class GGUFEncodedStorageDescriptor:
    layout_id: str
    dtype_tag: str
    ggml_type: GGMLType
    block_elements: int
    block_bytes: int
    element_count: int
    encoded_bytes: int

    def __post_init__(self) -> None:
        if not self.layout_id or self.dtype_tag != self.ggml_type.name:
            raise ValueError("encoded storage identity does not match its GGML type")
        if (
            not self.ggml_type.quantized
            or self.block_elements <= 0
            or self.block_bytes <= 0
            or self.block_elements != self.ggml_type.block_elements
            or self.block_bytes != self.ggml_type.block_bytes
            or self.element_count < 0
            or self.element_count % self.block_elements
            or self.encoded_bytes != self.element_count // self.block_elements * self.block_bytes
        ):
            raise ValueError("encoded storage dimensions do not match its GGML layout")


def open_gguf_artifact_file(path: Path) -> BinaryIO:
    if os.name != "nt":
        return path.open("rb")

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    raw_path = str(path.resolve())
    extended_path = (
        "\\\\?\\UNC\\" + raw_path[2:] if raw_path.startswith("\\\\") else "\\\\?\\" + raw_path
    )
    handle = create_file(
        extended_path,
        0x80000000,
        0x00000001 | 0x00000004,
        None,
        3,
        0x00000080,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = str(path)
        raise error
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close_handle(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


class _GGUFArtifactHandle:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = open_gguf_artifact_file(path)
        self.lock = threading.Lock()
        self.size = os.fstat(self.file.fileno()).st_size
        self._fingerprint = self._current_fingerprint()
        self._identity_fingerprint = self._current_identity_fingerprint()
        self._file_sha256: bytes | None = None
        self._detached_verified = False
        self._range_sha256: dict[tuple[int, int], bytes] = {}

    def _current_fingerprint(self) -> tuple[int, int, int, int, int]:
        stat = os.fstat(self.file.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _current_identity_fingerprint(self) -> tuple[int, int, int, int, int]:
        return self._current_fingerprint()

    def _path_names_retained_artifact(self) -> bool:
        try:
            path_stat = self.path.stat()
        except OSError:
            return False
        return os.path.samestat(path_stat, os.fstat(self.file.fileno()))

    def _detached_artifact_is_unchanged(self, fingerprint: tuple[int, int, int, int, int]) -> bool:
        if self._file_sha256 is None:
            return False
        self.file.seek(0)
        digest = hashlib.sha256()
        while chunk := self.file.read(8 * 1024**2):
            digest.update(chunk)
        if self._current_fingerprint() != fingerprint or digest.digest() != self._file_sha256:
            return False
        self._fingerprint = fingerprint
        self._identity_fingerprint = fingerprint
        self._detached_verified = True
        return True

    def _require_unchanged(self) -> None:
        fingerprint = self._current_fingerprint()
        path_names_retained_artifact = self._path_names_retained_artifact()
        if fingerprint == self._fingerprint and (
            path_names_retained_artifact or self._detached_verified
        ):
            return
        if not path_names_retained_artifact and self._detached_artifact_is_unchanged(fingerprint):
            return
        raise OSError(f"{self.path}: GGUF artifact changed after identity verification")

    def _require_identity_unchanged(self) -> None:
        if self._current_identity_fingerprint() != self._identity_fingerprint:
            raise OSError(f"{self.path}: GGUF artifact changed during identity verification")

    def parse(self) -> GGUFSource:
        with self.lock:
            self._require_identity_unchanged()
            self.file.seek(0)
            source = load_gguf(self.path, _artifact_file=self.file)
            self._require_identity_unchanged()
            return source

    def sha256(self, ranges: Iterable[tuple[int, int]]) -> str:
        digest = hashlib.sha256()
        ordered_ranges = sorted(set(ranges))
        for offset, nbytes in ordered_ranges:
            if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
                raise ValueError("GGUF authenticated range falls outside its artifact authority")
        range_digests = {byte_range: hashlib.sha256() for byte_range in ordered_ranges}
        with self.lock:
            self._require_identity_unchanged()
            self.file.seek(0)
            position = 0
            range_index = 0
            while chunk := self.file.read(8 * 1024**2):
                digest.update(chunk)
                chunk_end = position + len(chunk)
                while (
                    range_index < len(ordered_ranges) and ordered_ranges[range_index][0] < chunk_end
                ):
                    byte_range = ordered_ranges[range_index]
                    range_start, range_size = byte_range
                    range_end = range_start + range_size
                    overlap_start = max(position, range_start)
                    overlap_end = min(chunk_end, range_end)
                    if overlap_start < overlap_end:
                        range_digests[byte_range].update(
                            chunk[overlap_start - position : overlap_end - position]
                        )
                    if range_end > chunk_end:
                        break
                    range_index += 1
                position = chunk_end
            self._require_identity_unchanged()
            file_sha256 = digest.digest()
            self._file_sha256 = file_sha256
            self._range_sha256 = {
                byte_range: range_digest.digest()
                for byte_range, range_digest in range_digests.items()
            }
        return file_sha256.hex()

    def read(self, offset: int, nbytes: int) -> bytes:
        expected_sha256 = self._range_sha256.get((offset, nbytes))
        if expected_sha256 is None:
            raise OSError(
                f"{self.path}: GGUF range was not authenticated during identity verification"
            )
        with self.lock:
            self._require_unchanged()
            self.file.seek(offset)
            data = self.file.read(nbytes)
            self._require_unchanged()
        if len(data) != nbytes:
            raise OSError(
                f"{self.path}: GGUF range at byte {offset} is truncated"
                f" ({len(data)} of {nbytes} bytes)"
            )
        if hashlib.sha256(data).digest() != expected_sha256:
            raise OSError(f"{self.path}: GGUF artifact changed after identity verification")
        return data

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class GGUFFileSlice:
    source_path: Path
    offset: int
    nbytes: int
    _artifact: _GGUFArtifactHandle = dataclass_field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source_path != self._artifact.path:
            raise ValueError("GGUF file slice path does not match its artifact authority")
        if self.offset < 0 or self.nbytes < 0 or self.offset + self.nbytes > self._artifact.size:
            raise ValueError("GGUF file slice falls outside its artifact authority")

    def read(self) -> bytes:
        """Read this range from the verified artifact handle."""

        return self._artifact.read(self.offset, self.nbytes)


@dataclass(frozen=True)
class GGUFEncodedStorage:
    descriptor: GGUFEncodedStorageDescriptor
    file_slice: GGUFFileSlice
    mapper_id: str
    architecture: str
    family_id: str
    component: str
    source_name: str
    model_key: str
    logical_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.mapper_id,
                self.architecture,
                self.family_id,
                self.component,
                self.source_name,
                self.model_key,
            )
        ):
            raise ValueError("encoded storage tensor identity must not be empty")
        if math.prod(self.logical_shape) != self.descriptor.element_count:
            raise ValueError("encoded storage logical shape does not match its descriptor")
        if self.file_slice.nbytes != self.descriptor.encoded_bytes:
            raise ValueError("encoded storage range does not match its descriptor")

    @property
    def source_path(self) -> Path:
        return self.file_slice.source_path

    @property
    def offset(self) -> int:
        return self.file_slice.offset


class _Reader:
    def __init__(self, handle: BinaryIO, path: Path, size: int) -> None:
        self._handle = handle
        self.path = path
        self.size = size
        self.position = 0

    def bad(self, reason: str) -> MalformedGGUF:
        return MalformedGGUF(f"{self.path}: {reason}")

    def read(self, count: int, label: str) -> bytes:
        if count < 0 or count > self.size - self.position:
            raise self.bad(f"{label} is truncated at byte {self.position}")
        end = self.position + count
        if end > _MAX_HEADER_BYTES:
            raise self.bad(f"header and tensor index exceed {_MAX_HEADER_BYTES} bytes")
        data = self._handle.read(count)
        if len(data) != count:
            raise self.bad(f"{label} is truncated at byte {self.position}")
        self.position = end
        return data

    def unpack(self, fmt: str, label: str) -> int | float:
        size = struct.calcsize(fmt)
        return cast("int | float", struct.unpack("<" + fmt, self.read(size, label))[0])

    def u32(self, label: str) -> int:
        return cast("int", self.unpack("I", label))

    def u64(self, label: str) -> int:
        return cast("int", self.unpack("Q", label))

    def string(self, label: str) -> str:
        length = self.u64(f"{label} length")
        if length > _MAX_STRING_BYTES:
            raise self.bad(f"{label} length {length} exceeds {_MAX_STRING_BYTES} bytes")
        raw = self.read(length, label)
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise self.bad(f"{label} is not valid UTF-8") from error
        if "\x00" in value:
            raise self.bad(f"{label} contains NUL")
        return value

    def metadata_value(self, label: str) -> GGUFMetadataValue:
        code = self.u32(f"{label} type")
        try:
            value_type = GGUFValueType(code)
        except ValueError as error:
            raise self.bad(f"{label} has invalid metadata type {code}") from error
        if value_type is not GGUFValueType.ARRAY:
            return GGUFMetadataValue(value_type, self.scalar(value_type, label))
        element_code = self.u32(f"{label} array element type")
        try:
            element_type = GGUFValueType(element_code)
        except ValueError as error:
            raise self.bad(f"{label} array has invalid element type {element_code}") from error
        if element_type is GGUFValueType.ARRAY:
            raise self.bad(f"{label} array element type must be scalar")
        count = self.u64(f"{label} array count")
        if count > _MAX_ARRAY_ITEMS or count > self.size - self.position:
            raise self.bad(f"{label} array count {count} exceeds the bounded input")
        values = tuple(self.scalar(element_type, f"{label}[{index}]") for index in range(count))
        return GGUFMetadataValue(value_type, values, element_type)

    def scalar(self, value_type: GGUFValueType, label: str) -> GGUFScalar:
        formats = {
            GGUFValueType.UINT8: "B",
            GGUFValueType.INT8: "b",
            GGUFValueType.UINT16: "H",
            GGUFValueType.INT16: "h",
            GGUFValueType.UINT32: "I",
            GGUFValueType.INT32: "i",
            GGUFValueType.FLOAT32: "f",
            GGUFValueType.UINT64: "Q",
            GGUFValueType.INT64: "q",
            GGUFValueType.FLOAT64: "d",
        }
        if value_type is GGUFValueType.STRING:
            return self.string(label)
        if value_type is GGUFValueType.BOOL:
            raw = cast("int", self.unpack("b", label))
            if raw not in (0, 1):
                raise self.bad(f"{label} boolean must be encoded as 0 or 1")
            return bool(raw)
        fmt = formats.get(value_type)
        if fmt is None:
            raise self.bad(f"{label} has invalid scalar type {value_type.name}")
        return self.unpack(fmt, label)


@dataclass(frozen=True)
class _RawTensor:
    name: str
    wire_shape: tuple[int, ...]
    ggml_type: GGMLType
    relative_offset: int
    nbytes: int


def _align(value: int, alignment: int) -> int:
    if value > _UINT64_MAX - (alignment - 1):
        raise OverflowError
    return (value + alignment - 1) & -alignment


def _metadata_uint32(
    reader: _Reader, metadata: Mapping[str, GGUFMetadataValue], key: str
) -> int | None:
    field = metadata.get(key)
    if field is None:
        return None
    if field.value_type is not GGUFValueType.UINT32 or type(field.value) is not int:
        raise reader.bad(f"{key} must be UINT32")
    return field.value


def load_gguf(
    path: Path,
    *,
    _artifact_file: BinaryIO | None = None,
) -> GGUFSource:
    """Parse a complete GGUF file without reading tensor payload bytes."""

    stream = path.open("rb") if _artifact_file is None else nullcontext(_artifact_file)
    with stream as handle:
        size = os.fstat(handle.fileno()).st_size
        reader = _Reader(handle, path, size)
        if reader.read(4, "magic") != _MAGIC:
            raise reader.bad("magic must be b'GGUF'")
        version_bytes = reader.read(4, "version")
        if version_bytes == struct.pack(">I", _VERSION):
            raise reader.bad("big-endian GGUF is unsupported")
        (version,) = struct.unpack("<I", version_bytes)
        if version != _VERSION:
            raise reader.bad(f"version must be 3, got {version}")
        tensor_count = reader.u64("tensor count")
        metadata_count = reader.u64("metadata count")
        if tensor_count > _MAX_ITEMS or metadata_count > _MAX_ITEMS:
            raise reader.bad(f"tensor and metadata counts must not exceed {_MAX_ITEMS}")
        if tensor_count + metadata_count > size - reader.position:
            raise reader.bad("tensor and metadata counts exceed the possible file contents")

        metadata: dict[str, GGUFMetadataValue] = {}
        for index in range(metadata_count):
            key = reader.string(f"metadata[{index}] key")
            if not key:
                raise reader.bad("metadata key must not be empty")
            if key in metadata:
                raise reader.bad(f"duplicate metadata key {key!r}")
            metadata[key] = reader.metadata_value(f"metadata {key!r}")

        alignment = _metadata_uint32(reader, metadata, "general.alignment")
        if alignment is None:
            alignment = _DEFAULT_ALIGNMENT
        if alignment == 0 or alignment & (alignment - 1):
            raise reader.bad("general.alignment must be a nonzero power of two")

        raw_tensors: list[_RawTensor] = []
        names: set[str] = set()
        expected_offset = 0
        has_quantized = False
        for index in range(tensor_count):
            name = reader.string(f"tensor[{index}] name")
            if not name:
                raise reader.bad("tensor name must not be empty")
            if len(name.encode("utf-8")) > _MAX_TENSOR_NAME_BYTES:
                raise reader.bad(f"tensor name {name!r} must be shorter than 128 bytes")
            if name in names:
                raise reader.bad(f"duplicate tensor name {name!r}")
            names.add(name)
            rank = reader.u32(f"tensor {name!r} rank")
            if rank > _MAX_RANK:
                raise reader.bad(f"tensor {name!r} rank {rank} exceeds {_MAX_RANK}")
            wire_shape = tuple(
                cast("int", reader.unpack("q", f"tensor {name!r} dimension[{axis}]"))
                for axis in range(rank)
            )
            if any(dimension < 0 for dimension in wire_shape):
                raise reader.bad(f"tensor {name!r} dimensions must be non-negative")
            elements = 1
            for dimension in wire_shape:
                if dimension and elements >= _INT64_MAX // dimension:
                    raise reader.bad(f"tensor {name!r} element count overflows int64")
                elements *= dimension
            type_code = reader.u32(f"tensor {name!r} ggml type")
            ggml_type = ADMITTED_GGML_TYPES.get(type_code)
            if ggml_type is None:
                if type_code < len(_GGML_TYPE_NAMES):
                    raise reader.bad(
                        f"tensor {name!r} uses unsupported ggml type {_GGML_TYPE_NAMES[type_code]}"
                    )
                raise reader.bad(f"tensor {name!r} has invalid ggml type {type_code}")
            row_elements = wire_shape[0] if wire_shape else 1
            if row_elements % ggml_type.block_elements:
                raise reader.bad(
                    f"tensor {name!r} row length {row_elements} is not divisible by "
                    f"{ggml_type.name} block size {ggml_type.block_elements}"
                )
            blocks = elements // ggml_type.block_elements
            if blocks > _UINT64_MAX // ggml_type.block_bytes:
                raise reader.bad(f"tensor {name!r} packed size overflows uint64")
            nbytes = blocks * ggml_type.block_bytes
            relative_offset = reader.u64(f"tensor {name!r} offset")
            if relative_offset != expected_offset:
                relation = (
                    "overlaps"
                    if relative_offset < expected_offset
                    else "is misaligned or leaves a gap"
                )
                raise reader.bad(
                    f"tensor {name!r} offset {relative_offset} {relation}; "
                    f"expected {expected_offset}"
                )
            try:
                expected_offset = _align(relative_offset + nbytes, alignment)
            except OverflowError as error:
                raise reader.bad(f"tensor {name!r} aligned span overflows uint64") from error
            raw_tensors.append(_RawTensor(name, wire_shape, ggml_type, relative_offset, nbytes))
            has_quantized |= ggml_type.quantized

        quantization_version = _metadata_uint32(reader, metadata, "general.quantization_version")
        if has_quantized and quantization_version is None:
            raise reader.bad("quantized tensors require general.quantization_version UINT32")
        if quantization_version == 0:
            raise reader.bad("general.quantization_version must be positive")

        try:
            data_offset = _align(reader.position, alignment)
        except OverflowError as error:
            raise reader.bad("tensor data offset overflows uint64") from error
        padding = reader.read(data_offset - reader.position, "tensor data padding")
        if any(padding):
            raise reader.bad("tensor data padding must contain only zero bytes")

    if expected_offset > _UINT64_MAX - data_offset:
        raise MalformedGGUF(f"{path}: tensor data span overflows uint64")
    expected_size = data_offset + expected_offset
    if expected_size != size:
        relation = "is truncated" if size < expected_size else "has trailing bytes"
        raise MalformedGGUF(
            f"{path}: tensor data {relation}; expected {expected_size} bytes, got {size}"
        )

    tensors = {
        tensor.name: GGUFTensor(
            tensor.name,
            tuple(reversed(tensor.wire_shape)),
            tensor.wire_shape,
            tensor.ggml_type,
            data_offset + tensor.relative_offset,
            tensor.relative_offset,
            tensor.nbytes,
        )
        for tensor in raw_tensors
    }
    return GGUFSource(path, version, alignment, data_offset, tensors, metadata)


def _required_string(source: GGUFSource, key: str) -> str:
    field = source.metadata_values.get(key)
    if field is None:
        raise GGUFMappingError(f"missing required {key} STRING")
    if field.value_type is not GGUFValueType.STRING or not isinstance(field.value, str):
        raise GGUFMappingError(f"{key} must be STRING")
    if not field.value:
        raise GGUFMappingError(f"{key} must not be empty")
    return field.value


def _logical_shapes(source: GGUFSource) -> dict[str, tuple[int, ...]]:
    shapes = {name: tensor.shape for name, tensor in source.tensors.items()}
    prefix = "comfy.gguf.orig_shape."
    for key, field in source.metadata_values.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        if name not in shapes:
            raise GGUFMappingError(f"{key} names unknown tensor {name!r}")
        if (
            field.value_type is not GGUFValueType.ARRAY
            or field.element_type is not GGUFValueType.INT32
            or not isinstance(field.value, tuple)
            or not field.value
            or any(type(dimension) is not int or dimension <= 0 for dimension in field.value)
        ):
            raise GGUFMappingError(f"{key} must be a nonempty ARRAY<INT32> of positive dimensions")
        logical = cast("tuple[int, ...]", field.value)
        if math.prod(logical) != math.prod(shapes[name]):
            raise GGUFMappingError(f"{key} element count does not match tensor {name!r}")
        shapes[name] = logical
    return shapes


class _GeometrySource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self._shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self._shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self._shapes[key], FLOAT32)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def map_gguf_diffusion_component(source: GGUFSource) -> GGUFComponentMap:
    """Map one admitted diffusion profile without treating parse success as admission."""

    architecture = _required_string(source, "general.architecture")
    admitted_families = frozenset(
        family.id
        for family in builtin_families()
        if family.engine.gguf_architecture == architecture
    )
    if not admitted_families:
        raise GGUFMappingError(f"unsupported diffusion GGUF architecture {architecture!r}")
    shapes = _logical_shapes(source)
    detection = builtin_family_registry().detect(_GeometrySource(shapes))
    detected_families = tuple(dict.fromkeys(item.family_id for item in detection.candidates))
    if len(detected_families) > 1:
        names = ", ".join(detected_families)
        raise GGUFMappingError(f"ambiguous diffusion GGUF mapping: {names}")
    if detection.best is None:
        raise GGUFMappingError(
            f"unknown {architecture!r} diffusion tensor mapping; no Dinkster family matched"
        )
    family_id = detection.best.family_id
    if family_id not in admitted_families:
        raise GGUFMappingError(
            f"GGUF architecture {architecture!r} conflicts with detected family {family_id!r}"
        )
    prefix = "model.diffusion_model."
    if not any(name.startswith(prefix) for name in detection.best.matched_keys):
        prefix = ""
    component_shapes = {
        name[len(prefix) :]: shape for name, shape in shapes.items() if name.startswith(prefix)
    }
    tensors = {
        model_key: GGUFComponentTensor(
            model_key,
            prefix + model_key,
            component_shapes[model_key],
            source.tensors[prefix + model_key].ggml_type,
            source.tensors[prefix + model_key].offset,
            source.tensors[prefix + model_key].nbytes,
        )
        for model_key in component_shapes
    }
    return GGUFComponentMap(_MAPPER_ID, architecture, family_id, "diffusion", prefix, tensors)


_TEXT_ARCHITECTURES = frozenset({"t5", "t5encoder"})

#: llama.cpp T5 encoder tensor names (t5encoder exports, as published
#: by city96's t5/umt5 encoder GGUFs) -> the HF-style model keys
#: :func:`dinkster_inference.t5_text.t5_layout` produces. ffn_gate is
#: wi_0 and ffn_up is wi_1: both XXL configs are gated-act.
_T5_EXACT_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "token_embd.weight": "shared.weight",
        "enc.output_norm.weight": "encoder.final_layer_norm.weight",
    }
)
_T5_BLOCK_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        "attn_q": "layer.0.SelfAttention.q",
        "attn_k": "layer.0.SelfAttention.k",
        "attn_v": "layer.0.SelfAttention.v",
        "attn_o": "layer.0.SelfAttention.o",
        "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
        "attn_norm": "layer.0.layer_norm",
        "ffn_gate": "layer.1.DenseReluDense.wi_0",
        "ffn_up": "layer.1.DenseReluDense.wi_1",
        "ffn_down": "layer.1.DenseReluDense.wo",
        "ffn_norm": "layer.1.layer_norm",
    }
)
_T5_BLOCK_RE = re.compile(r"enc\.blk\.(0|[1-9][0-9]*)\.([a-z_]+)\.weight")


def _t5_model_key(name: str) -> str | None:
    """Translate one llama.cpp T5 tensor name, or None if unknown.

    Injective by construction: the exact names and the block names
    are disjoint, and block translations differ whenever the block
    number or suffix differs (leading zeros are rejected)."""
    exact = _T5_EXACT_NAMES.get(name)
    if exact is not None:
        return exact
    match = _T5_BLOCK_RE.fullmatch(name)
    if match is None:
        return None
    suffix = _T5_BLOCK_SUFFIXES.get(match.group(2))
    if suffix is None:
        return None
    return f"encoder.block.{match.group(1)}.{suffix}.weight"


def map_gguf_text_component(source: GGUFSource) -> GGUFComponentMap:
    """Map one llama.cpp-named T5/UMT5 encoder onto the exact XXL layouts.

    Text GGUFs keep llama.cpp source names, so ``tensor_prefix`` is
    empty and every mapped ``source_name`` differs from its
    ``model_key``; planning sees only model keys."""

    architecture = _required_string(source, "general.architecture")
    if architecture not in _TEXT_ARCHITECTURES:
        raise GGUFMappingError(f"unsupported text GGUF architecture {architecture!r}")
    shapes = _logical_shapes(source)
    by_model_key: dict[str, str] = {}
    unmapped: list[str] = []
    for name in shapes:
        model_key = _t5_model_key(name)
        if model_key is None:
            unmapped.append(name)
        else:
            by_model_key[model_key] = name
    if unmapped:
        raise GGUFMappingError(
            f"unknown {architecture!r} text tensor name(s): " + ", ".join(sorted(unmapped))
        )
    geometries = {
        model_key: TensorGeometry(shapes[name], FLOAT32) for model_key, name in by_model_key.items()
    }
    try:
        config = detect_t5_config(geometries)
    except T5TextDetectError as error:
        raise GGUFMappingError(f"unsupported text GGUF layout: {error}") from error
    component = "t5xxl" if config == T5_XXL_CONFIG else "umt5xxl"
    tensors = {
        model_key: GGUFComponentTensor(
            model_key,
            name,
            shapes[name],
            source.tensors[name].ggml_type,
            source.tensors[name].offset,
            source.tensors[name].nbytes,
        )
        for model_key, name in by_model_key.items()
    }
    return GGUFComponentMap(
        _TEXT_MAPPER_ID, architecture, f"dinkster.text.{component}", component, "", tensors
    )


def map_gguf_component(source: GGUFSource) -> GGUFComponentMap:
    """Map one admitted GGUF onto its Dinkster component by declared architecture."""

    architecture = _required_string(source, "general.architecture")
    diffusion_architectures = frozenset(
        family.engine.gguf_architecture
        for family in builtin_families()
        if family.engine.gguf_architecture is not None
    )
    if architecture in diffusion_architectures:
        return map_gguf_diffusion_component(source)
    if architecture in _TEXT_ARCHITECTURES:
        return map_gguf_text_component(source)
    supported = ", ".join(sorted((*diffusion_architectures, *_TEXT_ARCHITECTURES)))
    raise GGUFMappingError(
        f"unsupported GGUF architecture {architecture!r} (supported: {supported})"
    )


_GGUF_STORAGE_DTYPES: Mapping[str, DType] = MappingProxyType(
    {"F32": FLOAT32, "F16": FLOAT16, "BF16": BFLOAT16}
)


@dataclass(frozen=True, init=False)
class GGUFWeightSource:
    """A mapped GGUF component exposed through the planning WeightSource contract.

    Keys are presented as ``tensor_prefix + model_key`` - the spelling
    assembly planning expects - never raw source names (identical for
    diffusion maps, different for text maps).

    ``residency_mode`` selects how quantized tensors live at execution
    time and is recorded in the route runtime facts. ``auto`` (the
    default) resolves once at construction - ``balanced`` when every
    quantized tensor uses an encoded-resident layout (the layouts in
    the builtin storage registry), ``speed`` otherwise - and only the
    resolved mode reaches the route facts, so identity states what
    executes, never how it was chosen. ``speed`` decodes
    everything to float32 at load through the CPU reference route.
    ``memory`` keeps quantized linear weights as encoded blocks
    resident on the execution device and decodes them on use (every
    quantized tensor must use an encoded-resident layout); tensors
    the executor cannot hold encoded
    still load through the reference decode. ``balanced`` is the
    encoded residency of ``memory`` plus a budgeted sticky cache of
    decoded weights; ``decoded_cache_budget`` caps that cache in bytes
    (None admits per offer while the execution device keeps the
    inference working reserve free, so concurrent auto caches share
    the device without overcommit) and is normalized into the route
    facts. All three modes decode the same encoded blocks with the
    same math, so ``speed`` and ``balanced`` outputs are bit-identical.
    ``memory`` decodes on each forward and remains bit-identical to the
    reference decode route."""

    source: GGUFSource
    component_map: GGUFComponentMap
    residency_mode: GGUFResidencyMode
    decoded_cache_budget: int | None
    runtime_facts: tuple[str, ...]
    _artifact: _GGUFArtifactHandle = dataclass_field(repr=False, compare=False)

    def __init__(
        self,
        path: Path,
        *,
        residency_mode: GGUFResidencySelection = "auto",
        decoded_cache_budget: int | None = None,
    ) -> None:
        from .gguf_identity import gguf_manifest_sha256

        if not isinstance(cast("object", path), Path):
            raise TypeError("GGUF artifact authority path must be pathlib.Path")
        if residency_mode not in GGUF_RESIDENCY_SELECTIONS:
            supported = ", ".join(GGUF_RESIDENCY_SELECTIONS)
            raise GGUFResidencyRefusal(
                f"unknown GGUF residency mode {residency_mode!r} (supported: {supported})"
            )
        if decoded_cache_budget is not None:
            if residency_mode != "balanced":
                raise GGUFResidencyRefusal(
                    "decoded_cache_budget applies to the balanced residency mode"
                    f" only, not {residency_mode!r}"
                )
            if type(decoded_cache_budget) is not int or decoded_cache_budget < 0:
                raise GGUFResidencyRefusal(
                    "decoded_cache_budget must be a non-negative byte count or None"
                )
        artifact = _GGUFArtifactHandle(path)
        try:
            source = artifact.parse()
            component_map = map_gguf_component(source)
            offending = next(
                (
                    tensor
                    for tensor in component_map.tensors.values()
                    if tensor.ggml_type.quantized
                    and tensor.ggml_type.code not in _ENCODED_LAYOUT_BY_CODE
                ),
                None,
            )
            resolved: GGUFResidencyMode
            if residency_mode == "auto":
                resolved = "speed" if offending is not None else "balanced"
            else:
                resolved = residency_mode
                if resolved != "speed" and offending is not None:
                    supported_layouts = ", ".join(
                        layout.ggml_type.name for layout in _ENCODED_LAYOUTS
                    )
                    raise GGUFResidencyRefusal(
                        f"{path}: {resolved} residency requires every quantized"
                        f" tensor to use an encoded-resident layout"
                        f" ({supported_layouts}); {offending.source_name!r} is"
                        f" {offending.ggml_type.name}"
                    )
            residency_mode = resolved
            file_sha256 = artifact.sha256(
                (tensor.offset, tensor.nbytes) for tensor in source.tensors.values()
            )
        except BaseException:
            artifact.close()
            raise
        if residency_mode == "speed":
            route_facts = (
                "gguf.route.provider_key=dinkster-gguf-cpu-reference",
                "gguf.route.implementation_version=v1",
                "gguf.route.kind=reference-decode",
                "gguf.route.device_kind=cpu",
                "gguf.route.device_capability=generic",
                "gguf.route.compute_dtype=float32",
                "gguf.route.accumulation_dtype=float32",
            )
        elif residency_mode == "memory":
            route_facts = (
                "gguf.route.provider_key=dinkster-gguf-torch-onuse",
                "gguf.route.implementation_version=v1",
                "gguf.route.kind=bounded-decode",
                "gguf.route.device_kind=any",
                "gguf.route.device_capability=generic",
                "gguf.route.compute_dtype=float32",
                "gguf.route.accumulation_dtype=float32",
            )
        else:
            budget = "auto" if decoded_cache_budget is None else str(decoded_cache_budget)
            route_facts = (
                "gguf.route.provider_key=dinkster-gguf-torch-onuse",
                "gguf.route.implementation_version=v1",
                "gguf.route.kind=cached-decode",
                "gguf.route.device_kind=any",
                "gguf.route.device_capability=generic",
                "gguf.route.compute_dtype=float32",
                "gguf.route.accumulation_dtype=float32",
                f"gguf.route.decoded_cache={budget}",
            )
        runtime_facts = (
            f"gguf.artifact.file_sha256={file_sha256}",
            f"gguf.artifact.manifest_sha256={gguf_manifest_sha256(source, component_map)}",
            f"gguf.artifact.mapper_id={component_map.mapper_id}",
            f"gguf.artifact.architecture={component_map.architecture}",
            f"gguf.artifact.family_id={component_map.family_id}",
            f"gguf.artifact.component={component_map.component}",
            *route_facts,
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "component_map", component_map)
        object.__setattr__(self, "residency_mode", residency_mode)
        object.__setattr__(self, "decoded_cache_budget", decoded_cache_budget)
        object.__setattr__(self, "runtime_facts", runtime_facts)
        object.__setattr__(self, "_artifact", artifact)

    @property
    def path(self) -> Path:
        return self.source.path

    @property
    def source_format(self) -> str:
        return "gguf"

    def file_slice(self, offset: int, nbytes: int) -> GGUFFileSlice:
        return GGUFFileSlice(self.path, offset, nbytes, self._artifact)

    def keys(self) -> Sequence[str]:
        return tuple(self.component_map.public_tensors())

    def entry(self, key: str) -> WeightEntry:
        prefix = self.component_map.tensor_prefix
        model_key = key[len(prefix) :] if key.startswith(prefix) else key
        tensor = self.component_map.tensors.get(model_key)
        if tensor is None or prefix + tensor.model_key != key:
            raise KeyError(key)
        dtype = (
            FLOAT32 if tensor.ggml_type.quantized else _GGUF_STORAGE_DTYPES[tensor.ggml_type.name]
        )
        return WeightEntry(
            key,
            TensorGeometry(tensor.logical_shape, dtype),
            tensor.offset,
            tensor.nbytes,
        )

    def metadata(self) -> Mapping[str, str]:
        return {}


def _encoded_layout(
    ggml_type: GGMLType, registry: Registry[GGUFEncodedLayout]
) -> GGUFEncodedLayout:
    expected = _ENCODED_LAYOUT_BY_CODE.get(ggml_type.code)
    if expected is None or not ggml_type.quantized:
        raise GGUFStorageRefusal(
            GGUFStorageRefusalCode.UNSUPPORTED_QUANT_TYPE,
            f"GGML type {ggml_type.name} is not registered for encoded storage",
        )
    layout = registry.get(expected.id)
    claims = tuple(
        candidate for candidate in registry if candidate.ggml_type.code == ggml_type.code
    )
    if layout != expected or claims != (expected,) or expected.ggml_type != ggml_type:
        raise GGUFStorageRefusal(
            GGUFStorageRefusalCode.LAYOUT_MISMATCH,
            f"registry does not uniquely bind {expected.id!r} to parsed {ggml_type.name} facts",
        )
    return expected


def admit_gguf_encoded_storage(
    authority: GGUFWeightSource,
    model_key: str,
    *,
    registry: Registry[GGUFEncodedLayout] | None = None,
) -> GGUFEncodedStorage:
    """Admit one tensor from a verified immutable GGUF artifact authority."""

    source = authority.source
    component_map = authority.component_map
    try:
        current_map = map_gguf_component(source)
    except GGUFMappingError as error:
        raise GGUFStorageRefusal(
            GGUFStorageRefusalCode.SOURCE_IDENTITY_MISMATCH,
            str(error),
        ) from error
    if component_map != current_map:
        raise GGUFStorageRefusal(
            GGUFStorageRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "verified GGUF component map does not match its parsed source",
        )
    tensor = component_map.tensors.get(model_key)
    if tensor is None:
        raise GGUFStorageRefusal(
            GGUFStorageRefusalCode.SOURCE_IDENTITY_MISMATCH,
            f"model tensor {model_key!r} is absent from the canonical component map",
        )
    source_tensor = source.tensor(tensor.source_name)

    layouts = registry if registry is not None else builtin_gguf_storage_registry()
    layout = _encoded_layout(source_tensor.ggml_type, layouts)
    if (
        source_tensor.numel % layout.ggml_type.block_elements
        or source_tensor.nbytes
        != source_tensor.numel // layout.ggml_type.block_elements * layout.ggml_type.block_bytes
    ):
        raise GGUFStorageRefusal(
            GGUFStorageRefusalCode.LAYOUT_MISMATCH,
            f"tensor {tensor.source_name!r} byte span does not match {layout.id!r}",
        )

    descriptor = GGUFEncodedStorageDescriptor(
        layout_id=layout.id,
        dtype_tag=source_tensor.ggml_type.name,
        ggml_type=source_tensor.ggml_type,
        block_elements=source_tensor.ggml_type.block_elements,
        block_bytes=source_tensor.ggml_type.block_bytes,
        element_count=source_tensor.numel,
        encoded_bytes=source_tensor.nbytes,
    )
    return GGUFEncodedStorage(
        descriptor=descriptor,
        file_slice=authority.file_slice(source_tensor.offset, source_tensor.nbytes),
        mapper_id=component_map.mapper_id,
        architecture=component_map.architecture,
        family_id=component_map.family_id,
        component=component_map.component,
        source_name=tensor.source_name,
        model_key=tensor.model_key,
        logical_shape=tensor.logical_shape,
    )


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _half(data: bytes, offset: int) -> float:
    return struct.unpack_from("<e", data, offset)[0]


def _scale_min(index: int, packed: bytes) -> tuple[int, int]:
    if index < 4:
        return packed[index] & 0x3F, packed[index + 4] & 0x3F
    return (
        (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4),
        (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
    )


def _decode_q4_0(block: bytes) -> list[float]:
    scale = _half(block, 0)
    quants = block[2:]
    return [
        *(_f32(scale * ((value & 0x0F) - 8)) for value in quants),
        *(_f32(scale * ((value >> 4) - 8)) for value in quants),
    ]


def _decode_q8_0(block: bytes) -> list[float]:
    scale = _half(block, 0)
    quants = struct.unpack_from("<32b", block, 2)
    return [_f32(scale * value) for value in quants]


def _decode_q4_k(block: bytes) -> list[float]:
    scale = _half(block, 0)
    minimum = _half(block, 2)
    scales = block[4:16]
    quants = block[16:]
    result: list[float] = []
    for chunk in range(4):
        first_scale, first_min = _scale_min(2 * chunk, scales)
        second_scale, second_min = _scale_min(2 * chunk + 1, scales)
        d1, m1 = _f32(scale * first_scale), _f32(minimum * first_min)
        d2, m2 = _f32(scale * second_scale), _f32(minimum * second_min)
        values = quants[32 * chunk : 32 * (chunk + 1)]
        result.extend(_f32(d1 * (value & 0x0F) - m1) for value in values)
        result.extend(_f32(d2 * (value >> 4) - m2) for value in values)
    return result


def _decode_q5_k(block: bytes) -> list[float]:
    scale = _half(block, 0)
    minimum = _half(block, 2)
    scales = block[4:16]
    high = block[16:48]
    low = block[48:]
    result: list[float] = []
    for chunk in range(4):
        first_scale, first_min = _scale_min(2 * chunk, scales)
        second_scale, second_min = _scale_min(2 * chunk + 1, scales)
        d1, m1 = _f32(scale * first_scale), _f32(minimum * first_min)
        d2, m2 = _f32(scale * second_scale), _f32(minimum * second_min)
        values = low[32 * chunk : 32 * (chunk + 1)]
        first_mask, second_mask = 1 << (2 * chunk), 2 << (2 * chunk)
        result.extend(
            _f32(d1 * ((value & 0x0F) + (16 if high[index] & first_mask else 0)) - m1)
            for index, value in enumerate(values)
        )
        result.extend(
            _f32(d2 * ((value >> 4) + (16 if high[index] & second_mask else 0)) - m2)
            for index, value in enumerate(values)
        )
    return result


def _decode_q6_k(block: bytes) -> list[float]:
    low = block[:128]
    high = block[128:192]
    scales = struct.unpack_from("<16b", block, 192)
    scale = _half(block, 208)
    result = [0.0] * 256
    for half in range(2):
        for index in range(32):
            subgroup = index // 16
            low_base, high_base, scale_base = 64 * half, 32 * half, 8 * half
            first = (
                (low[low_base + index] & 0x0F) | (((high[high_base + index] >> 0) & 3) << 4)
            ) - 32
            second = (
                (low[low_base + index + 32] & 0x0F) | (((high[high_base + index] >> 2) & 3) << 4)
            ) - 32
            third = (
                (low[low_base + index] >> 4) | (((high[high_base + index] >> 4) & 3) << 4)
            ) - 32
            fourth = (
                (low[low_base + index + 32] >> 4) | (((high[high_base + index] >> 6) & 3) << 4)
            ) - 32
            output = 128 * half + index
            result[output] = _f32(_f32(scale * scales[scale_base + subgroup]) * first)
            result[output + 32] = _f32(_f32(scale * scales[scale_base + subgroup + 2]) * second)
            result[output + 64] = _f32(_f32(scale * scales[scale_base + subgroup + 4]) * third)
            result[output + 96] = _f32(_f32(scale * scales[scale_base + subgroup + 6]) * fourth)
    return result


_DECODERS = {
    Q4_0.code: _decode_q4_0,
    Q8_0.code: _decode_q8_0,
    Q4_K.code: _decode_q4_k,
    Q5_K.code: _decode_q5_k,
    Q6_K.code: _decode_q6_k,
}


def decode_ggml_blocks(ggml_type: GGMLType, data: bytes) -> tuple[float, ...]:
    """Decode complete GGML blocks to values rounded to IEEE FP32."""

    if ADMITTED_GGML_TYPES.get(ggml_type.code) != ggml_type:
        raise GGUFDecodeError(f"unknown GGML type descriptor {ggml_type!r}")
    decoder = _DECODERS.get(ggml_type.code)
    if decoder is None:
        raise GGUFDecodeError(f"{ggml_type.name} has no admitted reference decoder")
    if len(data) % ggml_type.block_bytes:
        raise GGUFDecodeError(
            f"{ggml_type.name} data length {len(data)} is not a multiple of "
            f"{ggml_type.block_bytes} bytes"
        )
    values: list[float] = []
    for offset in range(0, len(data), ggml_type.block_bytes):
        values.extend(decoder(data[offset : offset + ggml_type.block_bytes]))
    return tuple(values)


def decode_gguf_encoded_storage(storage: GGUFEncodedStorage) -> tuple[float, ...]:
    """Decode admitted storage through the GGML reference implementation."""

    descriptor = storage.descriptor
    try:
        encoded_data = storage.file_slice.read()
    except OSError as error:
        raise GGUFDecodeError(str(error)) from error
    if len(encoded_data) != descriptor.encoded_bytes:
        raise GGUFDecodeError(
            f"{descriptor.dtype_tag} storage length does not match its admitted descriptor"
        )
    values = decode_ggml_blocks(descriptor.ggml_type, encoded_data)
    if len(values) != descriptor.element_count:
        raise GGUFDecodeError(
            f"{descriptor.dtype_tag} decoded element count does not match its admitted descriptor"
        )
    return values


__all__ = [
    "ADMITTED_GGML_TYPES",
    "GGMLType",
    "GGUFComponentMap",
    "GGUFComponentTensor",
    "GGUFDecodeError",
    "GGUFEncodedLayout",
    "GGUFEncodedStorage",
    "GGUFEncodedStorageDescriptor",
    "GGUFFileSlice",
    "GGUFMappingError",
    "GGUFMetadataValue",
    "GGUFResidencyMode",
    "GGUFResidencyRefusal",
    "GGUFSource",
    "GGUFStorageRefusal",
    "GGUFStorageRefusalCode",
    "GGUFTensor",
    "GGUFValueType",
    "GGUFWeightSource",
    "MalformedGGUF",
    "Q4_0",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "admit_gguf_encoded_storage",
    "builtin_gguf_storage_registry",
    "decode_ggml_blocks",
    "decode_gguf_encoded_storage",
    "load_gguf",
    "map_gguf_component",
    "map_gguf_diffusion_component",
    "map_gguf_text_component",
]
