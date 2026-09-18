"""The dinkster.splat codec: activated world-space gaussian splat tensors.

The torch-free runtime form is a mapping of batch-first tensor-likes in the
activated world-space layout ComfyUI uses for its SPLAT type:

- ``positions`` (B, N, 3): world-space centers
- ``scales`` (B, N, 3): linear per-axis radii (post-activation)
- ``rotations`` (B, N, 4): quaternions in wxyz order
- ``opacities`` (B, N, 1): values in [0, 1] (post-sigmoid)
- ``sh`` (B, N, K, 3): spherical-harmonic coefficients, band 0 first
- ``counts`` (B,), optional: valid gaussian count per batch item

Tensor values are duck-typed the same way as latents: live torch tensors,
numpy arrays, or :class:`EncodedLatentTensor` records all work, so encoded
values round-trip without torch. The byte contract is a fixed binary frame
(magic, version, JSON array manifest, then the raw little-endian buffers in
manifest order), deterministic for fingerprinting (hazard H4).

The PLY helpers serialize batch item 0 to the standard 3D gaussian splat
``.ply`` layout (log scales, logit opacities, channel-major ``f_rest``) and
parse such files back, mirroring ComfyUI's writer/parser semantics so files
interchange cleanly.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping
from typing import Any, cast

from .latent_codec import EncodedLatentTensor, tensor_record
from .model import stable_hash

__all__ = [
    "SPLAT_CODEC_MAGIC",
    "SPLAT_CODEC_VERSION",
    "SPLAT_FILE_DECODER_ID",
    "SPLAT_PLY_MIME",
    "decode_splat",
    "decode_splat_file",
    "encode_splat",
    "parse_ply_splat",
    "render_splat_ply",
    "splat_fingerprint",
    "splat_meta",
    "validate_splat_encoded",
]

SPLAT_CODEC_MAGIC = b"DINKSTER-SPLAT\x00"
SPLAT_CODEC_VERSION = 1
SPLAT_PLY_MIME = "model/ply"

# The manifest describes at most six fixed arrays; anything bigger is garbage.
_HEADER_LIMIT = 64 * 1024
# PLY headers past this bound would be refused by media classification anyway.
_PLY_HEADER_LIMIT = 64 * 1024

_FLOAT_KEYS = ("positions", "scales", "rotations", "opacities", "sh")
_FLOAT_DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4}
_COUNT_DTYPES = {"int32": 4, "int64": 8}
_COUNT_STRUCT = {"int32": "i", "int64": "q"}

# 0.5 / C0 with C0 the degree-0 SH basis constant; converts SH DC <-> base color.
_SH_C0 = 0.28209479177387814

TensorDecoder = Callable[[EncodedLatentTensor], object]


def _records(obj: object) -> dict[str, EncodedLatentTensor]:
    """Validate the runtime form and return its arrays in canonical order."""
    if not isinstance(obj, Mapping):
        raise TypeError("splat value must be a mapping of gaussian tensors")
    mapping = cast("Mapping[str, object]", obj)
    keys = set(mapping)
    if not keys.issuperset(_FLOAT_KEYS) or not keys.issubset((*_FLOAT_KEYS, "counts")):
        raise ValueError(
            "splat value must have exactly positions, scales, rotations, "
            "opacities, sh, and optionally counts"
        )
    records: dict[str, EncodedLatentTensor] = {}
    for key in (*_FLOAT_KEYS, *(("counts",) if "counts" in keys else ())):
        record = tensor_record(mapping[key])
        if record is None:
            raise TypeError(f"splat {key} must be a tensor-like array")
        records[key] = record
    _check_layout(records)
    return records


def _check_layout(records: Mapping[str, EncodedLatentTensor]) -> None:
    positions = records["positions"]
    if len(positions.shape) != 3 or positions.shape[2] != 3:
        raise ValueError("splat positions must have shape (batch, gaussians, 3)")
    batch, gaussians = positions.shape[0], positions.shape[1]
    if batch < 1 or gaussians < 1:
        raise ValueError("splat batch and gaussian dimensions must be at least 1")
    expected: dict[str, tuple[int, ...]] = {
        "positions": (batch, gaussians, 3),
        "scales": (batch, gaussians, 3),
        "rotations": (batch, gaussians, 4),
        "opacities": (batch, gaussians, 1),
    }
    for key in _FLOAT_KEYS:
        record = records[key]
        if record.dtype not in _FLOAT_DTYPES:
            raise ValueError(f"splat {key} dtype must be one of: float16, bfloat16, float32")
        if key == "sh":
            shape = record.shape
            if len(shape) != 4 or shape[0] != batch or shape[1] != gaussians or shape[3] != 3:
                raise ValueError("splat sh must have shape (batch, gaussians, coefficients, 3)")
            if shape[2] < 1:
                raise ValueError("splat sh must carry at least the DC coefficient")
        elif record.shape != expected[key]:
            raise ValueError(f"splat {key} shape {record.shape} does not match {expected[key]}")
        _check_buffer(key, record)
    counts = records.get("counts")
    if counts is not None:
        if counts.dtype not in _COUNT_DTYPES:
            raise ValueError("splat counts dtype must be int32 or int64")
        if counts.shape != (batch,):
            raise ValueError("splat counts must have shape (batch,)")
        _check_buffer("counts", counts)
        unpacked = struct.iter_unpack(f"<{_COUNT_STRUCT[counts.dtype]}", counts.data)
        if any(value < 0 or value > gaussians for (value,) in unpacked):
            raise ValueError("splat counts must lie between 0 and the gaussian dimension")


def _check_buffer(key: str, record: EncodedLatentTensor) -> None:
    item_size = _FLOAT_DTYPES.get(record.dtype) or _COUNT_DTYPES[record.dtype]
    expected = item_size
    for dimension in record.shape:
        expected *= dimension
    if len(record.data) != expected:
        raise ValueError(f"splat {key} byte length does not match its dtype and shape")


def encode_splat(obj: object) -> bytes:
    """Serialize the runtime form as the deterministic splat frame."""
    records = _records(obj)
    manifest = [[key, record.dtype, list(record.shape)] for key, record in records.items()]
    header = json.dumps({"arrays": manifest}, separators=(",", ":"), allow_nan=False).encode()
    frame = [SPLAT_CODEC_MAGIC, bytes((SPLAT_CODEC_VERSION,)), len(header).to_bytes(4, "little")]
    frame.append(header)
    frame.extend(record.data for record in records.values())
    return b"".join(frame)


def decode_splat(data: bytes, *, tensor_decoder: TensorDecoder | None = None) -> object:
    """Parse a splat frame back into the runtime mapping.

    Without ``tensor_decoder`` the arrays stay :class:`EncodedLatentTensor`
    records, which every splat entry point accepts."""
    data = bytes(data)
    offset = len(SPLAT_CODEC_MAGIC)
    if not data.startswith(SPLAT_CODEC_MAGIC) or len(data) < offset + 5:
        raise ValueError("invalid splat codec framing")
    if data[offset] != SPLAT_CODEC_VERSION:
        raise ValueError("unsupported splat codec version")
    header_size = int.from_bytes(data[offset + 1 : offset + 5], "little")
    if header_size > _HEADER_LIMIT:
        raise ValueError("splat codec header exceeds its size bound")
    body = offset + 5 + header_size
    if body > len(data):
        raise ValueError("splat codec header is truncated")
    try:
        header: object = json.loads(data[offset + 5 : body])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid splat codec header") from exc
    if not isinstance(header, dict):
        raise ValueError("splat codec header must carry an arrays manifest")
    manifest = cast("Mapping[str, object]", header).get("arrays")
    if not isinstance(manifest, list):
        raise ValueError("splat codec header must carry an arrays manifest")
    records: dict[str, EncodedLatentTensor] = {}
    for item in cast("list[object]", manifest):
        if not isinstance(item, list):
            raise ValueError("splat codec manifest entry is invalid")
        entry = cast("list[object]", item)
        if len(entry) != 3:
            raise ValueError("splat codec manifest entry is invalid")
        key, dtype, shape = entry
        if type(key) is not str or key in records or type(dtype) is not str:
            raise ValueError("splat codec manifest entry is invalid")
        if not isinstance(shape, list):
            raise ValueError("splat codec manifest shape is invalid")
        raw_dimensions = cast("list[object]", shape)
        if any(type(n) is not int or n < 0 for n in raw_dimensions):
            raise ValueError("splat codec manifest shape is invalid")
        dimensions = cast("list[int]", raw_dimensions)
        item_size = _FLOAT_DTYPES.get(dtype) or _COUNT_DTYPES.get(dtype)
        if item_size is None:
            raise ValueError(f"splat codec manifest dtype is unsupported: {dtype}")
        size = item_size
        for dimension in dimensions:
            size *= dimension
        if body + size > len(data):
            raise ValueError("splat codec buffers are truncated")
        records[key] = EncodedLatentTensor(dtype, tuple(dimensions), data[body : body + size])
        body += size
    if body != len(data):
        raise ValueError("splat codec frame carries trailing bytes")
    ordered = (*_FLOAT_KEYS, *(("counts",) if "counts" in records else ()))
    if tuple(records) != ordered:
        raise ValueError("splat codec manifest arrays are not in canonical order")
    _check_layout(records)
    if encode_splat(records) != data:
        raise ValueError("splat codec frame is not in canonical form")
    if tensor_decoder is None:
        return dict(records)
    return {key: tensor_decoder(record) for key, record in records.items()}


def splat_fingerprint(type_id: str) -> Callable[[object], str]:
    """A form-independent fingerprint over the deterministic splat frame."""

    def fingerprint(obj: object) -> str:
        return stable_hash([type_id.encode("utf-8"), encode_splat(obj)])

    return fingerprint


def splat_meta(obj: object) -> Mapping[str, object]:
    """Report batch size, padded gaussian count, and SH coefficient count."""
    records = _records(obj)
    return {
        "batch": records["positions"].shape[0],
        "gaussians": records["positions"].shape[1],
        "sh_coefficients": records["sh"].shape[2],
    }


def validate_splat_encoded(data: bytes | memoryview, metadata: Mapping[str, object]) -> None:
    """Require envelope metadata to describe the canonical encoded frame."""
    decoded = cast("Mapping[str, EncodedLatentTensor]", decode_splat(bytes(data)))
    facts = {
        "batch": decoded["positions"].shape[0],
        "gaussians": decoded["positions"].shape[1],
        "sh_coefficients": decoded["sh"].shape[2],
    }
    for key, value in facts.items():
        declared = metadata.get(key)
        if isinstance(declared, bool) or declared != value:
            raise ValueError(f"splat metadata {key} does not match encoded bytes")


def _numpy() -> Any:
    try:
        return __import__("numpy")
    except ImportError as exc:  # pragma: no cover - numpy ships with every runtime env
        raise RuntimeError("splat PLY serialization requires numpy") from exc


def _as_float32(record: EncodedLatentTensor, numpy: Any) -> Any:
    if record.dtype == "bfloat16":
        raw = numpy.frombuffer(record.data, dtype="<u2").astype(numpy.uint32) << 16
        return raw.view(numpy.float32).reshape(record.shape)
    kind = {"float16": "<f2", "float32": "<f4"}[record.dtype]
    return numpy.frombuffer(record.data, dtype=kind).reshape(record.shape).astype(numpy.float32)


def render_splat_ply(obj: object, *, limit: int | None = None) -> bytes:
    """Serialize batch item 0 as a standard binary 3D gaussian splat PLY.

    Activated values are inverted to the storage convention (log scales,
    logit opacities); normals are written as zeros and ``f_rest`` is
    channel-major, matching ComfyUI's writer byte for byte. With ``limit``
    set, oversized outputs are refused before any conversion is allocated."""
    numpy = _numpy()
    records = _records(obj)
    counts = records.get("counts")
    end = records["positions"].shape[1]
    if counts is not None:
        item_size = _COUNT_DTYPES[counts.dtype]
        end = struct.unpack(f"<{_COUNT_STRUCT[counts.dtype]}", counts.data[:item_size])[0]
    if end == 0:
        raise ValueError("splat batch item 0 has no gaussians")
    n = int(end)
    rest = 3 * (records["sh"].shape[2] - 1)
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(rest)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    if len(header) > _PLY_HEADER_LIMIT:
        raise ValueError("splat PLY header exceeds the interchange header bound")
    if limit is not None and len(header) + n * len(names) * 4 > limit:
        raise ValueError(f"serialized gaussian splat exceeds the {limit}-byte output limit")
    xyz = _as_float32(records["positions"], numpy)[0, :end]
    normals = numpy.zeros_like(xyz)
    sh = _as_float32(records["sh"], numpy)[0, :end]
    f_dc = sh[:, 0, :]
    f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
    opacity = _as_float32(records["opacities"], numpy)[0, :end].reshape(n, 1)
    opacity = opacity.clip(1e-6, 1 - 1e-6)
    opacity = numpy.log(opacity / (1.0 - opacity))
    scale = numpy.log(_as_float32(records["scales"], numpy)[0, :end].clip(min=1e-8))
    rotation = _as_float32(records["rotations"], numpy)[0, :end]
    columns = numpy.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rotation], axis=1)
    return header.encode("ascii") + numpy.ascontiguousarray(columns.astype("<f4")).tobytes()


_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}


def parse_ply_splat(data: bytes) -> object:
    """Parse a gaussian splat PLY into the runtime form (batch of one).

    Mirrors ComfyUI's reader: log scales and logit opacities are re-activated,
    quaternions normalized, ``f_rest`` regrouped from channel-major layout,
    and plain colored point clouds (red/green/blue) mapped onto the SH DC
    band. Missing attributes take neutral defaults."""
    numpy = _numpy()
    end = data.find(b"end_header")
    if end < 0:
        raise ValueError("splat PLY is missing its end_header terminator")
    header = data[:end].decode("ascii", "replace")
    body = end + len(b"end_header")
    body += 2 if data[body : body + 2] == b"\r\n" else 1
    count, in_vertex, elements = 0, False, 0
    props: list[tuple[str, str]] = []
    for line in header.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == "format" and parts[1] != "binary_little_endian":
            raise ValueError(f"splat PLY format {parts[1]!r} is unsupported")
        if parts[0] == "element":
            elements += 1
            in_vertex = elements == 1 and parts[1] == "vertex"
            if elements == 1:
                if not in_vertex or len(parts) != 3 or not parts[2].isdigit():
                    raise ValueError("splat PLY must declare vertex as its first element")
                count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("splat PLY vertex list properties are unsupported")
            if len(parts) != 3 or parts[1] not in _PLY_DTYPES:
                raise ValueError(f"splat PLY property {line.strip()!r} is unsupported")
            props.append((parts[2], "<" + _PLY_DTYPES[parts[1]]))
    if elements == 0 or count < 1 or not props:
        raise ValueError("splat PLY must declare a non-empty vertex element")
    record_dtype = numpy.dtype(props)
    if body + count * record_dtype.itemsize > len(data):
        raise ValueError("splat PLY vertex data is truncated")
    array = numpy.frombuffer(data, record_dtype, count=count, offset=body)
    names = array.dtype.names

    def column(key: str) -> Any:
        return array[key].astype(numpy.float32)

    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("splat PLY vertex element must carry x, y, and z")
    xyz = numpy.stack([column("x"), column("y"), column("z")], 1)
    if "scale_0" in names:
        scale = numpy.exp(numpy.stack([column("scale_0"), column("scale_1"), column("scale_2")], 1))
    else:
        scale = numpy.full((count, 3), 0.01, numpy.float32)
    if "rot_0" in names:
        rotation = numpy.stack(
            [column("rot_0"), column("rot_1"), column("rot_2"), column("rot_3")], 1
        )
        rotation = rotation / numpy.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    else:
        rotation = numpy.tile(numpy.array([1, 0, 0, 0], numpy.float32), (count, 1))
    if "opacity" in names:
        opacity = 1.0 / (1.0 + numpy.exp(-column("opacity")))
    else:
        opacity = numpy.ones(count, numpy.float32)
    if "f_dc_0" in names:
        dc = numpy.stack([column("f_dc_0"), column("f_dc_1"), column("f_dc_2")], 1)
        rest_names = sorted(
            (key for key in names if key.startswith("f_rest_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        if rest_names:
            rest = numpy.stack([column(key) for key in rest_names], 1)
            bands = rest.shape[1] // 3
            rest = rest.reshape(count, 3, bands).transpose(0, 2, 1)
            sh = numpy.concatenate([dc[:, None, :], rest], 1)
        else:
            sh = dc[:, None, :]
    elif "red" in names:
        rgb = numpy.stack([column("red"), column("green"), column("blue")], 1) / 255.0
        sh = ((rgb.astype(numpy.float32) - 0.5) / _SH_C0)[:, None, :]
    else:
        sh = numpy.zeros((count, 1, 3), numpy.float32)
    return {
        "positions": xyz[None],
        "scales": scale.astype(numpy.float32)[None],
        "rotations": rotation.astype(numpy.float32)[None],
        "opacities": opacity.astype(numpy.float32).reshape(count, 1)[None],
        "sh": sh.astype(numpy.float32)[None],
    }


SPLAT_FILE_DECODER_ID = "dinkster.splat-file@1"
"""Stable identity of :func:`decode_splat_file` for coerced-input cache
fingerprints (typed assets: identity = asset digest + provider identity).
Bump the ``@N`` suffix whenever the decode SEMANTICS change - same file
bytes producing a different runtime value is a new provider identity."""


def decode_splat_file(asset: object) -> object:
    """One gaussian splat PLY asset -> the splat runtime form.

    The ``asset<dinkster.splat>`` decode provider. ``asset`` is the base asset
    runtime object, duck-typed to its ``open()`` protocol so this module
    never imports dinkster-assets (which depends on this package)."""
    opener = getattr(asset, "open", None)
    if opener is None:
        raise TypeError(
            f"decode_splat_file expects an asset with open(), got {type(asset).__name__}"
        )
    with opener() as handle:
        data = handle.read()
    try:
        return parse_ply_splat(data)
    except ValueError as exc:
        name = getattr(asset, "name", "") or "asset"
        raise ValueError(f"cannot decode '{name}' as a gaussian splat: {exc}") from exc
