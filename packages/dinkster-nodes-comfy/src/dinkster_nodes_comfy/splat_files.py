"""Serialize gaussian splats through the public 3D file asset boundary."""

from __future__ import annotations

import gzip
import importlib
import logging
import math
import struct
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    FILE_3D_KSPLAT_TYPE,
    FILE_3D_PLY_TYPE,
    FILE_3D_SPLAT_ANY_TYPE,
    FILE_3D_SPLAT_TYPE,
    FILE_3D_SPZ_TYPE,
    FILE_3D_TYPE,
    SPLAT_TYPE,
    AssetRef,
    ComboWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    digest_bytes,
    materialize_splat,
    parse_ply_splat,
    produced_output_scratch,
    render_splat_ply,
)

_SH_C0 = 0.28209479177387814


def _read_splat(data: bytes) -> tuple[Any, ...]:
    np = importlib.import_module("numpy")
    if len(data) % 32:
        raise ValueError("File3DToSplat: .splat size is not a multiple of 32 bytes")
    records = np.frombuffer(
        data,
        np.dtype([("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("quat", "u1", 4)]),
    )
    rgba = records["rgba"].astype(np.float32) / 255.0
    rotation = (records["quat"].astype(np.float32) - 128.0) / 128.0
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    return (
        records["xyz"].astype(np.float32),
        records["scale"].astype(np.float32),
        rotation,
        rgba[:, 3].copy(),
        ((rgba[:, :3] - 0.5) / _SH_C0)[:, None, :],
    )


def _read_ksplat(data: bytes) -> tuple[Any, ...]:
    np = importlib.import_module("numpy")
    if len(data) < 4096:
        raise ValueError("File3DToSplat: truncated .ksplat header")
    if data[0] != 0:
        raise ValueError(f"File3DToSplat: unsupported .ksplat version {data[0]}.{data[1]}")
    sections = struct.unpack_from("<I", data, 4)[0]
    level = struct.unpack_from("<H", data, 20)[0]
    if level not in (0, 1, 2):
        raise ValueError(f"File3DToSplat: invalid .ksplat compression level {level}")
    base = 4096 + sections * 1024
    if base > len(data):
        raise ValueError("File3DToSplat: truncated .ksplat section headers")
    parts: list[tuple[Any, ...]] = []
    for section in range(sections):
        offset = 4096 + section * 1024
        count, capacity, bucket_size, bucket_count, block_size, bucket_store = struct.unpack_from(
            "<IIIIfH", data, offset
        )
        scale_range = struct.unpack_from("<I", data, offset + 24)[0] or (1 if level == 0 else 32767)
        full_buckets, partial_buckets, degree = struct.unpack_from("<IIH", data, offset + 32)
        sh_components = {0: 0, 1: 9, 2: 24, 3: 45}.get(degree, 0)
        center_type, float_type = ("<f4", "<f4") if level == 0 else ("<u2", "<f2")
        fields = [
            ("center", center_type, 3),
            ("scale", float_type, 3),
            ("rotation", float_type, 4),
            ("color", "u1", 4),
        ]
        if sh_components:
            fields.append(("sh", ("<f4", "<f2", "u1")[level], sh_components))
        record_type = np.dtype(fields)
        metadata_size = partial_buckets * 4
        bucket_bytes = bucket_store * bucket_count + metadata_size
        if count > capacity or base + bucket_bytes + capacity * record_type.itemsize > len(data):
            raise ValueError("File3DToSplat: truncated .ksplat section data")
        if count:
            records = np.frombuffer(data, record_type, count=count, offset=base + bucket_bytes)
            rgba = records["color"].astype(np.float32) / 255.0
            rotation = records["rotation"].astype(np.float32)
            rotation /= np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
            xyz = records["center"].astype(np.float32)
            if level:
                buckets = np.frombuffer(
                    data, "<f4", count=bucket_count * 3, offset=base + metadata_size
                ).reshape(-1, 3)
                if bucket_size == 0 or full_buckets > bucket_count:
                    raise ValueError("File3DToSplat: invalid .ksplat buckets")
                full_count = min(full_buckets * bucket_size, count)
                indices = np.empty(count, np.int64)
                indices[:full_count] = np.arange(full_count) // bucket_size
                if full_count < count:
                    lengths = np.frombuffer(data, "<u4", count=partial_buckets, offset=base)
                    tail = np.repeat(full_buckets + np.arange(partial_buckets), lengths)
                    if len(tail) < count - full_count or np.any(tail >= bucket_count):
                        raise ValueError("File3DToSplat: invalid .ksplat partial buckets")
                    indices[full_count:] = tail[: count - full_count]
                xyz = (xyz - scale_range) * (block_size / 2.0 / scale_range) + buckets[indices]
            parts.append(
                (
                    xyz,
                    records["scale"].astype(np.float32),
                    rotation,
                    rgba[:, 3].copy(),
                    ((rgba[:, :3] - 0.5) / _SH_C0)[:, None, :],
                )
            )
        base += bucket_bytes + capacity * record_type.itemsize
    if not parts:
        raise ValueError("File3DToSplat: .ksplat has no splats")
    return tuple(np.concatenate([part[index] for part in parts]) for index in range(5))


def _read_spz(data: bytes) -> tuple[Any, ...]:
    np = importlib.import_module("numpy")
    raw = gzip.decompress(data)
    magic, version, count, _, fractional_bits, _, _ = struct.unpack_from("<IIIBBBB", raw)
    if magic != 0x5053474E:
        raise ValueError("File3DToSplat: invalid .spz (bad magic)")
    offset = 16
    if version == 1:
        xyz = np.frombuffer(raw, "<f2", count=count * 3, offset=offset).astype(np.float32)
        xyz = xyz.reshape(count, 3)
        offset += count * 6
    elif version in (2, 3):
        packed = np.frombuffer(raw, np.uint8, count=count * 9, offset=offset)
        packed = packed.reshape(count, 3, 3).astype(np.int64)
        signed = packed[..., 0] | (packed[..., 1] << 8) | (packed[..., 2] << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        xyz = (signed / (1 << fractional_bits)).astype(np.float32)
        offset += count * 9
    else:
        raise ValueError(f"File3DToSplat: unsupported .spz version {version}")
    alpha = np.frombuffer(raw, np.uint8, count=count, offset=offset).astype(np.float32) / 255.0
    offset += count
    color = np.frombuffer(raw, np.uint8, count=count * 3, offset=offset).reshape(count, 3)
    rgb = (color.astype(np.float32) / 255.0 - 0.5) * (_SH_C0 / 0.15) + 0.5
    offset += count * 3
    scale = np.frombuffer(raw, np.uint8, count=count * 3, offset=offset).reshape(count, 3)
    scale = np.exp(scale.astype(np.float32) / 16.0 - 10.0)
    offset += count * 3
    if version == 3:
        remaining = np.frombuffer(raw, "<u4", count=count, offset=offset).astype(np.int64)
        largest = (remaining >> 30) & 3
        quaternion = np.zeros((count, 4), np.float32)
        sum_squared = np.zeros(count, np.float64)
        for component in (3, 2, 1, 0):
            active = component != largest
            magnitude = (remaining & 0x1FF).astype(np.float64) / (0x1FF * math.sqrt(2))
            value = np.where((remaining >> 9) & 1, -magnitude, magnitude)
            remaining = np.where(active, remaining >> 10, remaining)
            quaternion[active, component] = value[active]
            sum_squared += np.where(active, value * value, 0.0)
        quaternion[np.arange(count), largest] = np.sqrt(np.clip(1.0 - sum_squared, 0, None))
        rotation = quaternion[:, [3, 0, 1, 2]]
    else:
        packed_rotation = np.frombuffer(raw, np.uint8, count=count * 3, offset=offset)
        xyz_rotation = packed_rotation.reshape(count, 3).astype(np.float32) / 127.5 - 1.0
        w = np.sqrt(np.clip(1.0 - (xyz_rotation**2).sum(1), 0, None))
        rotation = np.concatenate([w[:, None], xyz_rotation], 1)
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    return xyz, scale, rotation, alpha, ((rgb - 0.5) / _SH_C0)[:, None, :]


class File3DToSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="File3DToSplat",
            display_name="Get Splat",
            search_terms=("load splat", "ply to splat", "import splat", "file to splat"),
            category="3d/splat",
            description=(
                "Parse a splat File3D into a gaussian splat. "
                "Inverse of Create 3D File (from Splat). "
                "Supported format:  PLY, SPLAT, KSPLAT, SPZ. PLY carries full spherical harmonics, "
                "the other formats are base color only. "
                "Format is auto-detected from the file contents."
            ),
            inputs=(
                InputSpec(
                    "model_3d",
                    TypeExpr.union(
                        FILE_3D_TYPE,
                        FILE_3D_SPLAT_ANY_TYPE,
                        FILE_3D_PLY_TYPE,
                        FILE_3D_SPLAT_TYPE,
                        FILE_3D_KSPLAT_TYPE,
                        FILE_3D_SPZ_TYPE,
                    ),
                    doc="A gaussian splat 3D file",
                ),
            ),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_3d: Mapping[str, object]) -> Mapping[str, object]:
        source = model_3d["source"]
        if not isinstance(source, (AssetRef, LocalAssetRef)):
            raise TypeError("File3DToSplat requires a FILE_3D asset identity")
        data = source.read_bytes()
        format = str(model_3d["format"] or "").lower()
        if format not in ("ply", "splat", "ksplat", "spz"):
            if data[:3] == b"ply":
                format = "ply"
            elif data[:2] == b"\x1f\x8b":
                format = "spz"
            elif len(data) >= 2 and data[0] == 0 and data[1] >= 1:
                format = "ksplat"
            elif len(data) % 32 == 0:
                format = "splat"
            else:
                raise ValueError("File3DToSplat: could not determine splat format from contents")
        if format == "ply":
            arrays = parse_ply_splat(data)
        else:
            np = importlib.import_module("numpy")
            values = {"splat": _read_splat, "ksplat": _read_ksplat, "spz": _read_spz}[format](data)
            arrays = {
                key: np.ascontiguousarray(value.reshape(-1, 1) if key == "opacities" else value)[
                    None
                ]
                for key, value in zip(
                    ("positions", "scales", "rotations", "opacities", "sh"), values, strict=True
                )
            }
        return cls.outputs(splat=materialize_splat(arrays))


def _ksplat_bytes(value: Any, end: int) -> bytes:
    np = importlib.import_module("numpy")
    xyz = value.positions[0, :end].cpu().numpy().astype(np.float32)
    scale = value.scales[0, :end].cpu().numpy().astype(np.float32)
    rotation = value.rotations[0, :end].cpu().numpy().astype(np.float32)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    rgb = np.clip(value.sh[0, :end, 0, :].cpu().numpy().astype(np.float32) * _SH_C0 + 0.5, 0, 1)
    opacity = value.opacities[0, :end].cpu().numpy().astype(np.float32).reshape(end, 1).clip(0, 1)
    rgba = np.round(np.concatenate([rgb, opacity], axis=1) * 255.0).astype(np.uint8)
    records = np.empty(end, dtype=[("f", "<f4", 10), ("c", "u1", 4)])
    records["f"] = np.concatenate([xyz, scale, rotation], axis=1).astype("<f4")
    records["c"] = rgba
    header = bytearray(4096)
    header[1] = 1
    struct.pack_into("<IIII", header, 4, 1, 1, end, end)
    section = bytearray(1024)
    struct.pack_into("<II", section, 0, end, end)
    struct.pack_into("<I", section, 28, end * 44)
    return bytes(header) + bytes(section) + bytes(records.tobytes())


def _spz_bytes(value: Any, end: int) -> bytes:
    np = importlib.import_module("numpy")
    xyz = value.positions[0, :end].cpu().numpy().astype(np.float32)
    signed = np.clip(np.round(xyz * (1 << 12)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    unsigned = (signed & 0xFFFFFF).astype(np.uint32)
    positions = (
        np.stack([unsigned & 0xFF, (unsigned >> 8) & 0xFF, (unsigned >> 16) & 0xFF], axis=-1)
        .reshape(end, 9)
        .astype(np.uint8)
    )
    opacity = (
        np.round(value.opacities[0, :end].cpu().numpy().astype(np.float32).reshape(end) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    rgb = value.sh[0, :end, 0, :].cpu().numpy().astype(np.float32) * _SH_C0 + 0.5
    colors = np.round(((rgb - 0.5) / (_SH_C0 / 0.15) + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    log_scale = np.log(value.scales[0, :end].cpu().numpy().astype(np.float32).clip(min=1e-9))
    scales = np.round((log_scale + 10.0) * 16.0).clip(0, 255).astype(np.uint8)
    rotation = value.rotations[0, :end].cpu().numpy().astype(np.float32)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    rotation[rotation[:, 0] < 0] *= -1.0
    rotations = np.round((rotation[:, 1:4] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    header = struct.pack("<IIIBBBB", 0x5053474E, 2, end, 0, 12, 0, 0)
    return gzip.compress(
        header
        + positions.tobytes()
        + opacity.tobytes()
        + colors.tobytes()
        + scales.tobytes()
        + rotations.tobytes()
    )


class SplatToFile3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplatToFile3D",
            display_name="Create 3D File (from Splat)",
            search_terms=("gaussian to ply", "splat to file", "export gaussian"),
            category="3d/splat",
            description=(
                "Serialize a gaussian splat to a File3D object for Save / Preview 3D nodes. "
                "Supports one item per batch only."
            ),
            inputs=(
                InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                InputSpec(
                    "format",
                    TypeExpr.concrete(CORE_COMBO),
                    default="ply",
                    widget=ComboWidget(options=("ply", "ksplat", "spz")),
                ),
            ),
            outputs=(OutputSpec("model_3d", TypeExpr.concrete(FILE_3D_SPLAT_ANY_TYPE)),),
        )

    @classmethod
    def execute(cls, *, splat: object, format: str = "ply") -> Mapping[str, object]:
        if format not in ("ply", "ksplat", "spz"):
            raise ValueError(f"Unsupported splat format: {format!r}")
        value = materialize_splat(splat)
        if value.positions.shape[0] > 1:
            logging.warning(
                "SplatToFile3D supports one item per batch only. Got %d; using first.",
                value.positions.shape[0],
            )
        end = int(value.counts[0].item()) if value.counts is not None else value.positions.shape[1]
        if end == 0:
            raise ValueError("SplatToFile3D: gaussian is empty")
        data = (
            render_splat_ply(splat)
            if format == "ply"
            else _ksplat_bytes(value, end)
            if format == "ksplat"
            else _spz_bytes(value, end)
        )
        vault = produced_output_scratch()
        if vault is None:
            raise RuntimeError("SplatToFile3D requires a produced-output store")
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        source = AssetRef(
            digest, f"splat.{format}", len(data), "application/octet-stream", resolver=vault
        )
        return cls.outputs(model_3d={"format": format, "source": source})
