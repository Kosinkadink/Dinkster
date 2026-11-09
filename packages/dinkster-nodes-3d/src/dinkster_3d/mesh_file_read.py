"""Byte-only OBJ and STL geometry readers for native mesh import."""

from __future__ import annotations

import importlib
import logging
import re
import struct
from typing import Any

np = importlib.import_module("numpy")


def _srgb_to_linear(colors: Any) -> Any:
    return np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def load_obj(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\\\n", "")
    positions: list[tuple[float, ...]] = []
    position_colors: list[tuple[float, ...]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, ...]] = []
    corner_map: dict[tuple[int, int | None, int | None], int] = {}
    out_positions: list[tuple[float, ...]] = []
    out_colors: list[tuple[float, ...]] = []
    out_uvs: list[tuple[float, float]] = []
    out_normals: list[tuple[float, ...]] = []
    faces: list[tuple[int, int, int]] = []
    has_color = any_uv = any_normal = missing_normal = warned_material = False

    def resolve(index: str, count: int) -> int:
        value = int(index)
        resolved = value - 1 if value > 0 else count + value
        if value == 0 or not 0 <= resolved < count:
            raise ValueError(f"OBJ index {value} out of range for {count} entries")
        return resolved

    def corner(spec: str) -> int:
        nonlocal any_uv, any_normal, missing_normal
        parts = spec.split("/")
        vi = resolve(parts[0], len(positions))
        ti = resolve(parts[1], len(uvs)) if len(parts) > 1 and parts[1] else None
        ni = resolve(parts[2], len(normals)) if len(parts) > 2 and parts[2] else None
        key = (vi, ti, ni)
        cached = corner_map.get(key)
        if cached is not None:
            return cached
        index = len(out_positions)
        out_positions.append(positions[vi])
        out_colors.append(position_colors[vi])
        if ti is not None:
            any_uv = True
            u, v = uvs[ti]
            out_uvs.append((u, 1.0 - v))
        else:
            out_uvs.append((0.0, 0.0))
        if ni is not None:
            any_normal = True
            out_normals.append(normals[ni])
        else:
            missing_normal = True
            out_normals.append((0.0, 0.0, 0.0))
        corner_map[key] = index
        return index

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            positions.append(tuple(float(value) for value in parts[1:4]))
            if len(parts) >= 7:
                position_colors.append(tuple(float(value) for value in parts[4:7]))
                has_color = True
            else:
                position_colors.append((1.0, 1.0, 1.0))
        elif tag == "vt":
            uvs.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
        elif tag == "vn":
            normals.append(tuple(float(value) for value in parts[1:4]))
        elif tag == "f":
            specs = parts[1:]
            if len(specs) < 3:
                continue
            indices = [corner(spec) for spec in specs]
            for index in range(1, len(indices) - 1):
                faces.append((indices[0], indices[index], indices[index + 1]))
        elif tag in ("mtllib", "usemtl") and not warned_material:
            warned_material = True
            logging.warning("Get3DComponents: OBJ materials (.mtl) are not loaded; geometry only")

    if not faces:
        raise ValueError("OBJ contains no faces")
    primitive: dict[str, Any] = {
        "positions": np.array(out_positions, np.float32),
        "faces": np.array(faces, np.int64),
        "uvs": np.array(out_uvs, np.float32) if any_uv else None,
        "colors": (
            _srgb_to_linear(np.clip(np.array(out_colors, np.float32), 0.0, 1.0))
            if has_color
            else None
        ),
        "normals": None,
        "tangents": None,
        "material": None,
    }
    if any_normal and missing_normal:
        logging.warning("Get3DComponents: OBJ has faces without vn indices; normals dropped")
    elif any_normal:
        values = np.array(out_normals, np.float32)
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        if float(lengths.max()) > 1e-6:
            primitive["normals"] = values / np.maximum(lengths, 1e-12)
    return primitive


_STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])
_STL_ASCII_FACET = re.compile(
    rb"facet\s+normal\s+(\S+)\s+(\S+)\s+(\S+).*?"
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+).*?"
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+).*?"
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)",
    re.DOTALL,
)


def load_stl(data: bytes) -> dict[str, Any]:
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if count > 0 and 84 + count * _STL_RECORD.itemsize == len(data):
        return _load_stl_binary(data, count)
    if b"solid" in data[:9]:
        facets = _STL_ASCII_FACET.findall(data)
        if not facets:
            raise ValueError("ASCII STL contains no facets")
        values = np.array(facets).astype(np.float32)
        return _stl_primitive(values[:, 3:12].reshape(-1, 3), np.repeat(values[:, 0:3], 3, axis=0))
    available = (len(data) - 84) // _STL_RECORD.itemsize if len(data) >= 84 else 0
    count = min(count, available)
    if count <= 0:
        raise ValueError("not a valid STL file (neither binary layout nor ASCII 'solid')")
    return _load_stl_binary(data, count)


def _load_stl_binary(data: bytes, count: int) -> dict[str, Any]:
    records = np.frombuffer(data, _STL_RECORD, count=count, offset=84)
    positions = records["verts"].reshape(-1, 3).astype(np.float32)
    normals = np.repeat(records["normal"], 3, axis=0).astype(np.float32)
    return _stl_primitive(positions, normals, _stl_binary_colors(data[:80], records["attr"]))


def _stl_binary_colors(header: bytes, attributes: Any) -> Any:
    position = header.find(b"COLOR=")
    if position < 0 or position + 10 > len(header):
        return None
    default = np.frombuffer(header, np.uint8, 3, position + 6).astype(np.float32) / 255.0
    values = attributes.astype(np.uint16)
    per_face = (
        np.stack([values & 0x1F, (values >> 5) & 0x1F, (values >> 10) & 0x1F], axis=1).astype(
            np.float32
        )
        / 31.0
    )
    colors = np.where(((values & 0x8000) != 0)[:, None], default[None, :], per_face)
    return _srgb_to_linear(np.repeat(colors, 3, axis=0))


def _stl_primitive(positions: Any, normals: Any, colors: Any = None) -> dict[str, Any]:
    if positions.shape[0] == 0:
        raise ValueError("STL contains no facets")
    faces = np.arange(positions.shape[0], dtype=np.int64).reshape(-1, 3)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12) if float(lengths.max()) > 1e-6 else None
    return {
        "positions": np.ascontiguousarray(positions),
        "faces": faces,
        "uvs": None,
        "colors": colors,
        "normals": normals,
        "tangents": None,
        "material": None,
    }
