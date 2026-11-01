"""Binary glTF buffers and skinned animation geometry."""

from __future__ import annotations

import importlib
import json
import struct
from typing import Any

np = importlib.import_module("numpy")


class GLBWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def add(
        self,
        values: Any,
        kind: str,
        *,
        dtype: str = "float32",
        target: int | None = None,
        bounds: bool = False,
    ) -> int:
        array = np.ascontiguousarray(values, dtype=dtype)
        if kind == "SCALAR":
            array = array.reshape(-1)
        data = array.tobytes()
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": len(self.buffer),
            "byteLength": len(data),
        }
        self.buffer.extend(data)
        self.buffer.extend(b"\0" * (-len(self.buffer) % 4))
        if target is not None:
            view["target"] = target
        accessor: dict[str, Any] = {
            "bufferView": len(self.views),
            "componentType": {"float32": 5126, "uint16": 5123, "uint32": 5125}[dtype],
            "count": int(array.shape[0]),
            "type": kind,
        }
        if bounds:
            if kind == "SCALAR":
                accessor["min"] = [float(array.min())] if array.size else [0.0]
                accessor["max"] = [float(array.max())] if array.size else [0.0]
            else:
                accessor["min"] = array.min(axis=0).tolist() if len(array) else [0.0] * 3
                accessor["max"] = array.max(axis=0).tolist() if len(array) else [0.0] * 3
        self.views.append(view)
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def to_bytes(self, gltf: dict[str, Any]) -> bytes:
        gltf["buffers"] = [{"byteLength": len(self.buffer)}]
        gltf["bufferViews"] = self.views
        gltf["accessors"] = self.accessors
        metadata = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        metadata += b" " * (-len(metadata) % 4)
        binary = bytes(self.buffer)
        binary += b"\0" * (-len(binary) % 4)
        return (
            struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + len(binary))
            + struct.pack("<II", len(metadata), 0x4E4F534A)
            + metadata
            + struct.pack("<II", len(binary), 0x004E4942)
            + binary
        )


def inverse_bind_matrices(state: Any) -> Any:
    q = state[:, 3:7]
    x, y, z, w = (q[..., index] for index in range(4))
    norm = x * x + y * y + z * z + w * w
    s = np.where(norm > 0, 2.0 / norm, 0.0)
    rotation = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    rotation[..., 0, 0] = 1 - s * (y * y + z * z)
    rotation[..., 0, 1] = s * (x * y - z * w)
    rotation[..., 0, 2] = s * (x * z + y * w)
    rotation[..., 1, 0] = s * (x * y + z * w)
    rotation[..., 1, 1] = 1 - s * (x * x + z * z)
    rotation[..., 1, 2] = s * (y * z - x * w)
    rotation[..., 2, 0] = s * (x * z - y * w)
    rotation[..., 2, 1] = s * (y * z + x * w)
    rotation[..., 2, 2] = 1 - s * (x * x + y * y)
    matrix = np.zeros((len(state), 4, 4), dtype=np.float32)
    matrix[:, :3, :3] = rotation * state[:, 7, None, None]
    matrix[:, :3, 3] = state[:, :3]
    matrix[:, 3, 3] = 1.0
    return np.linalg.inv(matrix.astype(np.float64)).astype(np.float32).transpose(0, 2, 1)


def compute_normals(vertices: Any, faces: Any, *, orphan_y: bool = False) -> Any:
    a, b, c = (vertices[faces[:, index]] for index in range(3))
    face_normals = np.cross(b - a, c - a).astype(np.float32)
    normals = np.zeros_like(vertices, dtype=np.float32)
    for index in range(3):
        np.add.at(normals, faces[:, index], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if orphan_y:
        return np.where(
            lengths > 1e-8,
            normals / np.maximum(lengths, 1e-12),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        ).astype(np.float32)
    lengths[lengths < 1e-8] = 1.0
    return (normals / lengths).astype(np.float32)


def flat_shade_mesh(vertices: Any, faces: Any, joints: Any, weights: Any) -> tuple[Any, ...]:
    count = len(faces)
    positions = np.zeros((count * 3, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    new_joints = np.zeros((count * 3, 4), dtype=np.uint16)
    new_weights = np.zeros((count * 3, 4), dtype=np.float32)
    a, b, c = (vertices[faces[:, index]] for index in range(3))
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(length > 1e-8, normal / np.maximum(length, 1e-12), [[0.0, 1.0, 0.0]])
    for index in range(3):
        positions[index::3] = vertices[faces[:, index]]
        normals[index::3] = normal
        new_joints[index::3] = joints[faces[:, index]]
        new_weights[index::3] = weights[faces[:, index]]
    return (
        positions,
        normals,
        np.arange(count * 3, dtype=np.uint32).reshape(count, 3),
        new_joints,
        new_weights,
    )


def rotation_align(source: Any, target: Any) -> Any:
    cosine = float(np.dot(source, target))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-8:
        return (
            np.eye(3, dtype=np.float32) if cosine > 0 else np.diag([1, -1, -1]).astype(np.float32)
        )
    axis = cross / sine
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + sine * skew + (1.0 - cosine) * (skew @ skew)).astype(
        np.float32
    )


def make_lit_material(
    roughness: float = 0.85, double_sided: bool = False, opacity: float = 1.0
) -> dict[str, Any]:
    alpha = float(max(0.0, min(1.0, opacity)))
    material: dict[str, Any] = {
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, alpha],
            "metallicFactor": 0.0,
            "roughnessFactor": float(max(0.0, min(1.0, roughness))),
        }
    }
    if alpha < 1.0:
        material["alphaMode"] = "BLEND"
    if double_sided:
        material["doubleSided"] = True
    return material


def smooth_time(values: Any, window: int) -> Any:
    array = np.asarray(values, dtype=np.float64)
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-x * x / (2.0 * max(0.5, window / 4.0) ** 2))
    kernel /= kernel.sum()
    padded = np.concatenate(
        [
            np.broadcast_to(array[:1], (half,) + array.shape[1:]),
            array,
            np.broadcast_to(array[-1:], (half,) + array.shape[1:]),
        ],
        axis=0,
    )
    result = np.zeros_like(array)
    for index, weight in enumerate(kernel):
        result += weight * padded[index : index + len(array)]
    return result
