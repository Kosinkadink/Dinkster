"""Skinned sphere and limb meshes for OpenPose animation files."""

from __future__ import annotations

import importlib
from typing import Any

from .sam3d_glb import compute_normals, flat_shade_mesh, rotation_align

np = importlib.import_module("numpy")


def _shade(
    vertices: Any, faces: Any, joints: Any, weights: Any, colors: Any, smooth: bool
) -> tuple[Any, ...]:
    if smooth:
        return (
            vertices,
            compute_normals(vertices, faces, orphan_y=True),
            faces,
            joints,
            weights,
            colors.astype(np.float32),
        )
    return (
        *flat_shade_mesh(vertices, faces, joints, weights),
        colors[faces].reshape(-1, 3).astype(np.float32),
    )


def _sphere() -> tuple[Any, Any]:
    vertices: list[list[float]] = [[0, -1, 0]]
    for i in range(1, 10):
        latitude = -0.5 * np.pi + np.pi * i / 10
        y, radius = float(np.sin(latitude)), float(np.cos(latitude))
        for k in range(16):
            phi = 2.0 * np.pi * k / 16
            vertices.append([radius * float(np.cos(phi)), y, radius * float(np.sin(phi))])
    north = len(vertices)
    vertices.append([0, 1, 0])
    faces: list[list[int]] = [[0, 1 + k, 1 + (k + 1) % 16] for k in range(16)]
    for i in range(8):
        low, high = 1 + i * 16, 1 + (i + 1) * 16
        for k in range(16):
            a, b, c, d = low + k, low + (k + 1) % 16, high + (k + 1) % 16, high + k
            faces.extend([[a, c, b], [a, d, c]])
    faces.extend([[north, 129 + (k + 1) % 16, 129 + k] for k in range(16)])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def spheres(
    positions: Any, radius: float, colors: Any, joint_indices: Any, smooth: bool
) -> tuple[Any, ...]:
    vertices, faces = _sphere()
    count, nv, nf = len(positions), len(vertices), len(faces)
    v = np.zeros((count * nv, 3), dtype=np.float32)
    f = np.zeros((count * nf, 3), dtype=np.uint32)
    j = np.zeros((count * nv, 4), dtype=np.uint16)
    w = np.zeros((count * nv, 4), dtype=np.float32)
    c = np.zeros((count * nv, 3), dtype=np.float32)
    for index in range(count):
        start, stop = index * nv, (index + 1) * nv
        v[start:stop] = vertices * radius + positions[index]
        f[index * nf : (index + 1) * nf] = faces + start
        j[start:stop, 0] = int(joint_indices[index])
        w[start:stop, 0] = 1.0
        c[start:stop] = colors[index]
    return _shade(v, f, j, w, c, smooth)


def _limb(length: float, width: float, shape: str, end_fraction: float) -> tuple[Any, Any]:
    width = max(1e-6, min(float(width), float(length) * 0.5 - 1e-6))
    if shape == "capsule":
        cap, cap_rings, body_rings, longitude = 0.0, 0, 0, 16
    elif shape == "ellipsoid":
        cap = max(1e-7, width * float(min(0.95, max(0.05, end_fraction))))
        cap_rings, body_rings, longitude = 3, 7, 12
    else:
        raise ValueError(f"Unknown limb shape {shape!r}")
    if 2.0 * cap >= length:
        cap = max(0.0, length * 0.5 - 1e-6)
    body_length = float(length) - 2.0 * cap
    vertices: list[list[float]] = []
    rings: list[int] = []
    tail = -1

    def ring(radius: float, y: float) -> None:
        rings.append(len(vertices))
        for k in range(longitude):
            phi = 2.0 * np.pi * k / longitude
            vertices.append([radius * float(np.cos(phi)), float(y), radius * float(np.sin(phi))])

    if cap_rings:
        vertices.append([0, 0, 0])
        for i in range(1, cap_rings + 1):
            theta = (np.pi * 0.5) * i / cap_rings
            ring(cap * np.sin(theta), cap * (1.0 - np.cos(theta)))
    else:
        ring(width, 0.0)
    for i in range(1, body_rings + 1):
        u = i / (body_rings + 1)
        radius = cap + (width - cap) * float(np.sin(np.pi * u))
        ring(radius, cap + body_length * u)
    if cap_rings:
        for i in range(cap_rings):
            theta = (np.pi * 0.5) * (cap_rings - i) / cap_rings
            ring(cap * np.sin(theta), float(length) - cap * (1.0 - np.cos(theta)))
        tail = len(vertices)
        vertices.append([0, float(length), 0])
    else:
        ring(width, float(length))
    faces: list[list[int]] = []
    if cap_rings:
        faces.extend([[0, rings[0] + k, rings[0] + (k + 1) % longitude] for k in range(longitude)])
    for low, high in zip(rings[:-1], rings[1:], strict=True):
        for k in range(longitude):
            a, b, c, d = low + k, low + (k + 1) % longitude, high + (k + 1) % longitude, high + k
            faces.extend([[a, c, b], [a, d, c]])
    if cap_rings:
        faces.extend(
            [[tail, rings[-1] + (k + 1) % longitude, rings[-1] + k] for k in range(longitude)]
        )
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def sticks(
    positions: Any,
    pairs: Any,
    width: float,
    colors: Any,
    base: int,
    shape: str,
    smooth: bool,
    end_fraction: float,
) -> tuple[Any, ...]:
    vs: list[Any] = []
    fs: list[Any] = []
    js: list[Any] = []
    ws: list[Any] = []
    cs: list[Any] = []
    offset = 0
    for index, (a, b) in enumerate(pairs):
        head, tail = positions[a], positions[b]
        direction = tail - head
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        rotation = rotation_align(np.array([0, 1, 0], dtype=np.float32), direction / length)
        effective_width = max(0.001, width if shape == "capsule" else min(length * 0.1, width))
        vertices, faces = _limb(length, effective_width, shape, end_fraction)
        count = len(vertices)
        joints = np.zeros((count, 4), dtype=np.uint16)
        joints[:, 0] = index + base
        weights = np.zeros((count, 4), dtype=np.float32)
        weights[:, 0] = 1.0
        vs.append(vertices @ rotation.T + head)
        fs.append(faces + offset)
        js.append(joints)
        ws.append(weights)
        cs.append(np.tile(colors[index], (count, 1)).astype(np.float32))
        offset += count
    if not vs:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint32),
            np.zeros((0, 4), np.uint16),
            np.zeros((0, 4), np.float32),
            np.zeros((0, 3), np.float32),
        )
    return _shade(
        np.concatenate(vs),
        np.concatenate(fs),
        np.concatenate(js),
        np.concatenate(ws),
        np.concatenate(cs),
        smooth,
    )
