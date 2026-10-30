"""Native mesh inspection and rotation with complete material preservation."""

from __future__ import annotations

import copy
import importlib
import logging
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    MESH_TYPE,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    materialize_mesh,
    report_progress,
)

_MESH = TypeExpr.concrete(MESH_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)


class RotateMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RotateMesh",
            display_name="Rotate Mesh",
            category="3d/mesh",
            description=(
                "Rotate a mesh. Euler XYZ applies X then Y then Z about the world axes (degrees). "
                "Quaternion is (w, x, y, z), auto-normalized."
            ),
            inputs=(InputSpec("mesh", _MESH),),
            combos=(
                DynamicComboSpec(
                    "mode",
                    (
                        DynamicComboOption(
                            "euler_xyz",
                            tuple(
                                InputSpec(
                                    f"angle_{axis}",
                                    _FLOAT,
                                    default=0.0,
                                    widget=NumberWidget(min=-360.0, max=360.0, step=0.1),
                                    doc=f"Rotation around the {axis.upper()} axis in degrees.",
                                )
                                for axis in "xyz"
                            ),
                        ),
                        DynamicComboOption(
                            "quaternion",
                            tuple(
                                InputSpec(
                                    f"q{axis}",
                                    _FLOAT,
                                    default=1.0 if axis == "w" else 0.0,
                                    widget=NumberWidget(min=-1.0, max=1.0, step=0.001),
                                )
                                for axis in "wxyz"
                            ),
                        ),
                    ),
                ),
            ),
            input_order=("mesh", "mode"),
            outputs=(OutputSpec("mesh", _MESH),),
        )

    @classmethod
    def execute(cls, *, mesh: Any, mode: str, **inputs: Any) -> Mapping[str, object]:
        mesh = materialize_mesh(mesh)
        if mode == "euler_xyz":
            ax, ay, az = (math.radians(inputs[f"mode.angle_{axis}"]) for axis in "xyz")
            if ax == 0.0 and ay == 0.0 and az == 0.0:
                return cls.outputs(mesh=mesh)
            cx, sx = math.cos(ax), math.sin(ax)
            cy, sy = math.cos(ay), math.sin(ay)
            cz, sz = math.cos(az), math.sin(az)
            rows = [
                [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                [-sy, sx * cy, cx * cy],
            ]
        elif mode == "quaternion":
            qw, qx, qy, qz = (inputs[f"mode.q{axis}"] for axis in "wxyz")
            norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if norm < 1e-8:
                raise ValueError("RotateMesh: quaternion has zero magnitude")
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
            if qw == 1.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
                return cls.outputs(mesh=mesh)
            rows = [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        else:
            raise ValueError(f"RotateMesh: unknown mode {mode!r}")
        torch = importlib.import_module("torch")

        def rotate(value: Any) -> Any:
            matrix = torch.tensor(rows, device=value.device, dtype=value.dtype)
            return value @ matrix.T

        result = copy.copy(mesh)
        result.vertices = rotate(mesh.vertices)
        if mesh.normals is not None:
            result.normals = rotate(mesh.normals)
        if mesh.tangents is not None:
            result.tangents = mesh.tangents.clone()
            result.tangents[..., :3] = rotate(mesh.tangents[..., :3])
        return cls.outputs(mesh=result)


class GetMeshInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetMeshInfo",
            display_name="Get Mesh Info",
            category="3d/mesh",
            inputs=(InputSpec("mesh", _MESH),),
            outputs=(
                OutputSpec("MESH", _MESH, display_name="mesh"),
                OutputSpec(
                    "STRING", TypeExpr.concrete(CORE_STRING), display_name="info", preview=True
                ),
            ),
        )

    @staticmethod
    def _fmt(count: int) -> str:
        text = f"{count:,}"
        if count >= 1_000_000:
            text += f" ({count / 1_000_000:.2f}M)"
        elif count >= 10_000:
            text += f" ({count / 1_000:.1f}K)"
        return text

    @classmethod
    def execute(cls, *, mesh: Any) -> Mapping[str, object]:
        mesh = materialize_mesh(mesh)
        batch = mesh.vertices.shape[0]
        if mesh.vertex_counts is not None:
            vertex_counts = [int(count) for count in mesh.vertex_counts.tolist()]
            face_counts = [int(count) for count in mesh.face_counts.tolist()]
        else:
            vertex_counts = [int(mesh.vertices.shape[1])] * batch
            face_counts = [int(mesh.faces.shape[1])] * batch
        attrs: list[str] = []
        for name in (
            "uvs",
            "vertex_colors",
            "normals",
            "tangents",
            "texture",
            "metallic_roughness",
            "normal_map",
        ):
            value = getattr(mesh, name)
            if value is not None:
                if name in ("texture", "metallic_roughness", "normal_map"):
                    attrs.append(f"{name} {int(value.shape[-3])}\u00d7{int(value.shape[-2])}")
                else:
                    attrs.append(name)
        lines: list[str] = []
        if batch > 1:
            lines.extend(
                (
                    f"Batch:      {batch} meshes",
                    f"Vertices:   {cls._fmt(sum(vertex_counts))} total",
                    f"Faces:      {cls._fmt(sum(face_counts))} total",
                )
            )
            for index in range(batch):
                lines.append(
                    f"  [{index}]  {vertex_counts[index]:>10,} verts  \u00b7  "
                    f"{face_counts[index]:>10,} faces"
                )
        else:
            lines.extend(
                (
                    f"Vertices:   {cls._fmt(vertex_counts[0])}",
                    f"Faces:      {cls._fmt(face_counts[0])}",
                )
            )
        lines.append(f"Attributes: {', '.join(attrs) if attrs else 'none'}")
        info = "\n".join(lines)
        report_progress(1, 1, text=info)
        return cls.outputs(MESH=mesh, STRING=info)


class MergeMeshes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MergeMeshes",
            display_name="Merge Meshes",
            category="3d/mesh",
            description=(
                "Concatenate N meshes into one by offsetting face indices and stacking verts, "
                "faces, uvs, and colors."
            ),
            input_families=(
                InputFamilySpec(
                    "meshes",
                    (InputSpec("mesh", _MESH),),
                    min_members=2,
                    max_members=50,
                    member_prefix="mesh",
                    optional_after_minimum=True,
                ),
            ),
            outputs=(OutputSpec("mesh", _MESH),),
        )

    @classmethod
    def execute(cls, *, meshes: Mapping[str, object]) -> Mapping[str, object]:
        values = [materialize_mesh(value) for value in meshes.values()]
        if not values:
            raise ValueError("MergeMeshes: need at least one mesh")
        torch = importlib.import_module("torch")
        any_uvs = any(value.uvs is not None for value in values)
        any_colors = any(value.vertex_colors is not None for value in values)
        color_channels = max(
            (
                int(value.vertex_colors.shape[-1])
                for value in values
                if value.vertex_colors is not None
            ),
            default=3,
        )
        vertices: list[Any] = []
        faces: list[Any] = []
        uvs: list[Any] = []
        colors: list[Any] = []
        texture = None
        offset = 0
        for value in values:
            # Only the first batch item participates; counts exclude padded geometry.
            vertex_count = (
                int(value.vertex_counts[0].item())
                if value.vertex_counts is not None
                else value.vertices.shape[1]
            )
            face_count = (
                int(value.face_counts[0].item())
                if value.face_counts is not None
                else value.faces.shape[1]
            )
            v = value.vertices[0, :vertex_count].cpu()
            vertices.append(v)
            faces.append(value.faces[0, :face_count].cpu() + offset)
            offset += v.shape[0]
            if any_uvs:
                uvs.append(
                    value.uvs[0, :vertex_count].cpu()
                    if value.uvs is not None
                    else v.new_zeros((v.shape[0], 2))
                )
            if any_colors:
                c = (
                    value.vertex_colors[0, :vertex_count].cpu()
                    if value.vertex_colors is not None
                    else v.new_ones((v.shape[0], color_channels))
                )
                if c.shape[-1] < color_channels:
                    c = torch.cat(
                        (c, c.new_ones((c.shape[0], color_channels - c.shape[-1]))), dim=-1
                    )
                colors.append(c)
            if value.texture is not None:
                if texture is None:
                    texture = value.texture.cpu()
                else:
                    logging.warning(
                        "MergeMeshes: dropping extra texture from input; only one texture is kept."
                    )
        return cls.outputs(
            mesh=TriangleMeshBatch(
                vertices=torch.cat(vertices, dim=0).unsqueeze(0),
                faces=torch.cat(faces, dim=0).unsqueeze(0),
                uvs=torch.cat(uvs, dim=0).unsqueeze(0) if any_uvs else None,
                vertex_colors=torch.cat(colors, dim=0).unsqueeze(0) if any_colors else None,
                texture=texture,
            )
        )


MESH_NODES = (RotateMesh, GetMeshInfo, MergeMeshes)
