"""Bake high-poly occlusion into an existing low-poly UV layout."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_mesh,
    report_progress,
)


class BakeAmbientOcclusion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BakeAmbientOcclusion",
            display_name="Bake Ambient Occlusion",
            category="3d/texturing",
            description=(
                "Bakes an ambient-occlusion map from a high-poly mesh into a low-poly's UV "
                "layout (white = open, dark = crevices). Feed the UV-unwrapped low_poly and the "
                "high_poly it was decimated from. Outputs a grayscale image for "
                "Apply Texture To Mesh's occlusion input "
                "(packed into the ORM map / occlusionTexture)."
            ),
            inputs=(
                InputSpec("low_poly", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("high_poly", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=8192, step=64),
                ),
                InputSpec(
                    "samples",
                    TypeExpr.concrete(CORE_INT),
                    default=64,
                    widget=NumberWidget(min=4, max=1024, step=4),
                    doc="Rays per texel. More = smoother, slower. Raise if grainy.",
                ),
                InputSpec(
                    "max_distance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.5,
                    widget=NumberWidget(min=0.01, max=2.0, step=0.01),
                    doc="Ray length, as a fraction of the bounding box diagonal. "
                    "Smaller = tighter, more local occlusion.",
                ),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=2.0, step=0.05),
                    doc="Scales the occlusion. >1 darkens, <1 lightens.",
                ),
                InputSpec(
                    "bias",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.01,
                    widget=NumberWidget(min=0.0001, max=0.2, step=0.0005),
                    doc="Ray origin lift off the surface, as a fraction of the bounding box "
                    "diagonal. Raise if even surfaces show dark blotches/holes.",
                ),
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE), display_name="occlusion"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int,
        samples: int,
        max_distance: float,
        strength: float,
        bias: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        from .mesh_bake_ops import bake_occlusion, vertex_normals
        from .mesh_texture import normalize_uvs

        low, high = materialize_mesh(low_poly), materialize_mesh(high_poly)
        if low.uvs is None:
            raise ValueError(
                "BakeAmbientOcclusion: low_poly has no UVs. Connect the UV-unwrapped low-poly "
                "(the same one used for the other bakes); this node never unwraps."
            )
        device = management.get_torch_device()
        batch = int(low.vertices.shape[0])
        high_batch = int(high.vertices.shape[0])
        images: list[Any] = []
        for i in range(batch):
            start, end = (1000 * i) // batch, (1000 * (i + 1)) // batch
            n = (
                int(low.vertex_counts[i])
                if low.vertex_counts is not None
                else low.vertices.shape[1]
            )
            f = int(low.face_counts[i]) if low.face_counts is not None else low.faces.shape[1]
            if f == 0:
                logging.warning("BakeAmbientOcclusion: skipping batch %d (empty mesh)", i)
                images.append(torch.ones((int(resolution), int(resolution), 3)))
                report_progress(end, 1000)
                continue
            vertices, faces = (
                low.vertices[i, :n].to(device).float(),
                low.faces[i, :f].to(device).long(),
            )
            uv = low.uvs[i, :n] if low.uvs.ndim == 3 else low.uvs[:n]
            uv = torch.from_numpy(normalize_uvs(uv.detach().cpu().numpy())).to(device)
            normals = (
                low.normals[i, :n].to(device).float()
                if low.normals is not None
                else vertex_normals(vertices, faces)
            )
            hi = i if high_batch > 1 else 0
            hn = (
                int(high.vertex_counts[hi])
                if high.vertex_counts is not None
                else high.vertices.shape[1]
            )
            hf = int(high.face_counts[hi]) if high.face_counts is not None else high.faces.shape[1]
            image = bake_occlusion(
                high.vertices[hi, :hn].to(device).float(),
                high.faces[hi, :hf].to(device).long(),
                vertices,
                faces,
                uv,
                normals,
                int(resolution),
                int(samples),
                float(max_distance),
                float(strength),
                float(bias),
                start,
                end,
            )
            images.append(image.float().cpu())
            report_progress(end, 1000)
        return cls.outputs(IMAGE=torch.stack([image.clamp(0.0, 1.0) for image in images]))


class BakeNormalMapFromMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BakeNormalMapFromMesh",
            display_name="Bake Normal Map from Mesh",
            category="3d/texturing",
            description=(
                "Bakes a tangent-space normal map (glTF/OpenGL +Y) from a high-poly mesh into a "
                "low-poly's UV layout, capturing detail lost during decimation. "
                "Feed the UV-unwrapped "
                "low_poly and the high_poly it was decimated from. Outputs an image for "
                "Apply Texture To Mesh's normal_map input."
            ),
            inputs=(
                InputSpec("low_poly", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("high_poly", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=8192, step=64),
                    display_name="resolution",
                ),
                InputSpec(
                    "cage_distance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.05,
                    widget=NumberWidget(min=0.001, max=0.5, step=0.001),
                    doc="Surface search band, as a fraction of the bounding box diagonal. "
                    "Raise for wrong/missing patches under heavy decimation; "
                    "lower if it grabs across gaps.",
                ),
                InputSpec(
                    "ignore_backfaces",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    doc="Skip high-poly surfaces facing away from the texel, so "
                    "crevices/enclosed spaces don't grab the opposite wall. "
                    "Disable only if the high-poly winding is inconsistent.",
                ),
            ),
            outputs=(
                OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE), display_name="normal_map"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int,
        cage_distance: float = 0.05,
        ignore_backfaces: bool = True,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        from .mesh_bake_ops import bake_normal_map, vertex_normals
        from .mesh_texture import normalize_uvs, normals_and_tangents

        low, high = materialize_mesh(low_poly), materialize_mesh(high_poly)
        if low.uvs is None:
            raise ValueError(
                "BakeNormalMapFromMesh: low_poly has no UVs. Connect the UV-unwrapped "
                "low-poly (the same one you fed to BakeTextureFromVoxel); this node bakes "
                "onto existing UVs and never unwraps."
            )
        device = management.get_torch_device()
        images: list[Any] = []
        batch, high_batch = int(low.vertices.shape[0]), int(high.vertices.shape[0])
        for i in range(batch):
            n = (
                int(low.vertex_counts[i])
                if low.vertex_counts is not None
                else low.vertices.shape[1]
            )
            f = int(low.face_counts[i]) if low.face_counts is not None else low.faces.shape[1]
            if f == 0:
                logging.warning("BakeNormalMapFromMesh: skipping batch %d (empty mesh)", i)
                images.append(torch.full((int(resolution), int(resolution), 3), 0.5))
                report_progress(i + 1, batch)
                continue
            vertices = low.vertices[i, :n].to(device).float()
            faces = low.faces[i, :f].to(device).long()
            uv = low.uvs[i, :n] if low.uvs.ndim == 3 else low.uvs[:n]
            uv = torch.from_numpy(normalize_uvs(uv.detach().cpu().numpy())).to(device)
            low_normals = low.normals[i, :n].to(device).float() if low.normals is not None else None
            normals, tangents = normals_and_tangents(vertices, faces, uv, low_normals)
            hi = i if high_batch > 1 else 0
            hn = (
                int(high.vertex_counts[hi])
                if high.vertex_counts is not None
                else high.vertices.shape[1]
            )
            hf = int(high.face_counts[hi]) if high.face_counts is not None else high.faces.shape[1]
            high_vertices = high.vertices[hi, :hn].to(device).float()
            high_faces = high.faces[hi, :hf].to(device).long()
            high_normals = (
                high.normals[hi, :hn].to(device).float()
                if high.normals is not None
                else vertex_normals(high_vertices, high_faces)
            )
            image = bake_normal_map(
                high_vertices,
                high_faces,
                high_normals,
                vertices,
                faces,
                uv,
                normals,
                tangents,
                int(resolution),
                float(cage_distance),
                bool(ignore_backfaces),
            )
            images.append(image.float().cpu())
            report_progress(i + 1, batch)
        return cls.outputs(IMAGE=torch.stack([image.clamp(0.0, 1.0) for image in images]))
