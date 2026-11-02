"""Attach and extract baked textures through the public mesh contract."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    IMAGE_TYPE,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_mesh,
)


class MeshTextureToImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MeshTextureToImage",
            display_name="Mesh Texture to Image",
            category="3d/texturing",
            description=(
                "Extracts a mesh's baked textures as individual IMAGEs: base_color, metallic, "
                "roughness, occlusion and normal_map. Channels with nothing baked come back "
                "neutral (occlusion white, normal flat)."
            ),
            inputs=(InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(IMAGE_TYPE))
                for name in ("base_color", "metallic", "roughness", "occlusion", "normal_map")
            ),
        )

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        source = materialize_mesh(mesh)

        def as_image(texture: Any) -> Any:
            if texture is None:
                return None
            image = texture.float().clamp(0.0, 1.0).cpu()
            return image.unsqueeze(0) if image.ndim == 3 else image

        base = as_image(source.texture)
        orm = as_image(source.metallic_roughness)
        normal = as_image(source.normal_map)
        if base is None:
            raise ValueError(
                "MeshTextureToImage: mesh has no baseColor texture. Run "
                "BakeTextureFromVoxel first (PaintMesh only sets vertex colors, not a texture)."
            )
        if orm is None:
            orm = torch.zeros_like(base)
        if normal is None:
            normal = torch.ones_like(base) * torch.tensor([0.5, 0.5, 1.0])
        return cls.outputs(
            base_color=base,
            metallic=orm[..., 2:3].expand(-1, -1, -1, 3).contiguous(),
            roughness=orm[..., 1:2].expand(-1, -1, -1, 3).contiguous(),
            occlusion=(
                orm[..., 0:1].expand(-1, -1, -1, 3).contiguous()
                if source.occlusion_in_mr
                else torch.ones_like(base)
            ),
            normal_map=normal,
        )


def normalize_uvs(uvs: Any) -> Any:
    np = importlib.import_module("numpy")
    values = uvs.astype(np.float32)
    lo, hi = values.min(axis=0), values.max(axis=0)
    if lo.min() < -1e-4 or hi.max() > 1.0001:
        extent = float((hi - lo).max())
        if extent > 0:
            values = ((values - lo) / extent).astype(np.float32)
    return values


def normals_and_tangents(vertices: Any, faces: Any, uvs: Any, normals: Any) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    if normals is None:
        face_normals = torch.cross(vertices[i1] - vertices[i0], vertices[i2] - vertices[i0], dim=-1)
        lo = vertices.min(0).values
        inverse = 1.0 / (float((vertices.max(0).values - lo).max().clamp_min(1e-9)) * 1e-5)
        quantized = ((vertices - lo) * inverse).round().to(torch.int64)
        _, groups = torch.unique(quantized, dim=0, return_inverse=True)
        accumulated = vertices.new_zeros((int(groups.max()) + 1, 3))
        for indices in (i0, i1, i2):
            accumulated.scatter_add_(
                0, groups[indices].unsqueeze(-1).expand_as(face_normals), face_normals
            )
        normals = torch.nn.functional.normalize(accumulated[groups], p=2, dim=-1, eps=1e-6)
    e1, e2 = vertices[i1] - vertices[i0], vertices[i2] - vertices[i0]
    d1, d2 = uvs[i1] - uvs[i0], uvs[i2] - uvs[i0]
    denominator = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    inverse = 1.0 / torch.where(
        denominator.abs() < 1e-20, torch.full_like(denominator, 1e-20), denominator
    )
    tangent = (d2[:, 1:2] * e1 - d1[:, 1:2] * e2) * inverse[:, None]
    bitangent = (d1[:, 0:1] * e2 - d2[:, 0:1] * e1) * inverse[:, None]
    tangent_sum, bitangent_sum = torch.zeros_like(vertices), torch.zeros_like(vertices)
    for indices in (i0, i1, i2):
        tangent_sum.scatter_add_(0, indices[:, None].expand(-1, 3), tangent)
        bitangent_sum.scatter_add_(0, indices[:, None].expand(-1, 3), bitangent)
    unit_normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    tangent = torch.nn.functional.normalize(
        tangent_sum - unit_normals * (unit_normals * tangent_sum).sum(-1, keepdim=True),
        dim=-1,
        eps=1e-6,
    )
    handedness = torch.sign((torch.cross(unit_normals, tangent, dim=-1) * bitangent_sum).sum(-1))
    handedness = torch.where(handedness == 0, torch.ones_like(handedness), handedness)
    return normals, torch.cat([tangent, handedness[:, None]], dim=-1)


class ApplyTextureToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ApplyTextureToMesh",
            display_name="Apply Texture to Mesh",
            category="3d/texturing",
            description=(
                "Attaches baked texture IMAGEs to a mesh's UV layout for SaveGLB. "
                "Feed the SAME mesh you baked"
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("base_color", TypeExpr.concrete(IMAGE_TYPE)),
                *(
                    InputSpec(name, TypeExpr.concrete(IMAGE_TYPE), required=False)
                    for name in ("metallic", "roughness", "occlusion", "normal_map")
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        base_color: object,
        metallic: object | None = None,
        roughness: object | None = None,
        occlusion: object | None = None,
        normal_map: object | None = None,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        source = materialize_mesh(mesh)
        uvs = source.uvs
        if uvs is None:
            raise ValueError(
                "ApplyTextureToMesh: mesh has no UVs. Connect the same UV-unwrapped mesh "
                "you fed to BakeTextureFromVoxel."
            )
        normalized = uvs.clone()
        if uvs.ndim == 3:
            for index in range(uvs.shape[0]):
                count = (
                    int(source.vertex_counts[index])
                    if source.vertex_counts is not None
                    else source.vertices.shape[1]
                )
                values = normalize_uvs(uvs[index, :count].detach().cpu().numpy())
                normalized[index, :count] = torch.from_numpy(values).to(normalized)
        else:
            normalized = torch.from_numpy(normalize_uvs(uvs.detach().cpu().numpy())).to(uvs)
        out: Any = copy.copy(source)
        out.uvs = normalized
        out.texture = materialize_image(base_color).float().clamp(0.0, 1.0).cpu()
        if normal_map is not None:
            initialize_inference_process()
            device = importlib.import_module(
                "dinkster_inference.model_management"
            ).get_torch_device()
            batch = int(source.vertices.shape[0])
            maximum = (
                int(source.vertices.shape[1])
                if source.vertices.ndim == 3
                else int(source.vertices.shape[0])
            )
            out.tangents = torch.zeros((batch, maximum, 4), dtype=torch.float32)
            out.normals = torch.zeros((batch, maximum, 3), dtype=torch.float32)
            for index in range(batch):
                nv = (
                    int(source.vertex_counts[index])
                    if source.vertex_counts is not None
                    else source.vertices.shape[1]
                )
                nf = (
                    int(source.face_counts[index])
                    if source.face_counts is not None
                    else source.faces.shape[1]
                )
                vertices, faces = source.vertices[index, :nv], source.faces[index, :nf]
                if faces.numel() == 0:
                    continue
                uv = normalized[index, :nv] if normalized.ndim == 3 else normalized[:nv]
                normals = (
                    source.normals[index, :nv].to(device).float()
                    if source.normals is not None
                    else None
                )
                normals, tangents = normals_and_tangents(
                    vertices.to(device).float(),
                    faces.to(device).long(),
                    uv.to(device).float(),
                    normals,
                )
                out.tangents[index, :nv] = tangents.cpu()
                out.normals[index, :nv] = normals.cpu()
            out.normal_map = materialize_image(normal_map).float().clamp(0.0, 1.0).cpu()
        maps = tuple(
            materialize_image(value) if value is not None else None
            for value in (metallic, roughness, occlusion)
        )
        provided = [value for value in maps if value is not None]
        if provided:
            batch = int(provided[0].shape[0])
            height = max(int(value.shape[1]) for value in provided)
            width = max(int(value.shape[2]) for value in provided)

            def channel(image: Any, default: float) -> Any:
                if image is None:
                    return torch.full((batch, height, width, 1), default)
                value = image[..., 0:1].float().clamp(0.0, 1.0).cpu()
                if int(value.shape[1]) != height or int(value.shape[2]) != width:
                    value = torch.nn.functional.interpolate(
                        value.permute(0, 3, 1, 2),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                return value

            out.metallic_roughness = torch.cat(
                [channel(maps[2], 1.0), channel(maps[1], 1.0), channel(maps[0], 0.0)], dim=-1
            )
            if occlusion is not None:
                out.occlusion_in_mr = True
        return cls.outputs(mesh=out)
