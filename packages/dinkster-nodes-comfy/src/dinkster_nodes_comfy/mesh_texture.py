"""Extract baked mesh texture channels through the public mesh contract."""

from __future__ import annotations

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
