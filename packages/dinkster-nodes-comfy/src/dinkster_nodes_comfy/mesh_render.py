"""Single-view rendering of canonical mesh batches."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    COLOR_TYPE,
    CORE_COMBO,
    CORE_INT,
    IMAGE_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    MASK_TYPE,
    MESH_TYPE,
    ColorWidget,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_mesh,
)


class RenderMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RenderMesh",
            display_name="Render Mesh",
            search_terms=("mesh to image", "render mesh", "preview textured mesh"),
            category="3d/mesh",
            description=(
                "Ray-casts a single view of a mesh. The camera and optional model transform "
                "can come from a Load3D / Preview3D viewer."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "mode",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(
                        tuple(
                            ComboOption(mode, mode)
                            for mode in (
                                "auto",
                                "texture",
                                "vertex colors",
                                "solid",
                                "normal",
                                "depth",
                            )
                        )
                    ),
                    doc=(
                        "What to render. auto: texture if present, else vertex colours, "
                        "else shaded clay."
                    ),
                ),
                *(
                    InputSpec(
                        axis,
                        TypeExpr.concrete(CORE_INT),
                        default=1024,
                        widget=NumberWidget(min=64, max=4096, step=8),
                    )
                    for axis in ("width", "height")
                ),
                InputSpec(
                    "background",
                    TypeExpr.concrete(COLOR_TYPE),
                    default="#000000",
                    widget=ColorWidget(),
                ),
                InputSpec(
                    "model_3d_info",
                    TypeExpr.concrete(LOAD3D_MODEL_INFO_TYPE),
                    required=False,
                    doc="Model transform from the same Load3D / Preview3D viewer. "
                    "Connect it with camera_info to match the viewer framing.",
                ),
                InputSpec(
                    "camera_info",
                    TypeExpr.concrete(LOAD3D_CAMERA_TYPE),
                    required=False,
                    doc="Camera from a Load3D / Preview3D viewer or a Create Camera Info "
                    "node. If none is connected, a default front view is auto-framed.",
                ),
            ),
            outputs=(
                OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE), display_name="image"),
                OutputSpec("MASK", TypeExpr.concrete(MASK_TYPE), display_name="mask"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        mode: str,
        width: int,
        height: int,
        background: str,
        model_3d_info: Any = None,
        camera_info: Any = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        from .mesh_render_ops import build_bvh, camera_basis, model_transform, render_view

        mesh = materialize_mesh(mesh)
        if int(mesh.vertices.shape[0]) > 1:
            logging.warning(
                "RenderMesh: one item per batch only; using the first of %d.",
                int(mesh.vertices.shape[0]),
            )
        dev = management.get_torch_device()
        n = int(mesh.vertex_counts[0]) if mesh.vertex_counts is not None else mesh.vertices.shape[1]
        fcount = int(mesh.face_counts[0]) if mesh.face_counts is not None else mesh.faces.shape[1]
        vertices, faces = (
            mesh.vertices[0, :n].to(dev).float(),
            mesh.faces[0, :fcount].to(dev).long(),
        )
        transform = model_3d_info[0] if model_3d_info else None
        if transform is not None:
            vertices = model_transform(vertices, transform)

        def item(name: str) -> Any:
            value = getattr(mesh, name, None)
            if value is None:
                return None
            selected = cast(Any, value)[0] if isinstance(value, list) or value.ndim == 3 else value
            return selected[:n].to(dev).float()

        texture: Any = mesh.texture
        uvs: Any = mesh.uvs
        have_texture = texture is not None and uvs is not None
        have_colors = mesh.vertex_colors is not None
        resolved = mode
        if resolved == "auto":
            resolved = "texture" if have_texture else ("vertex colors" if have_colors else "solid")
        uv = texture_hwc = colors = normals = None
        render_normal = False
        if resolved == "texture" and texture is not None and uvs is not None:
            uv = (uvs[0, :n] if uvs.ndim == 3 else uvs[:n]).to(dev).float()
            texture_hwc = texture[0].to(dev).float()
        elif resolved == "vertex colors" and have_colors:
            colors = item("vertex_colors")[:, :3].clamp(0.0, 1.0).pow(1.0 / 2.2)
        elif resolved == "normal":
            render_normal = True
            normals = item("normals")
            if normals is not None and transform is not None:
                normals = model_transform(normals, transform, is_normal=True)
        tri = vertices[faces]
        bvh = build_bvh(tri)
        center = (vertices.amax(0) + vertices.amin(0)) * 0.5
        up_hint = torch.tensor([0.0, 1.0, 0.0], device=dev)
        if camera_info is not None:

            def vec(data: Any) -> Any:
                return torch.tensor(
                    [float(data.get(axis, 0.0)) for axis in ("x", "y", "z")], device=dev
                )

            eye = vec(camera_info.get("position", {}))
            target = camera_info.get("target")
            forward, right, up = camera_basis(eye, vec(target) if target else center, up_hint)
            fov = float(camera_info.get("fov", 0) or 0) or 40.0
            zoom = float(camera_info.get("zoom", 1.0) or 1.0)
            fov_rad = 2.0 * math.atan(math.tan(math.radians(fov) * 0.5) / max(zoom, 1e-3))
        else:
            fov_rad = math.radians(40.0)
            radius = float((vertices - center).norm(dim=-1).amax().clamp_min(1e-6))
            distance = radius / math.tan(fov_rad * 0.5) * 1.04
            eye = center + torch.tensor([0.0, 0.0, 1.0], device=dev) * distance
            forward, right, up = camera_basis(eye, center, up_hint)
        image, hit, depth = render_view(
            tri,
            bvh,
            uv,
            faces,
            texture_hwc,
            eye,
            forward,
            right,
            up,
            fov_rad,
            int(height),
            int(width),
            vertex_colors=colors,
            vertex_normals=normals,
            render_normal=render_normal,
        )
        if resolved == "depth":
            image = torch.zeros_like(image)
            if bool(hit.any()):
                dh = depth[hit]
                span = max(float(dh.max()) - float(dh.min()), 1e-6)
                normalized = ((float(dh.max()) - depth) / span).clamp(0.0, 1.0)
                image = torch.where(hit[..., None], normalized[..., None].expand(-1, -1, 3), image)
        bg = background.lstrip("#")
        background_rgb = torch.tensor(
            [int(bg[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], device=dev
        )
        output = torch.where(hit[..., None], image.clamp(0.0, 1.0), background_rgb)
        device, dtype = management.intermediate_device(), management.intermediate_dtype()
        return cls.outputs(
            IMAGE=output[None].to(device, dtype), MASK=hit.float()[None].to(device, dtype)
        )
