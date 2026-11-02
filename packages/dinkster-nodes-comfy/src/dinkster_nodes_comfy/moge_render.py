"""Depth, surface-normal and mask images from MoGe geometry."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    IMAGE_TYPE,
    MOGE_GEOMETRY_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)


def _normals_from_points(points: Any) -> Any:
    torch = importlib.import_module("torch")
    finite = torch.isfinite(points).all(dim=-1)
    pts = torch.where(finite.unsqueeze(-1), points, torch.zeros_like(points))
    dx = pts[..., :, 2:, :] - pts[..., :, :-2, :]
    dy = pts[..., 2:, :, :] - pts[..., :-2, :, :]
    dx = torch.nn.functional.pad(dx.permute(0, 3, 1, 2), (1, 1, 0, 0)).permute(0, 2, 3, 1)
    dy = torch.nn.functional.pad(dy.permute(0, 3, 1, 2), (0, 0, 1, 1)).permute(0, 2, 3, 1)
    # OpenCV's Y-down coordinates require dy x dx for outward-facing normals.
    normal = torch.nn.functional.normalize(torch.cross(dy, dx, dim=-1), dim=-1)
    return torch.where(finite.unsqueeze(-1), normal, torch.zeros_like(normal))


def _normalize_disparity(depth: Any) -> Any:
    torch = importlib.import_module("torch")
    out = torch.zeros_like(depth)
    for i in range(depth.shape[0]):
        value = depth[i]
        valid = torch.isfinite(value) & (value > 0)
        if not valid.any():
            continue
        disp = torch.where(valid, 1.0 / value.clamp_min(1e-6), torch.zeros_like(value))
        lo = torch.quantile(disp[valid], 0.001)
        hi = torch.quantile(disp[valid], 0.999)
        norm = ((disp - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        out[i] = torch.where(valid, norm, torch.zeros_like(norm))
    return out


class MoGeRender(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MoGeRender",
            display_name="Render MoGe Geometry",
            description="Render a depth map or normal map from geometry data",
            category="image/geometry estimation",
            search_terms=("moge", "render", "geometry", "depth", "normal"),
            inputs=(
                InputSpec("moge_geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
                InputSpec(
                    "output",
                    TypeExpr.concrete(CORE_COMBO),
                    default="depth",
                    widget=ComboWidget(
                        options=(
                            "depth",
                            "depth_colored",
                            "normal_opengl",
                            "normal_directx",
                            "mask",
                        )
                    ),
                    doc=(
                        "DirectX vs OpenGL controls the normal-map green-channel convention. "
                        "DirectX: green = -Y down (Unreal). OpenGL: green = +Y up "
                        "(Blender, Substance, Unity, glTF)."
                    ),
                ),
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, moge_geometry: Any, output: str = "depth") -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        geometry = materialize_latent(moge_geometry)
        is_normal = output in ("normal_directx", "normal_opengl")
        if output in ("depth", "depth_colored"):
            if "depth" not in geometry:
                raise ValueError("moge_geometry has no depth output.")
            source = geometry["depth"]
        elif is_normal:
            if "normal" in geometry:
                source = geometry["normal"]
            elif "points" in geometry:
                source = geometry["points"]
            else:
                raise ValueError(
                    "moge_geometry has neither normals nor points to derive normals from."
                )
        elif output == "mask":
            if "mask" not in geometry:
                raise ValueError("moge_geometry has no mask output.")
            source = geometry["mask"]
        else:
            raise ValueError(f"Unknown output mode: {output}")

        images: list[Any] = []
        for i in range(source.shape[0]):
            value = source[i : i + 1].float()
            if output in ("depth", "depth_colored"):
                disparity = _normalize_disparity(value)
                if output == "depth_colored":
                    colormap = importlib.import_module("dinkster_inference.ldm.colormap")
                    images.append(colormap.turbo(disparity))
                else:
                    images.append(disparity.unsqueeze(-1).expand(*disparity.shape, 3).contiguous())
            elif is_normal:
                normal = value if "normal" in geometry else _normals_from_points(value)
                normal = normal * normal.new_tensor(
                    [1.0, -1.0 if output.endswith("_opengl") else 1.0, -1.0]
                )
                images.append((normal * 0.5 + 0.5).clamp(0.0, 1.0))
            else:
                images.append(value.unsqueeze(-1).expand(*value.shape, 3).contiguous())
        return {
            "IMAGE": torch.cat(images, dim=0).to(
                device=management.intermediate_device(), dtype=management.intermediate_dtype()
            )
        }
