"""Native Gaussian splat image and turntable rendering."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    COLOR_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LOAD3D_CAMERA_TYPE,
    MASK_TYPE,
    SPLAT_TYPE,
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
    materialize_image,
    materialize_splat,
    report_progress,
)


class RenderSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RenderSplat",
            display_name="Render Splat",
            category="3d/splat",
            search_terms=("splat to image", "render splat", "gaussian turntable"),
            description=(
                "Render a gaussian splat as an image with an anisotropic EWA rasterizer "
                "(oriented elliptical splats, antialiased, depth-sorted front-to-back). "
                "The camera comes from a camera_info input (Load / Preview 3D, or a Create "
                "Camera Info node); leave it empty to auto-frame the splat. Set frames "
                "greater than 1 for a turntable batch of images to feed a Video node."
            ),
            inputs=(
                InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=2048, step=8),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=2048, step=8),
                ),
                InputSpec(
                    "frames",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=-240, max=240),
                    doc=(
                        "-1, 0, 1 = single still image; >1 = turntable, the camera orbits over a "
                        "full 360 turn (works with any camera_info). Negative value orbits "
                        "the other way."
                    ),
                ),
                InputSpec(
                    "splat_scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.1, max=5.0, step=0.05),
                    doc=(
                        "Multiplier on each splat's projected footprint (lower = crisper points, "
                        "higher = softer/fuller surface)."
                    ),
                ),
                InputSpec(
                    "sharpen",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=2.0,
                    widget=NumberWidget(min=1.0, max=8.0, step=0.5),
                    doc=(
                        "Sharpen overlapping splats: 1.0 = physically-correct blend; higher biases "
                        "each pixel toward its dominant (nearest) splat for crisper texture, "
                        "without shrinking splats or opening gaps. Non-physical above 1."
                    ),
                ),
                InputSpec(
                    "headlight_shading",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=3.0, step=0.05),
                    doc=(
                        "Diffuse shading from a light at the camera (headlight), using the splat "
                        "surfel normals: darkens surfaces that turn away from view to reveal "
                        "form/curvature. 0 = flat albedo, 1 = strongest shading."
                    ),
                ),
                InputSpec(
                    "opacity_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Cull gaussians with opacity below this (removes faint floaters).",
                ),
                InputSpec(
                    "render_style",
                    TypeExpr.concrete(CORE_COMBO),
                    default="color",
                    widget=ComboWidget(
                        tuple(
                            ComboOption(mode, mode) for mode in ("color", "clay", "depth", "normal")
                        )
                    ),
                    doc=(
                        "What the image output shows: color, clay (neutral-albedo shaded), "
                        "depth (near=bright), normal (OpenGL normal map)."
                    ),
                ),
                InputSpec(
                    "background",
                    TypeExpr.concrete(COLOR_TYPE),
                    default="#000000",
                    widget=ColorWidget(),
                ),
                InputSpec(
                    "bg_image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc=(
                        "Optional background plate composited behind the splat (overrides the "
                        "solid background colour). Resized to the render size; a batch is used "
                        "per frame, "
                        "a single image for all. color/clay only."
                    ),
                ),
                InputSpec(
                    "camera_info",
                    TypeExpr.concrete(LOAD3D_CAMERA_TYPE),
                    required=False,
                    doc=(
                        "Camera to render from - a Load3D / Preview3D camera or a Create Camera "
                        "Info node. If empty, the splat is auto-framed from a default 3/4 view."
                    ),
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
        splat: object,
        width: int,
        height: int,
        frames: int,
        splat_scale: float,
        sharpen: float,
        headlight_shading: float,
        opacity_threshold: float,
        background: str,
        render_style: str,
        camera_info: object = None,
        bg_image: object = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        ops = importlib.import_module("dinkster_nodes_comfy.splat_render_ops")
        value = materialize_splat(splat)
        device = management.get_torch_device()
        color = background.lstrip("#")
        bg = (
            tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))
            if len(color) == 6
            else (0.0, 0.0, 0.0)
        )
        plates = None
        if bg_image is not None:
            utils = importlib.import_module("dinkster_inference.utils")
            plates = materialize_image(bg_image)[..., :3].movedim(-1, 1)
            plates = (
                utils.common_upscale(plates, width, height, "bicubic", "disabled")
                .movedim(1, -1)
                .clamp(0, 1)
            )
        frame_count = abs(int(frames)) or 1
        direction = -1.0 if frames < 0 else 1.0
        total = value.positions.shape[0] * frame_count
        images: list[Any] = []
        masks: list[Any] = []
        index = 0
        for batch in range(value.positions.shape[0]):
            count = (
                int(value.counts[batch].item())
                if value.counts is not None
                else value.positions.shape[1]
            )
            xyz, scale, rot = (
                tensor[batch, :count].to(device=device, dtype=torch.float32)
                for tensor in (value.positions, value.scales, value.rotations)
            )
            rgb = (
                value.sh[batch, :count, 0, :].to(device=device, dtype=torch.float32)
                * 0.28209479177387814
                + 0.5
            ).clamp(0, 1)
            opacity = (
                value.opacities[batch, :count].to(device=device, dtype=torch.float32).reshape(-1)
            )
            if opacity_threshold > 0:
                keep = opacity >= opacity_threshold
                xyz, rgb, opacity, scale, rot = (
                    tensor[keep] for tensor in (xyz, rgb, opacity, scale, rot)
                )
            camera = camera_info
            if camera is None:
                center = xyz.mean(0) if xyz.shape[0] else torch.zeros(3, device=device)
                extent = (
                    ops.quantile((xyz - center).norm(dim=-1), 0.99).clamp_min(1e-4)
                    if xyz.shape[0]
                    else torch.tensor(1.0, device=device)
                )
                distance = float(extent / (math.tan(math.radians(35) / 2) * 0.9))
                camera = ops.default_camera(center, distance, device)
            for frame in range(frame_count):
                current = (
                    camera
                    if frame_count == 1
                    else ops.orbit_camera(camera, direction * 360 * frame / frame_count, device)
                )
                plate = plates[index % plates.shape[0]] if plates is not None else bg
                image, mask = ops.render_gaussian(
                    xyz,
                    rgb,
                    opacity,
                    scale,
                    rot,
                    width,
                    height,
                    splat_scale,
                    plate,
                    current,
                    sharpen,
                    headlight_shading,
                    render_style,
                )
                images.append(image)
                masks.append(mask)
                index += 1
                if total > 1:
                    report_progress(index, total)
        return cls.outputs(IMAGE=torch.stack(images), MASK=torch.stack(masks))
