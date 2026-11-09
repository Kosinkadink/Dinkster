"""Monocular geometry from perspective splits of an equirectangular image."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    MOGE_GEOMETRY_TYPE,
    MOGE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    report_progress,
)


class MoGePanoramaInference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MoGePanoramaInference",
            display_name="Run MoGe Panorama Inference",
            category="image/geometry estimation",
            description=(
                "Run MoGe on an equirectangular panorama by splitting it into 12 perspective "
                "views, running inference on each, and merging the results into a single depth map."
            ),
            search_terms=(
                "moge",
                "panorama",
                "depth",
                "geometry",
                "depth estimation",
                "geometry estimation",
            ),
            inputs=(
                InputSpec("moge_model", TypeExpr.concrete(MOGE_MODEL_TYPE)),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Equirectangular panorama (any aspect).",
                ),
                InputSpec(
                    "resolution_level",
                    TypeExpr.concrete(CORE_INT),
                    default=9,
                    widget=NumberWidget(min=0, max=9),
                    doc="Per-view detail (0 = fastest, 9 = most detailed).",
                ),
                InputSpec(
                    "split_resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=512,
                    widget=NumberWidget(min=256, max=1024),
                    doc="Resolution of each perspective split.",
                ),
                InputSpec(
                    "merge_resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=1920,
                    widget=NumberWidget(min=256, max=8192),
                    doc="Long-side resolution of the merged equirect distance map.",
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=4,
                    widget=NumberWidget(min=1, max=12),
                    doc="Views per inference batch (12 splits total).",
                ),
                InputSpec(
                    "refine_steps",
                    TypeExpr.concrete(CORE_INT),
                    default=3,
                    advanced=True,
                    widget=NumberWidget(min=0, max=8),
                    doc=(
                        "MoGe-3 only: sparse volumetric refinement passes over the predicted "
                        "depth. "
                        "More passes sharpen fine detail and edges at a roughly linear cost. "
                        "0 disables refinement. Ignored by MoGe-1 / MoGe-2."
                    ),
                ),
            ),
            outputs=(OutputSpec("moge_geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        moge_model: Any,
        image: Any,
        resolution_level: int = 9,
        split_resolution: int = 512,
        merge_resolution: int = 1920,
        batch_size: int = 4,
        refine_steps: int = 3,
    ) -> Mapping[str, object]:
        image = materialize_image(image)
        if image.shape[0] != 1:
            raise ValueError(
                f"MoGePanoramaInference takes a single image (got batch of {image.shape[0]})"
            )
        initialize_inference_process()
        torch = importlib.import_module("torch")
        panorama = importlib.import_module("dinkster_inference.ldm.moge.panorama")
        management = importlib.import_module("dinkster_inference.model_management")
        image = image[..., :3]
        height, width = int(image.shape[1]), int(image.shape[2])
        scale = min(merge_resolution / max(height, width), 1.0)
        merge_h, merge_w = max(int(height * scale), 32), max(int(width * scale), 32)
        extrinsics, intrinsics = panorama.get_panorama_cameras()
        management.load_model_gpu(moge_model.patcher)
        pixels = image[0].movedim(-1, -3).to(device=moge_model.load_device, dtype=moge_model.dtype)
        splits = panorama.split_panorama_image(pixels, extrinsics, intrinsics, split_resolution)
        views = splits.shape[0]
        levels: list[tuple[int, int]] = []
        level_w, level_h = merge_w, merge_h
        while True:
            levels.append((level_w, level_h))
            if max(level_w, level_h) <= 256:
                break
            level_w, level_h = level_w // 2, level_h // 2
        levels.reverse()
        weights = {size: 4**index for index, size in enumerate(levels)}
        total = views + views * len(levels) + sum(weights.values())
        done = 0
        distance_maps: list[Any] = []
        masks: list[Any] = []
        for index in range(0, views, batch_size):
            batch = splits[index : index + batch_size]
            result = moge_model.infer(
                batch,
                resolution_level=resolution_level,
                fov_x=90.0,
                force_projection=True,
                apply_mask=False,
                apply_metric_scale=False,
                refine_steps=refine_steps,
            )
            distance_maps.extend(list(result["points"].float().norm(dim=-1).cpu().numpy()))
            masks.extend(list(result["mask"].cpu().numpy()))
            done += batch.shape[0]
            report_progress(done, total)

        def on_view() -> None:
            nonlocal done
            done += 1
            report_progress(done, total)

        def on_solve_start(w: int, h: int) -> None:
            report_progress(done, total, text=f"MoGe panorama merge: solving {w}x{h}")

        def on_solve_end(w: int, h: int) -> None:
            nonlocal done
            done += weights[(w, h)]
            report_progress(done, total)

        depth_array, mask_array = panorama.merge_panorama_depth(
            merge_w,
            merge_h,
            distance_maps,
            masks,
            list(extrinsics),
            intrinsics,
            on_view=on_view,
            on_solve_start=on_solve_start,
            on_solve_end=on_solve_end,
        )
        depth = torch.from_numpy(depth_array)
        mask = torch.from_numpy(mask_array)
        if (merge_h, merge_w) != (height, width):
            depth = torch.nn.functional.interpolate(
                depth[None, None], size=(height, width), mode="bilinear", align_corners=False
            ).squeeze()
            mask = (
                torch.nn.functional.interpolate(
                    mask[None, None].float(), size=(height, width), mode="nearest"
                ).squeeze()
                > 0
            )
        if mask.any() and not mask.all():
            far = torch.quantile(depth[mask], 0.95) * 5.0
            depth = torch.where(mask, depth, far)
        directions = torch.from_numpy(
            panorama.spherical_uv_to_directions(panorama._uv_grid(height, width))
        )
        geometry = {
            "points": (directions * depth[..., None]).unsqueeze(0),
            "depth": depth.unsqueeze(0),
            "mask": mask.unsqueeze(0),
            "image": image.cpu(),
        }
        return {"moge_geometry": geometry}
