"""Pixal3D image features and projection conditioning."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_VIEW_AZIMUTHS = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}


def _encode_views(vision: Any, image: Any, out_device: Any) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    utils = importlib.import_module("dinkster_inference.utils")
    management.load_model_gpu(vision.patcher)
    device = management.get_torch_device()
    mean = torch.tensor(vision.image_mean or [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor(vision.image_std or [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    globals_by_size: dict[int, list[Any]] = {512: [], 1024: []}
    patches_by_size: dict[int, list[Any]] = {512: [], 1024: []}
    composites: list[Any] = []
    for pixels in image[..., :3]:
        item = pixels.movedim(-1, -3).unsqueeze(0).contiguous().float().clamp(0, 1)
        composites.append(item)
        for size in (512, 1024):
            resized = utils.common_upscale(item, size, size, "lanczos", "disabled").to(device)
            tokens = vision.model((resized - mean) / std, skip_norm_elementwise=True)[0]
            side = size // 16
            registers = tokens.shape[1] - 1 - side * side
            if registers < 0:
                raise ValueError(
                    f"_dinov3_patches_to_2d: got {tokens.shape[1]} tokens, expected "
                    f"1 (CLS) + N_reg + {side}*{side}={side * side} patches at image_size={size}, "
                    f"patch_size=16. Inferred N_reg={registers} which is invalid."
                )
            start = 1 + registers
            globals_by_size[size].append(tokens[:, :start].to(out_device))
            patches_by_size[size].append(
                tokens[:, start:]
                .transpose(1, 2)
                .reshape(tokens.shape[0], -1, side, side)
                .contiguous()
                .to(out_device)
            )
    return {
        "global_512": torch.cat(globals_by_size[512], dim=0),
        "global_1024": torch.cat(globals_by_size[1024], dim=0),
        "patches_512": torch.cat(patches_by_size[512], dim=0),
        "patches_1024": torch.cat(patches_by_size[1024], dim=0),
        "composites": composites,
    }


def _naf_features(
    naf: Any,
    features: Any,
    composites: list[Any],
    size: int,
    target: tuple[int, int],
    out_device: Any,
    device: Any,
) -> Any:
    if naf is None:
        return None
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    utils = importlib.import_module("dinkster_inference.utils")
    management.load_model_gpu(naf)
    inner = naf.model
    dtype = next(inner.parameters()).dtype
    output = torch.empty(
        (len(composites), features.shape[1], *target), device=out_device, dtype=dtype
    )
    for index, image in enumerate(composites):
        resized = utils.common_upscale(image, size, size, "lanczos", "disabled").to(device, dtype)
        inner(
            resized,
            features[index : index + 1].to(device, dtype),
            target,
            output=output[index : index + 1],
        )
    return output


def _build_conditioning(
    vision: Any,
    image: Any,
    transform: Any,
    angle: Any,
    mesh_scale: Any,
    num_views: int = 1,
) -> Mapping[str, object]:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    projection = importlib.import_module("dinkster_inference.ldm.trellis2.model")
    out_device, device = management.intermediate_device(), management.get_torch_device()
    features = _encode_views(vision, image, out_device)
    batch_size = image.shape[0] // num_views
    patches_512, patches_1024 = features["patches_512"], features["patches_1024"]
    composites = features["composites"]
    naf = vision.naf
    shape_512 = _naf_features(naf, patches_512, composites, 512, (512, 512), out_device, device)
    shape_1024 = _naf_features(naf, patches_1024, composites, 1024, (512, 512), out_device, device)
    texture_1024 = _naf_features(
        naf, patches_1024, composites, 1024, (1024, 1024), out_device, device
    )
    global_512 = features["global_512"].unflatten(0, (batch_size, num_views)).mean(dim=1)
    global_1024 = features["global_1024"].unflatten(0, (batch_size, num_views)).mean(dim=1)
    pack = {
        "stages": {
            "ss": {"feature_map": patches_512, "feature_map_hr": None, "image_resolution": 512},
            "shape_512": {
                "feature_map": patches_512,
                "feature_map_hr": shape_512,
                "image_resolution": 512,
            },
            "shape_1024": {
                "feature_map": patches_1024,
                "feature_map_hr": shape_1024,
                "image_resolution": 1024,
            },
            "tex_1024": {
                "feature_map": patches_1024,
                "feature_map_hr": texture_1024,
                "image_resolution": 1024,
            },
        },
        "transform_matrix": transform.to(out_device),
        "camera_angle_x": angle.to(out_device),
        "mesh_scale": mesh_scale.to(out_device),
        "num_views": num_views,
        "patch_size": 16,
    }
    projected = projection.compute_stage_proj_feats(
        pack,
        "ss",
        dense_grid_resolution=16,
        batch_size=batch_size,
        device=device,
    )
    return dict(
        positive=[
            [
                global_512,
                {"embeds": global_1024, "proj_feat_pack": pack, "trellis2_proj_feats": projected},
            ]
        ],
        negative=[
            [
                torch.zeros_like(global_512),
                {
                    "embeds": torch.zeros_like(global_1024),
                    "proj_feat_pack": pack,
                    "trellis2_proj_feats": projected,
                },
            ]
        ],
    )


class Pixal3DConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        return NodeSchema(
            node_type="Pixal3DConditioning",
            display_name="Pixal3D Conditioning",
            category="model/conditioning/trellis",
            inputs=(
                InputSpec(
                    "clip_vision_model",
                    TypeExpr.concrete(CLIP_VISION_TYPE),
                    doc="DINOv3 ViT-L/16 ClipVision.",
                ),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Preprocessed image from ImageCropToMask (pad_factor=1.1 for Pixal3D).",
                ),
                InputSpec(
                    "camera_angle_x",
                    TypeExpr.concrete(CORE_FLOAT),
                    display_name="fov",
                    default=49.13,
                    widget=NumberWidget(min=1.0, max=170.0, step=0.01),
                    doc=(
                        "Horizontal FOV in degrees. Wire a MoGeGeometryToFOV "
                        "(axis='horizontal', unit='degrees') for a per-image FoV "
                        "(matches upstream default)."
                    ),
                ),
            ),
            outputs=(OutputSpec("positive", conditioning), OutputSpec("negative", conditioning)),
        )

    @classmethod
    def execute(
        cls, *, clip_vision_model: Any, image: object, camera_angle_x: float
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        projection = importlib.import_module("dinkster_inference.ldm.trellis2.model")
        pixels = materialize_image(image)
        batch_size = pixels.shape[0]
        angle = math.radians(float(camera_angle_x))
        distance = torch.full((batch_size,), 0.5 / math.tan(angle / 2.0))
        transform = projection.build_proj_transform_matrix(distance, batch_size, distance.device)
        return _build_conditioning(
            clip_vision_model,
            pixels,
            transform,
            torch.full((batch_size,), angle),
            torch.ones(batch_size),
        )


class Pixal3DMultiViewConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        return NodeSchema(
            node_type="Pixal3DMultiViewConditioning",
            display_name="Pixal3D Multi-View Conditioning",
            category="model/conditioning/trellis",
            inputs=(
                InputSpec(
                    "clip_vision_model",
                    TypeExpr.concrete(CLIP_VISION_TYPE),
                    doc="DINOv3 ViT-L/16 ClipVision with bundled NAF weights.",
                ),
                InputSpec(
                    "fov",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=20.0,
                    widget=NumberWidget(min=1.0, max=170.0, step=0.01, round=False),
                    doc=(
                        "Horizontal FOV in degrees of the views as framed: 20 for rig renders "
                        "and most multi-view generators, or MoGeGeometryToFOV on one of the "
                        "views for photos."
                    ),
                ),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(IMAGE_TYPE),
                        required=False,
                        doc=(
                            f"Square view of the object's {name} side, with alpha or on a black "
                            "background, framed like the rig: the object spans about 1/1.1 of "
                            "the frame at its widest, the same scale in every view. The first "
                            "connected view (front, left, back, right order) is the front the "
                            "mesh is posed to."
                        ),
                    )
                    for name in _VIEW_AZIMUTHS
                ),
            ),
            outputs=(OutputSpec("positive", conditioning), OutputSpec("negative", conditioning)),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip_vision_model: Any,
        fov: float,
        front: object = None,
        left: object = None,
        back: object = None,
        right: object = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        views = {
            name: materialize_image(value)
            for name, value in (("front", front), ("left", left), ("back", back), ("right", right))
            if value is not None
        }
        names = list(views)
        if not names:
            raise ValueError("Pixal3DMultiViewConditioning needs at least one view")
        batch_size, num_views = views[names[0]].shape[0], len(names)
        if names[0] != "front":
            logging.warning(
                "Pixal3DMultiViewConditioning: no front view, the mesh will be posed with "
                "the %s view as its front",
                names[0],
            )
        items: list[Any] = []
        for batch in range(batch_size):
            for name in names:
                view = views[name][batch % views[name].shape[0]][None]
                if view.shape[-1] == 4:
                    view = view[..., :3] * view[..., 3:4]
                if view.shape[1:3] != (1024, 1024):
                    view = utils.common_upscale(
                        view.movedim(-1, 1), 1024, 1024, "lanczos", "disabled"
                    ).movedim(1, -1)
                items.append(view)
        angle = math.radians(fov)
        azimuth = torch.deg2rad(
            torch.tensor(
                [_VIEW_AZIMUTHS[name] - _VIEW_AZIMUTHS[names[0]] for name in names],
                dtype=torch.float32,
            )
        )
        elevation = torch.zeros_like(azimuth)
        backward = torch.stack(
            [
                torch.sin(azimuth) * torch.cos(elevation),
                -torch.cos(azimuth) * torch.cos(elevation),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        side = torch.stack(
            [torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)], dim=-1
        )
        transform = torch.eye(4).repeat(num_views, 1, 1)
        transform[:, :3, :3] = torch.stack(
            [side, torch.cross(backward, side, dim=-1), backward], dim=-1
        )
        transform[:, :3, 3] = backward * (1.1 * 0.5 / math.tan(angle / 2.0))
        return _build_conditioning(
            clip_vision_model,
            torch.cat(items, dim=0),
            transform.repeat(batch_size, 1, 1),
            torch.full((batch_size * num_views,), angle),
            torch.ones(batch_size),
            num_views=num_views,
        )
