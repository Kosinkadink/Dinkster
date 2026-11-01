"""Depth Anything 3 loading, geometry prediction, visualization and meshing."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    DA3_GEOMETRY_TYPE,
    DA3_MODEL_TYPE,
    IMAGE_TYPE,
    MESH_TYPE,
    AssetRef,
    AssetWidget,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_vision_packet,
)


class LoadDA3Model(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadDA3Model",
            display_name="Load Depth Anything 3",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        kind="model/geometry-estimation", accept=("application/octet-stream",)
                    ),
                ),
                InputSpec(
                    "weight_dtype",
                    TypeExpr.concrete(CORE_COMBO),
                    default="default",
                    widget=ComboWidget(("default", "fp16", "bf16", "fp32")),
                ),
            ),
            outputs=(OutputSpec("DA3_MODEL", TypeExpr.concrete(DA3_MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_name: object, weight_dtype: str) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = str(model_name.local_path())
        initialize_inference_process()
        torch = importlib.import_module("torch")
        sd = importlib.import_module("dinkster_inference.sd")
        options: dict[str, object] = {}
        if weight_dtype == "fp16":
            options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            options["dtype"] = torch.float32
        return cls.outputs(DA3_MODEL=sd.load_diffusion_model(path, model_options=options))


class DA3Inference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DA3Inference",
            display_name="Run Depth Anything 3",
            category="image/geometry estimation",
            description="Run Depth Anything 3 on an image. In multi-view mode each image is "
            "treated as a separate view of the same scene.",
            search_terms=(
                "depth",
                "geometry",
                "da3",
                "depth anything",
                "monocular",
                "pointmap",
                "sky",
                "3d",
                "metric depth",
                "disparity",
            ),
            inputs=(
                InputSpec("da3_model", TypeExpr.concrete(DA3_MODEL_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=504,
                    widget=NumberWidget(min=140, max=2520, step=14),
                    doc="Model processing resolution; output returns to the original size.",
                ),
                InputSpec(
                    "resize_method",
                    TypeExpr.concrete(CORE_COMBO),
                    default="upper_bound_resize",
                    widget=ComboWidget(("upper_bound_resize", "lower_bound_resize")),
                    doc="Scale the longest side (upper) or shortest side (lower) to resolution.",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "mode",
                    (
                        DynamicComboOption("mono"),
                        DynamicComboOption(
                            "multiview",
                            (
                                InputSpec(
                                    "ref_view_strategy",
                                    TypeExpr.concrete(CORE_COMBO),
                                    default="saddle_balanced",
                                    widget=ComboWidget(
                                        ("saddle_balanced", "saddle_sim_range", "first", "middle")
                                    ),
                                ),
                                InputSpec(
                                    "pose_method",
                                    TypeExpr.concrete(CORE_COMBO),
                                    default="cam_dec",
                                    widget=ComboWidget(("cam_dec", "ray_pose")),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            input_order=("da3_model", "image", "resolution", "resize_method", "mode"),
            outputs=(OutputSpec("da3_geometry", TypeExpr.concrete(DA3_GEOMETRY_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        da3_model: Any,
        image: Any,
        resolution: int = 504,
        resize_method: str = "upper_bound_resize",
        mode: str = "mono",
        **inputs: Any,
    ) -> Mapping[str, object]:
        image = materialize_image(image)
        assert image.ndim == 4 and image.shape[-1] == 3, (
            f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"
        )
        diffusion = da3_model.model.diffusion_model
        if mode != "mono":
            pose_method = inputs["mode.pose_method"]
            has_camera = diffusion.cam_dec is not None
            has_dualdpt = diffusion.head_type == "dualdpt"
            if not has_camera and not has_dualdpt:
                raise ValueError(
                    "multi-view mode requires Small or Base model. The loaded model "
                    f"(head_type='{diffusion.head_type}') does not support cross-view "
                    "attention or camera pose estimation. Switch mode to 'mono', or "
                    "load Small or Base model for mult-view."
                )
            if pose_method == "cam_dec" and not has_camera:
                raise ValueError(
                    "pose_method='cam_dec' requires a camera decoder, but the loaded "
                    f"model (head_type='{diffusion.head_type}') does not have one. "
                    "Use pose_method='ray_pose' instead."
                )
            if pose_method == "ray_pose" and not has_dualdpt:
                raise ValueError(
                    "pose_method='ray_pose' requires a DualDPT head, but the loaded "
                    f"model has a '{diffusion.head_type}' head. Use pose_method='cam_dec' instead."
                )
        initialize_inference_process()
        torch = importlib.import_module("torch")
        mm = importlib.import_module("dinkster_inference.model_management")
        preprocess = importlib.import_module("dinkster_inference.ldm.depth_anything_3.preprocess")
        mm.load_model_gpu(da3_model)
        device = mm.get_torch_device()
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32
        batch, height, width, _ = image.shape
        geometry: dict[str, Any] = {"image": image.cpu(), "mode": mode}
        collected: dict[str, list[Any]] = {key: [] for key in ("depth", "confidence", "sky")}
        for start in range(batch if mode == "mono" else 1):
            pixels = image[start : start + 1] if mode == "mono" else image
            x = preprocess.preprocess_image(
                pixels.to(device), process_res=resolution, method=resize_method
            ).to(dtype=dtype)
            with torch.no_grad():
                output = (
                    diffusion(x)
                    if mode == "mono"
                    else diffusion(
                        x.unsqueeze(0),
                        use_ray_pose=inputs["mode.pose_method"] == "ray_pose",
                        ref_view_strategy=inputs["mode.ref_view_strategy"],
                    )
                )
            for source, target in (
                ("depth", "depth"),
                ("depth_conf", "confidence"),
                ("sky", "sky"),
            ):
                if source in output:
                    values = (
                        torch.nn.functional.interpolate(
                            output[source].unsqueeze(1).float(),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(1)
                        .cpu()
                    )
                    collected[target].append(values)
            if mode != "mono":
                if "extrinsics" in output and "intrinsics" in output:
                    geometry["extrinsics"] = output["extrinsics"].float().cpu().contiguous()
                    geometry["intrinsics"] = output["intrinsics"].float().cpu().contiguous()
                else:
                    geometry["extrinsics"] = torch.eye(4)[None, None].expand(1, batch, 4, 4).clone()
                    geometry["intrinsics"] = torch.eye(3)[None, None].expand(1, batch, 3, 3).clone()
        geometry["depth"] = torch.cat(collected.pop("depth"), dim=0).contiguous()
        for key, values in collected.items():
            if values:
                geometry[key] = torch.cat(values, dim=0).contiguous()
        return cls.outputs(da3_geometry=geometry)


def _normalize_confidence(confidence: Any, torch: Any) -> Any:
    values: list[Any] = []
    for frame in confidence:
        lo, hi = frame.min(), frame.max()
        values.append((frame - lo) / (hi - lo) if hi > lo else torch.ones_like(frame))
    return torch.stack(values)


class DA3Render(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        depth_inputs = (
            InputSpec(
                "normalization",
                TypeExpr.concrete(CORE_COMBO),
                default="v2_style",
                widget=ComboWidget(("v2_style", "min_max", "raw")),
            ),
            InputSpec("apply_sky_clip", TypeExpr.concrete(CORE_BOOLEAN), default=False),
        )
        colored = (InputSpec("colored", TypeExpr.concrete(CORE_BOOLEAN), default=False),)
        return NodeSchema(
            node_type="DA3Render",
            display_name="Render Depth Anything 3",
            category="image/geometry estimation",
            description="Render a depth map, confidence map, or sky mask from "
            "Depth Anything 3 geometry data.",
            inputs=(InputSpec("da3_geometry", TypeExpr.concrete(DA3_GEOMETRY_TYPE)),),
            combos=(
                DynamicComboSpec(
                    "output",
                    (
                        DynamicComboOption("depth", depth_inputs),
                        DynamicComboOption("depth_colored", depth_inputs),
                        DynamicComboOption("sky_mask", colored),
                        DynamicComboOption("confidence", colored),
                    ),
                ),
            ),
            input_order=("da3_geometry", "output"),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, da3_geometry: Any, output: str = "depth", **inputs: Any
    ) -> Mapping[str, object]:
        geometry = materialize_vision_packet(da3_geometry)
        initialize_inference_process()
        torch = importlib.import_module("torch")
        preprocess = importlib.import_module("dinkster_inference.ldm.depth_anything_3.preprocess")
        if output in ("depth", "depth_colored"):
            depth, sky = geometry["depth"], geometry.get("sky")
            if inputs["output.apply_sky_clip"]:
                if sky is None:
                    raise ValueError(
                        "apply_sky_clip=True requires a sky tensor in the da3_geometry input, "
                        "but none is present. Run with Mono/Metric models "
                        "or set apply_sky_clip=False."
                    )
                depth = torch.stack(
                    [preprocess.apply_sky_aware_clip(depth[i], sky[i]) for i in range(len(depth))]
                )
            normalization = inputs["output.normalization"]
            if normalization == "v2_style":
                result = torch.stack(
                    [
                        preprocess.normalize_depth_v2_style(
                            depth[i], sky[i] if sky is not None else None
                        )
                        for i in range(len(depth))
                    ]
                )
            elif normalization == "min_max":
                result = preprocess.normalize_depth_min_max(depth)
            else:
                result = depth
            if normalization != "raw":
                result = result.clamp(0.0, 1.0)
            colored = output == "depth_colored"
        elif output == "sky_mask":
            if "sky" not in geometry:
                raise ValueError("geometry has no sky output; run with Mono/Metric models.")
            result, colored = geometry["sky"], inputs["output.colored"]
        elif output == "confidence":
            if "confidence" not in geometry:
                raise ValueError(
                    "da3_geometry has no confidence output; run with Small/Base models."
                )
            result = _normalize_confidence(geometry["confidence"], torch)
            colored = inputs["output.colored"]
        else:
            raise ValueError(f"Unknown output mode: {output}")
        if colored:
            result = importlib.import_module("dinkster_inference.ldm.colormap").turbo(result)
        else:
            result = result.unsqueeze(-1).expand(*result.shape, 3).contiguous()
        return cls.outputs(IMAGE=result.float())


class DA3GeometryToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DA3GeometryToMesh",
            display_name="Convert DA3 Geometry to Mesh",
            description="Convert a depth map into a triangulated 3D mesh.",
            category="image/geometry estimation",
            search_terms=("da3", "depth anything", "mesh", "geometry", "3d", "triangulate"),
            inputs=(
                InputSpec("da3_geometry", TypeExpr.concrete(DA3_GEOMETRY_TYPE)),
                InputSpec(
                    "batch_index",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=4096),
                ),
                InputSpec(
                    "decimation",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=8),
                ),
                InputSpec(
                    "discontinuity_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.04,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "confidence_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.1,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec("use_sky_mask", TypeExpr.concrete(CORE_BOOLEAN), default=True),
                InputSpec("texture", TypeExpr.concrete(CORE_BOOLEAN), default=True),
            ),
            outputs=(OutputSpec("MESH", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        da3_geometry: Any,
        batch_index: int = 0,
        decimation: int = 1,
        discontinuity_threshold: float = 0.04,
        confidence_threshold: float = 0.1,
        use_sky_mask: bool = True,
        texture: bool = True,
    ) -> Mapping[str, object]:
        geometry = materialize_vision_packet(da3_geometry)
        depth_all = geometry["depth"]
        batch = depth_all.shape[0]
        if batch_index >= batch:
            raise ValueError(
                f"batch_index {batch_index} is out of range; DA3_GEOMETRY has batch size {batch}."
            )
        initialize_inference_process()
        torch = importlib.import_module("torch")
        depth = depth_all[batch_index].clone()
        depth[~torch.isfinite(depth)] = 0.0
        height, width = depth.shape
        if "intrinsics" in geometry:
            intrinsics = geometry["intrinsics"][0, batch_index].float()
        else:
            logging.getLogger(__name__).warning(
                "DA3_GEOMETRY has no intrinsics; using a ~60 degree FOV estimate."
            )
            focal = float(width) * 0.7
            intrinsics = torch.tensor(
                [
                    [focal, 0.0, (width - 1) / 2.0],
                    [0.0, focal, (height - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        u, v = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=depth.device),
            torch.arange(height, dtype=torch.float32, device=depth.device),
            indexing="xy",
        )
        pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        points = torch.einsum(
            "ij,hwj->hwi", torch.linalg.inv(intrinsics.to(depth.device)), pixels
        ) * depth.unsqueeze(-1)
        if "extrinsics" in geometry:
            extrinsic = geometry["extrinsics"][0, batch_index].to(points.device).float()
            if torch.isfinite(extrinsic).all():
                rotation = extrinsic[:3, :3].T
                points = (points.reshape(-1, 3) @ rotation.T - rotation @ extrinsic[:3, 3]).reshape(
                    height, width, 3
                )
            else:
                logging.getLogger(__name__).warning(
                    "DA3 extrinsics are non-finite; using camera-space coordinates."
                )
        mask = torch.ones(height, width, dtype=torch.bool)
        if use_sky_mask and "sky" in geometry:
            mask = mask & (geometry["sky"][batch_index] < 0.5)
        if "confidence" in geometry and confidence_threshold > 0.0:
            conf = _normalize_confidence(
                geometry["confidence"][batch_index : batch_index + 1], torch
            )[0]
            mask = mask & (conf >= confidence_threshold)
        mask = mask & (depth_all[batch_index] > 0) & torch.isfinite(depth_all[batch_index])
        points = points.clone()
        points[~mask] = float("inf")
        helper = importlib.import_module("dinkster_inference.ldm.moge.geometry")
        vertices, faces, uvs = helper.triangulate_grid_mesh(
            points,
            decimation=decimation,
            discontinuity_threshold=discontinuity_threshold,
            depth=depth,
        )
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError(
                "DA3GeometryToMesh produced an empty mesh. Try raising discontinuity_threshold, "
                "lowering confidence_threshold, or disabling use_sky_mask."
            )
        vertices = vertices * torch.tensor([1.0, -1.0, -1.0], dtype=vertices.dtype)
        faces = faces[:, [0, 2, 1]].contiguous()
        mesh = TriangleMeshBatch(
            vertices=vertices.unsqueeze(0),
            faces=faces.unsqueeze(0),
            uvs=uvs.unsqueeze(0),
            texture=geometry["image"][batch_index : batch_index + 1] if texture else None,
        )
        return cls.outputs(MESH=mesh)
