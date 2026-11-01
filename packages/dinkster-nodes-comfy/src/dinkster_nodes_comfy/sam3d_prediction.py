"""Batched SAM3D Body prediction with tracking or detector crops."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MHR_POSE_DATA_TYPE,
    SAM3_TRACK_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_vision_packet,
    report_progress,
)


class SAM3DBody_Predict(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3DBody_Predict",
            display_name="Run SAM3D Body Prediction",
            category="image/detection",
            inputs=(
                InputSpec("sam3d_body_model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "track_data",
                    TypeExpr.concrete(SAM3_TRACK_DATA_TYPE),
                    required=False,
                    doc="Tracking data from SAM3 Video Track, required for multi-person detection",
                ),
                InputSpec(
                    "bboxes",
                    TypeExpr.concrete(BOUNDING_BOX_TYPE),
                    required=False,
                    default={"x": 0, "y": 0, "width": 512, "height": 512},
                    force_input=True,
                    doc="Per-frame bounding boxes used for better detection. Can be used as an "
                    "alternative to tracking data. ",
                ),
                InputSpec(
                    "run_hand_refinement",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    doc="Improves hand pose at the cost of extra inference time and memory use",
                ),
                InputSpec(
                    "fov",
                    TypeExpr.concrete(CORE_FLOAT),
                    advanced=True,
                    doc="Vertical FoV in degrees. Affects predicted depth and absolute scale. "
                    "0 = fall back to ~53 degrees (16:9).",
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=64,
                    widget=NumberWidget(min=1, max=512, step=1),
                    advanced=True,
                    doc="Max person crops to process as a batch. Larger values use more VRAM "
                    "for faster inference.",
                ),
            ),
            outputs=(OutputSpec("mhr_pose_data", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        sam3d_body_model: Any,
        image: Any,
        track_data: Any = None,
        bboxes: Any = None,
        run_hand_refinement: bool = True,
        fov: float = 0.0,
        batch_size: int = 64,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        image = materialize_image(image)
        inner = sam3d_body_model.model
        count, height, width, _ = image.shape
        boxes, masks = (None, None)
        if track_data is not None:
            boxes, masks = _tracking_inputs(
                materialize_vision_packet(track_data), count, height, width
            )
        if boxes is None and bboxes:
            boxes = _detection_boxes(bboxes, count)
            masks = None
        if boxes is None:
            full = torch.tensor([[0.0, 0.0, float(width), float(height)]], dtype=torch.float32)
            boxes = [full.clone() for _ in range(count)]
            masks = None
        camera = None
        if fov > 0:
            focal = height / (2.0 * math.tan(math.radians(fov) / 2.0))
            camera = torch.tensor(
                [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            )
        frames = [
            (image[index] * 255.0).clamp(0.0, 255.0).to(dtype=torch.uint8, device="cpu")
            if len(boxes[index])
            else None
            for index in range(count)
        ]
        groups: dict[int, list[int]] = {}
        for index, frame in enumerate(frames):
            people = len(boxes[index])
            if frame is not None and people:
                groups.setdefault(people, []).append(index)
        maximum = max(
            (
                min(len(indices), max(1, int(batch_size) // people)) * people
                for people, indices in groups.items()
            ),
            default=1,
        )
        management.load_models_gpu(
            [sam3d_body_model],
            memory_required=inner.memory_used_forward(maximum, run_hand_refinement),
        )
        output: list[list[dict[str, Any]]] = [[] for _ in range(count)]
        completed = count - sum(len(indices) for indices in groups.values())
        report_progress(completed, count)
        for people, indices in groups.items():
            chunk_frames = max(1, int(batch_size) // people)
            for start in range(0, len(indices), chunk_frames):
                chunk = indices[start : start + chunk_frames]
                result = _predict_chunk(
                    inner,
                    [frames[index] for index in chunk],
                    [boxes[index] for index in chunk],
                    [masks[index] for index in chunk] if masks is not None else None,
                    "full" if run_hand_refinement else "body",
                    camera,
                )
                for index, persons in zip(chunk, result, strict=True):
                    output[index] = persons
                completed += len(chunk)
                report_progress(completed, count)
        return cls.outputs(
            mhr_pose_data={
                "frames": output,
                "faces": inner.head_pose.faces_np(),
                "image_size": (int(height), int(width)),
                "canonical_colors": _canonical_colors(inner),
                "hand_vert_mask": _hand_mask(inner),
            }
        )


def _detection_boxes(bboxes: Any, count: int) -> list[Any]:
    torch = importlib.import_module("torch")
    normalized: list[Any]
    if isinstance(bboxes, dict):
        normalized = [[cast(dict[str, Any], bboxes)]]
    elif isinstance(bboxes[0], dict):
        normalized = [bboxes]
    else:
        normalized = list(bboxes)
    if len(normalized) == 1:
        normalized *= count
    normalized = (normalized + [[]] * count)[:count]
    return [
        torch.tensor(
            [[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in frame],
            dtype=torch.float32,
        )
        if frame
        else torch.zeros((0, 4), dtype=torch.float32)
        for frame in normalized
    ]


def _tracking_inputs(track: Any, count: int, height: int, width: int) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    packed = cast(dict[str, Any], track).get("packed_masks") if isinstance(track, dict) else None
    if packed is None or packed.shape[0] != count or packed.shape[1] == 0:
        return None, None
    management = importlib.import_module("dinkster_inference.model_management")
    unpack_masks = importlib.import_module("dinkster_inference.ldm.sam3.tracker").unpack_masks
    unpacked = unpack_masks(packed.to(management.get_torch_device()))
    frames, people, mask_h, mask_w = unpacked.shape
    resized = functional.interpolate(
        unpacked.float().reshape(frames * people, 1, mask_h, mask_w),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    binary = (resized > 0.5).to(torch.uint8).reshape(frames, people, height, width)
    cpu = binary.cpu()
    masks = [cpu[index, :, :, :, None].contiguous() for index in range(frames)]
    boxes: list[Any] = []
    for frame in binary:
        bounds: list[Any] = []
        for mask in frame:
            positive = mask > 0
            eroded = -functional.max_pool2d(
                -positive.to(torch.float32)[None, None], kernel_size=5, stride=1, padding=2
            )
            eroded = eroded[0, 0] > 0.5
            keep = torch.where(eroded.any(), eroded, positive)
            xs, ys = (
                torch.arange(width, device=keep.device),
                torch.arange(height, device=keep.device),
            )
            cols, rows = keep.any(dim=0), keep.any(dim=1)
            bbox = torch.stack(
                [
                    torch.where(cols, xs, width).amin(),
                    torch.where(rows, ys, height).amin(),
                    torch.where(cols, xs + 1, 0).amax(),
                    torch.where(rows, ys + 1, 0).amax(),
                ]
            )
            bounds.append(
                torch.where(positive.any(), bbox, bbox.new_tensor([0, 0, width, height]))
                .to(torch.float32)
                .cpu()
            )
        boxes.append(torch.stack(bounds))
    return boxes, masks


def _predict_chunk(
    inner: Any, images: list[Any], boxes: list[Any], masks: Any, mode: str, camera: Any
) -> list[list[dict[str, Any]]]:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    prepare_batch = importlib.import_module("dinkster_inference.ldm.sam3d_body.utils").prepare_batch
    people = len(boxes[0])
    flat_images = [image for image in images for _ in range(people)]
    flat_masks = None
    if masks is not None:
        flat_masks = torch.stack(
            [
                frame[index] if len(frame) > 1 else frame[0]
                for frame in masks
                for index in range(people)
            ]
        )
    batch = prepare_batch(
        flat_images,
        torch.stack([box for frame in boxes for box in frame]),
        input_size=inner.image_size,
        masks=flat_masks,
        masks_score=torch.ones(len(flat_images), dtype=torch.float32)
        if masks is not None
        else None,
        cam_int=camera,
    )
    batch = {
        key: value.to(management.get_torch_device()) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    raw = inner.run_inference(flat_images, batch, inference_type=mode, thresh_wrist_angle=1.4)
    if mode == "full":
        pose, left, right, _, _ = raw
    else:
        pose = raw
        left = right = None
    values = {
        key: value.float().cpu().numpy()
        for key, value in pose["mhr"].items()
        if value is not None and key != "faces"
    }
    bounds = batch["bbox"][0].cpu().numpy()
    left_bounds = right_bounds = None
    if left is not None and right is not None:
        left_bounds, right_bounds = _hand_bounds(left), _hand_bounds(right)
    del pose, batch, left, right, raw
    names = {
        "body_pose_params": "body_pose",
        "hand_pose_params": "hand",
        "scale_params": "scale",
        "shape_params": "shape",
        "expr_params": "face",
        "pred_global_rots": "joint_global_rots",
    }
    fields = (
        "focal_length",
        "pred_keypoints_3d",
        "pred_keypoints_2d",
        "pred_vertices",
        "pred_cam_t",
        "pred_pose_raw",
        "global_rot",
        "body_pose_params",
        "hand_pose_params",
        "scale_params",
        "shape_params",
        "expr_params",
        "pred_joint_coords",
        "pred_global_rots",
        "mhr_model_params",
    )
    result: list[list[dict[str, Any]]] = []
    for frame in range(len(images)):
        persons: list[dict[str, Any]] = []
        for person in range(people):
            index = frame * people + person
            value = {key: values[names.get(key, key)][index] for key in fields}
            value["bbox"] = bounds[index]
            value["mask"] = (
                (masks[frame][person] if len(masks[frame]) > 1 else masks[frame][0])
                if masks is not None
                else None
            )
            for key in ("pred_face_keypoints_3d", "pred_face_keypoints_2d"):
                value[key] = values[key][index] if key in values else None
            if left_bounds is not None and right_bounds is not None:
                value["lhand_bbox"], value["rhand_bbox"] = left_bounds[index], right_bounds[index]
            persons.append(value)
        result.append(persons)
    return result


def _hand_bounds(batch: Any) -> Any:
    torch = importlib.import_module("torch")
    centers = batch["bbox_center"].flatten(0, 1)
    half = batch["bbox_scale"].flatten(0, 1) * 0.5
    return torch.cat([centers - half, centers + half], dim=1).float().cpu().numpy()


def _canonical_colors(model: Any) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    head = model.head_pose
    vertices = head.canonical_vertices().float().cpu().numpy()
    faces = head.faces_np()
    normals = np.zeros_like(vertices, dtype=np.float32)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    ).astype(np.float32)
    for index in range(3):
        np.add.at(normals, faces[:, index], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    count = head.num_face_comps + 1

    def zeros(*shape: int) -> Any:
        return torch.zeros(count, *shape, device=head.scale_mean.device)

    expression = zeros(head.num_face_comps)
    expression[1:] = torch.eye(head.num_face_comps, device=head.scale_mean.device)
    posed = (
        head.mhr_forward(
            global_trans=zeros(3),
            global_rot=zeros(3),
            body_pose_params=zeros(130),
            hand_pose_params=zeros(head.num_hand_comps * 2),
            scale_params=zeros(head.num_scale_comps),
            shape_params=zeros(head.num_shape_comps),
            expr_params=expression,
        )
        .cpu()
        .numpy()
    )
    return {
        "positions": vertices.astype(np.float32),
        "norm": ((normals / lengths + 1.0) * 0.5).astype(np.float32),
        "face_mask": (np.linalg.norm(posed[1:] - posed[0][None], axis=2) > 1e-4).any(axis=0),
        "head_mask": (vertices[:, 1] > 1.43) & (np.abs(vertices[:, 0]) < 0.11),
        "face_region_rgb": head.face_region_rgb.detach().float().cpu().numpy(),
    }


def _hand_mask(model: Any) -> Any:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    head = model.head_pose

    def zeros(*shape: int) -> Any:
        return torch.zeros(1, *shape, device=head.scale_mean.device)

    _, keypoints, joints = head.mhr_forward(
        global_trans=zeros(3),
        global_rot=zeros(3),
        body_pose_params=zeros(130),
        hand_pose_params=zeros(head.num_hand_comps * 2),
        scale_params=zeros(head.num_scale_comps),
        shape_params=zeros(head.num_shape_comps),
        expr_params=zeros(head.num_face_comps),
        return_keypoints=True,
        return_joint_coords=True,
    )
    keypoints, joints = keypoints[0, :70].cpu().numpy(), joints[0].cpu().numpy()
    hand_joints = (np.linalg.norm(joints - keypoints[21:42].mean(axis=0), axis=1) < 0.15) | (
        np.linalg.norm(joints - keypoints[42:63].mean(axis=0), axis=1) < 0.15
    )
    mass = np.zeros(head.mhr.NUM_VERTS, dtype=np.float32)
    np.add.at(
        mass,
        head.mhr.lbs_vert_indices.cpu().numpy(),
        head.mhr.lbs_skin_weights.cpu().numpy()
        * hand_joints.astype(np.float32)[head.mhr.lbs_skin_indices.cpu().numpy()],
    )
    return mass >= 0.5
