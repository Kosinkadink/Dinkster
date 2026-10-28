"""SDPose feature capture and OpenPose frame conversion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    MODEL_TYPE,
    POSE_KEYPOINT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

from .lotus import LotusConditioning


def _openpose_frames(keypoints: list[Any], scores: list[Any], height: int, width: int) -> list[Any]:
    np = importlib.import_module("numpy")

    def flatten(kp: Any, sc: Any) -> list[float]:
        return np.stack([kp[:, 0], kp[:, 1], sc], axis=1).flatten().tolist()

    frames: list[Any] = []
    for frame_kp, frame_sc in zip(keypoints, scores, strict=False):
        people: list[Any] = []
        for raw_kp, raw_sc in zip(frame_kp, frame_sc, strict=False):
            kp, sc = np.array(raw_kp, dtype=np.float32), np.array(raw_sc, dtype=np.float32)
            if len(kp) >= 17:
                neck = (kp[5] + kp[6]) / 2
                neck_score = min(sc[5], sc[6]) if sc[5] > 0.3 and sc[6] > 0.3 else 0
                kp, sc = np.insert(kp, 17, neck, axis=0), np.insert(sc, 17, neck_score)
                source = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                target = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
                copied_kp, copied_sc = kp.copy(), sc.copy()
                copied_kp[target], copied_sc[target] = kp[source], sc[source]
                kp, sc = copied_kp, copied_sc
            face = np.concatenate([kp[24:92], kp[[14, 15]]], axis=0)
            face_scores = np.concatenate([sc[24:92], sc[[14, 15]]], axis=0)
            people.append(
                {
                    "pose_keypoints_2d": flatten(kp[:18], sc[:18]),
                    "foot_keypoints_2d": flatten(kp[18:24], sc[18:24]),
                    "face_keypoints_2d": flatten(face, face_scores),
                    "hand_right_keypoints_2d": flatten(kp[92:113], sc[92:113]),
                    "hand_left_keypoints_2d": flatten(kp[113:134], sc[113:134]),
                }
            )
        frames.append({"canvas_width": width, "canvas_height": height, "people": people})
    return frames


class SDPoseKeypointExtractor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SDPoseKeypointExtractor",
            display_name="SDPose Keypoint Extractor",
            category="image/detection",
            search_terms=("openpose", "pose detection", "preprocessor", "keypoints", "sdpose"),
            description=(
                "Extract pose keypoints from images using the SDPose model: "
                "https://huggingface.co/Comfy-Org/SDPose/tree/main/checkpoints"
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=16,
                    widget=NumberWidget(min=1, max=10000, step=1),
                ),
                InputSpec(
                    "bboxes",
                    TypeExpr.concrete(BOUNDING_BOX_TYPE),
                    required=False,
                    default={"x": 0, "y": 0, "width": 512, "height": 512},
                    force_input=True,
                    doc="Optional bounding boxes for more accurate detections. "
                    "Required for multi-person detection.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "keypoints",
                    TypeExpr.concrete(POSE_KEYPOINT_TYPE),
                    doc="Keypoints in OpenPose frame format (canvas_width, canvas_height, people)",
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, model: Any, vae: Any, image: Any, batch_size: int, bboxes: Any = None
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        utils = importlib.import_module("dinkster_inference.utils")
        sample = importlib.import_module("dinkster_inference.sample")
        image = materialize_image(image)
        height, width = image.shape[-3:-1]
        context = LotusConditioning.execute()["conditioning"]
        captured = None

        def output_patch(h: Any, hsp: Any, transformer_options: Any) -> tuple[Any, Any]:
            nonlocal captured
            if h.shape[1] == 640:
                captured = h.clone()
            return h, hsp

        clone = model.clone()
        clone.model_options["transformer_options"] = {
            "patches": {"output_block_patch": [output_patch]}
        }
        if not hasattr(model.model.diffusion_model, "heatmap_head"):
            raise ValueError(
                "The provided model does not have a heatmap_head. "
                "Please use SDPose model from here "
                "https://huggingface.co/Comfy-Org/SDPose/tree/main/checkpoints."
            )
        head = model.model.diffusion_model.heatmap_head
        model_w, model_h = int(head.heatmap_size[0]) * 4, int(head.heatmap_size[1]) * 4

        def resize(images: Any) -> tuple[Any, float, float]:
            h, w = images.shape[-3:-1]
            method = "area" if model_h <= h and model_w <= w else "bilinear"
            scaled = utils.common_upscale(
                images.permute(0, 3, 1, 2).float(), model_w, model_h, method, crop="disabled"
            )
            return scaled.permute(0, 2, 3, 1), model_w / w, model_h / h

        def remap(kp: Any, sx: float, sy: float, x: int = 0, y: int = 0) -> Any:
            kp = kp.copy() if isinstance(kp, np.ndarray) else np.array(kp, dtype=np.float32)
            invalid = kp[..., 0] < 0
            kp[..., 0], kp[..., 1] = kp[..., 0] / sx + x, kp[..., 1] / sy + y
            kp[invalid] = -1
            return kp

        def run(latent: Any) -> Any:
            nonlocal captured
            captured = None
            sample.sample(
                clone,
                noise=torch.zeros_like(latent),
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                positive=context,
                negative=context,
                latent_image=latent,
                disable_noise=True,
                disable_pbar=True,
            )
            return head(captured)

        keypoints: list[Any] = []
        scores: list[Any] = []
        total = image.shape[0]
        progress = utils.ProgressBar(total)
        if bboxes is None:
            for start in range(0, total, batch_size):
                resized, sx, sy = resize(image[start : start + batch_size])
                kp_batch, sc_batch = run(vae.encode(resized))
                for kp, sc in zip(kp_batch, sc_batch, strict=False):
                    keypoints.append([remap(kp, sx, sy)])
                    scores.append([sc])
                progress.update(len(kp_batch))
        else:
            if not isinstance(bboxes, list):
                bboxes = [[bboxes]]
            elif not bboxes:
                bboxes = [None] * total
            box_lists = cast(list[Any], bboxes)
            for index in range(total):
                img = image[index : index + 1]
                boxes = box_lists[min(index, len(box_lists) - 1)] if box_lists else None
                frame_kp: list[Any] = []
                frame_sc: list[Any] = []
                if boxes:
                    for box in boxes:
                        x1, y1 = max(0, int(box["x"])), max(0, int(box["y"]))
                        x2 = min(width, int(box["x"] + box["width"]))
                        y2 = min(height, int(box["y"] + box["height"]))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        resized, sx, sy = resize(img[:, y1:y2, x1:x2, :])
                        kp, sc = run(vae.encode(resized))
                        frame_kp.append(remap(kp[0], sx, sy, x1, y1))
                        frame_sc.append(sc[0])
                else:
                    resized, sx, sy = resize(img)
                    kp, sc = run(vae.encode(resized))
                    frame_kp.append(remap(kp[0], sx, sy))
                    frame_sc.append(sc[0])
                keypoints.append(frame_kp)
                scores.append(frame_sc)
                progress.update(1)
        return cls.outputs(keypoints=_openpose_frames(keypoints, scores, height, width))
