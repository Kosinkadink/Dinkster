"""SAM 3.1 text detection, box segmentation, and video tracking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from dinkster_api.v1 import ABSENT, Detection, Region, coerce_detection, declared_asset
from safetensors import safe_open
from torch.nn import functional as F

from .detector import SAM31Detector
from .sam31 import IMAGE_SIZE, SAM31InteractiveModel
from .text import SAM31TextEncoder
from .tracker import SAM31Tracker, unpack_masks

MODEL_ASSET = "sam31-multiplex-fp16"
REFINEMENT_PASSES = 2
_PROMPT_MODES = ("comma-separated", "literal")


class SAM31Model(SAM31InteractiveModel):
    def __init__(self) -> None:
        super().__init__()
        self.tracker = SAM31Tracker()

    def track(self, frames: torch.Tensor, initial_masks: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            packed = self.tracker.track_video(
                self.backbone.tracking_features,
                frames,
                initial_masks,
            )
        return unpack_masks(packed).to(dtype=torch.float32)


class SAM31DetectionModel(SAM31Model):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = SAM31TextEncoder()
        self.detector = SAM31Detector()


_MODEL: SAM31Model | None = None
_MODEL_DIGEST = ""


def _target_key(source: str) -> str | None:
    backbone = "detector.backbone.vision_backbone."
    tracker = "tracker.model."
    if source.startswith(backbone + "trunk.") or source.startswith(
        (
            backbone + "interactive_convs.",
            backbone + "propagation_convs.",
        )
    ):
        if ".attn.freqs_cis" in source:
            return None
        return "backbone." + source.removeprefix(backbone)
    if source.startswith(tracker):
        relative = source.removeprefix(tracker)
        return "tracker." + relative
    return None


def _detection_target_key(source: str) -> str | None:
    shared = _target_key(source)
    if shared is not None:
        return shared
    detector_fpn = "detector.backbone.vision_backbone.convs."
    if source.startswith(detector_fpn):
        return "detector.fpn_convs." + source.removeprefix(detector_fpn)
    language = "detector.backbone.language_backbone."
    if source.startswith(language + "resizer."):
        return "detector.language_resizer." + source.removeprefix(language + "resizer.")
    if source.startswith(language + "encoder."):
        target = "text_encoder." + source.removeprefix(language + "encoder.")
        target = target.replace("ln_final.", "final_layer_norm.")
        target = target.replace("transformer.resblocks.", "layers.")
        target = target.replace(".ln_1.", ".layer_norm1.")
        target = target.replace(".ln_2.", ".layer_norm2.")
        target = target.replace(".attn.", ".self_attention.")
        target = target.replace(".mlp.c_fc.", ".fc1.")
        return target.replace(".mlp.c_proj.", ".fc2.")
    if source.startswith("detector."):
        return source
    return None


def _load_state(
    path: str | Path,
    *,
    detection: bool = False,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as stored:
        for source in stored.keys():
            target = _detection_target_key(source) if detection else _target_key(source)
            if target is None:
                continue
            value = stored.get_tensor(source)
            if target.endswith((".in_proj_weight", ".in_proj_bias")):
                base, suffix = target.rsplit(".in_proj_", 1)
                ending = ".weight" if suffix == "weight" else ".bias"
                size = value.shape[0] // 3
                state[base + ".q_proj" + ending] = value[:size]
                state[base + ".k_proj" + ending] = value[size : 2 * size]
                state[base + ".v_proj" + ending] = value[2 * size :]
                continue
            target = target.replace(".mlp.lin1.", ".mlp.0.")
            target = target.replace(".mlp.lin2.", ".mlp.2.")
            target = target.replace(".norm_final_attn.", ".norm_final.")
            state[target] = value
            if (
                target.startswith("tracker.interactive_sam_")
                or target == "tracker.interactivity_no_mem_embed"
            ):
                state[target.removeprefix("tracker.")] = value
    return state


def load_model() -> SAM31Model:
    global _MODEL, _MODEL_DIGEST
    asset = declared_asset(MODEL_ASSET)
    if _MODEL is not None and _MODEL_DIGEST == asset.digest:
        return _MODEL
    with torch.device("meta"):
        model = SAM31Model()
    state = _load_state(asset.local_path())
    expected = set(model.state_dict())
    actual = set(state)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "SAM 3.1 checkpoint keys do not match: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_frequencies()
    model.tracker.reset_frequencies()
    model.float().eval()
    _MODEL = model
    _MODEL_DIGEST = asset.digest
    return model


def load_detection_model() -> SAM31DetectionModel:
    global _MODEL, _MODEL_DIGEST
    asset = declared_asset(MODEL_ASSET)
    if isinstance(_MODEL, SAM31DetectionModel) and _MODEL_DIGEST == asset.digest:
        return _MODEL
    with torch.device("meta"):
        model = SAM31DetectionModel()
    state = _load_state(asset.local_path(), detection=True)
    expected = set(model.state_dict())
    actual = set(state)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "SAM 3.1 detection checkpoint keys do not match: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_frequencies()
    model.tracker.reset_frequencies()
    model.float().eval()
    _MODEL = model
    _MODEL_DIGEST = asset.digest
    return model


def _frame(image: object) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(
            f"SAM 3.1 requires exactly one non-empty BHWC image frame, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("SAM 3.1 requires finite pixel values")
    array = np.clip(array[0], 0.0, 1.0)
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)
    if array.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(array, 3, axis=2))
    if array.shape[2] == 4:
        return np.ascontiguousarray(array[:, :, :3])
    raise ValueError(f"image frame must have 1, 3, or 4 channels, got {array.shape}")


def _corners(
    region: Region,
    *,
    height: int,
    width: int,
) -> tuple[float, float, float, float] | None:
    left = min(max(float(region.x), 0.0), float(width))
    top = min(max(float(region.y), 0.0), float(height))
    right = min(max(float(region.right), 0.0), float(width))
    bottom = min(max(float(region.bottom), 0.0), float(height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def prepare_frame(frame: np.ndarray) -> torch.Tensor:
    image = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).unsqueeze(0)
    return F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")


def _prompt(
    box: tuple[float, float, float, float],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    left, top, right, bottom = box
    return torch.tensor(
        [
            [
                [left / width * IMAGE_SIZE, top / height * IMAGE_SIZE],
                [right / width * IMAGE_SIZE, bottom / height * IMAGE_SIZE],
            ]
        ],
        dtype=torch.float32,
    )


def predict_logits(
    model: SAM31InteractiveModel,
    frame: np.ndarray,
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    bicubic: bool = False,
) -> list[np.ndarray]:
    prepared = prepare_frame(frame)
    if bicubic:
        image = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).unsqueeze(0)
        prepared = F.interpolate(
            image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False
        )
    with torch.inference_mode():
        features = model.encode_image(prepared)
        outputs: list[np.ndarray] = []
        for box in boxes:
            logits = model.segment(
                features,
                box=_prompt(box, height=frame.shape[0], width=frame.shape[1]),
            )
            for _ in range(REFINEMENT_PASSES - 1):
                logits = model.segment(features, mask=logits)
            restored = F.interpolate(
                logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            outputs.append(np.ascontiguousarray(restored.cpu().numpy(), dtype=np.float32))
    return outputs


def _immutable_mask(mask: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(mask, dtype=np.float32)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(contiguous.shape)


def _frames(image: object) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"SAM 3.1 detection requires a non-empty BHWC batch, got {array.shape}")
    return [_frame(array[index : index + 1]) for index in range(array.shape[0])]


def _phrases(prompt: str, prompt_mode: str) -> tuple[str, ...]:
    if type(prompt) is not str:
        raise TypeError("prompt must be a string")
    if prompt_mode not in _PROMPT_MODES:
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    if prompt_mode == "literal":
        stripped = prompt.strip()
        return (stripped,) if stripped else ()
    return tuple(value.strip() for value in prompt.split(",") if value.strip())


def _refine_detection_mask(
    model: SAM31DetectionModel,
    frame: np.ndarray,
    coarse: torch.Tensor,
    box: torch.Tensor,
) -> np.ndarray:
    height, width = frame.shape[:2]
    left, top, right, bottom = (float(value) for value in box)
    box_width, box_height = right - left, bottom - top
    crop_left = max(0, int(left - box_width * 0.1))
    crop_top = max(0, int(top - box_height * 0.1))
    crop_right = min(width, int(right + box_width * 0.1))
    crop_bottom = min(height, int(bottom + box_height * 0.1))
    coarse_full = F.interpolate(
        coarse[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return _immutable_mask((coarse_full > 0).numpy())

    mask_height, mask_width = coarse.shape
    mask_left = int(crop_left / width * mask_width)
    mask_top = int(crop_top / height * mask_height)
    mask_right = int(crop_right / width * mask_width)
    mask_bottom = int(crop_bottom / height * mask_height)
    if mask_right <= mask_left or mask_bottom <= mask_top:
        return _immutable_mask((coarse_full > 0).numpy())

    crop = np.ascontiguousarray(frame[crop_top:crop_bottom, crop_left:crop_right])
    features = model.encode_image(prepare_frame(crop))
    logits = coarse[mask_top:mask_bottom, mask_left:mask_right][None, None]
    for _ in range(REFINEMENT_PASSES):
        logits = model.segment(
            features,
            mask=F.interpolate(
                logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ),
        )
    refined = F.interpolate(
        logits,
        size=(crop_bottom - crop_top, crop_right - crop_left),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    result = coarse_full > 0
    result[crop_top:crop_bottom, crop_left:crop_right] |= refined > 0
    return _immutable_mask(result.numpy())


def _detect_frame(
    model: SAM31DetectionModel,
    frame: np.ndarray,
    encoded_phrases: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    min_score: float,
    max_results: int = -1,
    result_limit_mode: str = "count",
) -> list[Detection]:
    prepared = prepare_frame(frame)
    trunk = model.backbone.trunk(prepared)
    height, width = frame.shape[:2]
    candidates: list[tuple[float, int, int, str, torch.Tensor, torch.Tensor]] = []
    for phrase_index, (phrase, text, text_mask) in enumerate(encoded_phrases):
        boxes, logits, coarse_masks = model.detector(trunk, text, text_mask)
        scores = logits[0].sigmoid()
        for query_index in torch.argsort(scores, descending=True, stable=True).tolist():
            score = float(scores[query_index])
            if score <= min_score:
                break
            raw_box = boxes[0, query_index] * torch.tensor(
                (width, height, width, height),
                dtype=boxes.dtype,
                device=boxes.device,
            )
            candidates.append(
                (
                    score,
                    phrase_index,
                    query_index,
                    phrase,
                    raw_box,
                    coarse_masks[0, query_index],
                )
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if result_limit_mode == "slice-stop" or max_results >= 0:
        candidates = candidates[:max_results]
    detections: list[Detection] = []
    for score, _, _, phrase, raw_box, coarse_mask in candidates:
        left = min(max(float(raw_box[0]), 0.0), float(width))
        top = min(max(float(raw_box[1]), 0.0), float(height))
        right = min(max(float(raw_box[2]), 0.0), float(width))
        bottom = min(max(float(raw_box[3]), 0.0), float(height))
        mask = _refine_detection_mask(model, frame, coarse_mask, raw_box)
        detections.append(
            Detection(
                phrase,
                score,
                Region(left, top, max(0.0, right - left), max(0.0, bottom - top)),
                mask,
            )
        )
    return detections


def execute_detect(
    image: object,
    *,
    prompt: str,
    prompt_mode: str = "comma-separated",
    min_score: float,
    max_results: int = -1,
    result_limit_mode: str = "count",
) -> list[Detection]:
    """Detect each selected prompt phrase in every frame, in frame order."""
    if type(min_score) not in (int, float):
        raise TypeError("min_score must be a number")
    threshold = float(min_score)
    if not np.isfinite(threshold):
        raise ValueError("min_score must be finite")
    if result_limit_mode not in ("count", "slice-stop"):
        raise ValueError(f"unknown result limit mode: {result_limit_mode}")
    if type(max_results) is not int or (result_limit_mode == "count" and max_results < -1):
        raise ValueError("max_results must be an integer and at least -1 in count mode")
    frames = _frames(image)
    phrases = _phrases(prompt, prompt_mode)
    if not phrases or max_results == 0:
        return []
    model = load_detection_model()
    with torch.inference_mode():
        encoded = [(phrase, *model.text_encoder.encode(phrase)) for phrase in phrases]
        detections: list[Detection] = []
        for frame in frames:
            detections.extend(
                _detect_frame(
                    model,
                    frame,
                    encoded,
                    threshold,
                    max_results,
                    result_limit_mode,
                )
            )
    return detections


def execute_text_segment(
    image: object,
    *,
    prompt: str,
    prompt_mode: str = "comma-separated",
    min_score: float,
) -> tuple[list[Detection], list[np.ndarray]]:
    """Expose text-detected masks through the segmentation contract."""
    detections = execute_detect(
        image,
        prompt=prompt,
        prompt_mode=prompt_mode,
        min_score=min_score,
    )
    masks: list[np.ndarray] = []
    for detection in detections:
        mask = cast("np.ndarray", detection.mask)
        masks.append(_immutable_mask(mask).reshape(1, mask.shape[0], mask.shape[1]))
    return detections, masks


def execute_segment(
    image: object,
    detections: Sequence[object],
) -> tuple[list[Detection], list[np.ndarray]]:
    """Attach one source-sized SAM 3.1 mask to each box-prompt detection."""
    frame = _frame(image)
    items = [coerce_detection(item) for item in detections]
    if not items:
        return [], []
    height, width = frame.shape[:2]
    valid: list[tuple[int, tuple[float, float, float, float]]] = []
    for index, detection in enumerate(items):
        corners = _corners(detection.region, height=height, width=width)
        if corners is not None:
            valid.append((index, corners))

    masks = [_immutable_mask(np.zeros((height, width), dtype=np.float32)) for _ in items]
    if valid:
        logits = predict_logits(load_model(), frame, [box for _, box in valid])
        for values, (index, _) in zip(logits, valid, strict=True):
            masks[index] = _immutable_mask(values > 0.0)

    segmented = [
        Detection(item.label, item.score, item.region, mask)
        for item, mask in zip(items, masks, strict=True)
    ]
    batched_masks = [
        cast("np.ndarray", detection.mask).reshape(1, height, width) for detection in segmented
    ]
    return segmented, batched_masks


def execute_track(
    image: object,
    detections: object = ABSENT,
    *,
    initial_masks: object = ABSENT,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Track first-frame box or mask prompts across a BHWC frame batch."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(
            f"SAM 3.1 tracking requires a non-empty BHWC frame batch, got {array.shape}"
        )
    frames = [_frame(array[index : index + 1]) for index in range(array.shape[0])]
    height, width = frames[0].shape[:2]
    items = (
        []
        if detections is ABSENT
        else [coerce_detection(item) for item in cast("Sequence[object]", detections)]
    )
    has_initial_masks = initial_masks is not ABSENT
    if bool(items) == has_initial_masks:
        raise ValueError("tracking requires exactly one of detections or initial_masks")

    if has_initial_masks:
        initial_array = np.asarray(initial_masks, dtype=np.float32)
        if (
            initial_array.ndim != 3
            or initial_array.shape[0] < 1
            or initial_array.shape[1] < 1
            or initial_array.shape[2] < 1
        ):
            raise ValueError(
                f"initial_masks must be a non-empty BHW mask, got {initial_array.shape}"
            )
        if not np.isfinite(initial_array).all():
            raise ValueError("initial_masks must contain only finite values")
        initial = torch.from_numpy(
            np.array(initial_array, dtype=np.float32, copy=True, order="C")
        ).unsqueeze(1)
        output_indices = tuple(range(initial.shape[0]))
    else:
        valid = [
            (index, box)
            for index, item in enumerate(items)
            if (box := _corners(item.region, height=height, width=width)) is not None
        ]
        if not valid:
            outputs = [np.zeros((len(frames), height, width), dtype=np.float32) for _ in items]
            combined = np.zeros((len(frames), height, width), dtype=np.float32)
            return [_immutable_mask(mask) for mask in outputs], _immutable_mask(combined)
        model = load_model()
        first_logits = predict_logits(model, frames[0], [box for _, box in valid], bicubic=True)
        initial = (
            torch.from_numpy(np.stack([mask > 0 for mask in first_logits])).unsqueeze(1).float()
        )
        output_indices = tuple(index for index, _ in valid)

    outputs = [
        np.zeros((len(frames), height, width), dtype=np.float32)
        for _ in range(initial.shape[0] if has_initial_masks else len(items))
    ]
    tensor = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2))
    tracked = load_model().track(tensor, initial)
    restored = F.interpolate(tracked, size=(height, width), mode="bilinear", align_corners=False)
    for slot, index in enumerate(output_indices):
        outputs[index] = np.ascontiguousarray(restored[:, slot].numpy(), dtype=np.float32)
    combined = F.interpolate(
        tracked.amax(dim=1, keepdim=True),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0].numpy()
    return [_immutable_mask(mask) for mask in outputs], _immutable_mask(combined)


__all__ = [
    "MODEL_ASSET",
    "REFINEMENT_PASSES",
    "SAM31DetectionModel",
    "execute_detect",
    "execute_segment",
    "execute_track",
    "load_detection_model",
    "load_model",
    "predict_logits",
    "prepare_frame",
]
