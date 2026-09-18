"""RT-DETR v4 x-HGNet object detection over BHWC image batches."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from dinkster_api.v1 import Detection, Region, declared_asset
from safetensors.torch import load_file

from .rtdetr import COCO_CLASSES, RTv4

MODEL_ASSET = "rtdetr-v4-x-hgnet-fp16"
MODEL_INPUT_SIZE = 640
MODEL_BATCH_SIZE = 32

_MODEL: RTv4 | None = None
_MODEL_DIGEST = ""


def load_model() -> RTv4:
    """Load the declared fp16 checkpoint as a cached float32 CPU model."""
    global _MODEL, _MODEL_DIGEST
    asset = declared_asset(MODEL_ASSET)
    if _MODEL is not None and _MODEL_DIGEST == asset.digest:
        return _MODEL
    state = load_file(asset.local_path(), device="cpu")
    if not state or any(
        value.is_floating_point() and value.dtype != torch.float16 for value in state.values()
    ):
        raise ValueError("RT-DETR model artifact must contain fp16 floating tensors")
    model = RTv4(enc_h=384, device=torch.device("cpu"), dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.float().eval()
    _MODEL = model
    _MODEL_DIGEST = asset.digest
    return model


def _frames(image: object) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"image must have non-empty BHWC shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("RT-DETR detection requires finite pixel values")
    array = np.clip(array, 0.0, 1.0)
    if array.shape[3] == 3:
        return [np.ascontiguousarray(frame) for frame in array]
    if array.shape[3] == 1:
        return [np.ascontiguousarray(np.repeat(frame, 3, axis=2)) for frame in array]
    if array.shape[3] == 4:
        color = array[..., :3]
        alpha = array[..., 3:4]
        composited = np.clip(color * alpha + (1.0 - alpha), 0.0, 1.0)
        return [np.ascontiguousarray(frame) for frame in composited]
    raise ValueError(f"image frame must have 1, 3, or 4 channels, got {array.shape}")


def prepare_frames(frames: list[np.ndarray]) -> torch.Tensor:
    """Match ComfyUI's uncropped 640x640 bilinear preprocessing."""
    batch = torch.from_numpy(np.stack(frames)).movedim(-1, 1)
    return F.interpolate(batch, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode="bilinear")


def _parse_prompt(prompt: str, prompt_mode: str) -> frozenset[str]:
    if type(prompt) is not str:
        raise TypeError("prompt must be a string")
    if prompt_mode == "literal":
        stripped = prompt.strip()
        return frozenset((stripped.casefold(),)) if stripped else frozenset()
    if prompt_mode != "comma-separated":
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    return frozenset(name.strip().casefold() for name in prompt.split(",") if name.strip())


def _detections(
    result: dict[str, torch.Tensor],
    *,
    min_score: float,
    wanted: frozenset[str],
    max_results: int,
    result_limit_mode: str,
) -> list[Detection]:
    candidates: list[Detection] = []
    for box, label_value, score_value in zip(
        result["boxes"], result["labels"], result["scores"], strict=True
    ):
        score = float(score_value)
        if score <= min_score:
            continue
        label = COCO_CLASSES[int(label_value)]
        if wanted and label.casefold() not in wanted:
            continue
        left, top, right, bottom = (float(value) for value in box)
        candidates.append(Detection(label, score, Region(left, top, right - left, bottom - top)))
    candidates.sort(key=lambda detection: detection.score, reverse=True)
    if result_limit_mode == "slice-stop":
        return candidates[:max_results]
    return candidates if max_results < 0 else candidates[:max_results]


def execute_detect(
    image: object,
    *,
    prompt: str,
    prompt_mode: str = "comma-separated",
    min_score: float,
    max_results: int = -1,
    result_limit_mode: str = "count",
) -> list[Detection]:
    """Detect COCO objects per frame and concatenate frames in batch order."""
    if type(min_score) not in (int, float):
        raise TypeError("min_score must be a number")
    threshold = float(min_score)
    if not math.isfinite(threshold):
        raise ValueError("min_score must be finite")
    if result_limit_mode not in ("count", "slice-stop"):
        raise ValueError(f"unknown result limit mode: {result_limit_mode}")
    if type(max_results) is not int or (result_limit_mode == "count" and max_results < -1):
        raise ValueError("max_results must be an integer and at least -1 in count mode")
    frames = _frames(image)
    wanted = _parse_prompt(prompt, prompt_mode)
    if max_results == 0:
        return []
    model = load_model()
    detections: list[Detection] = []
    with torch.inference_mode():
        for offset in range(0, len(frames), MODEL_BATCH_SIZE):
            batch = frames[offset : offset + MODEL_BATCH_SIZE]
            prepared = prepare_frames(batch)
            results = model(prepared, (batch[0].shape[1], batch[0].shape[0]))
            for result in results:
                detections.extend(
                    _detections(
                        result,
                        min_score=threshold,
                        wanted=wanted,
                        max_results=max_results,
                        result_limit_mode=result_limit_mode,
                    )
                )
    return detections


__all__ = [
    "COCO_CLASSES",
    "MODEL_INPUT_SIZE",
    "RTv4",
    "execute_detect",
    "load_model",
    "prepare_frames",
]
