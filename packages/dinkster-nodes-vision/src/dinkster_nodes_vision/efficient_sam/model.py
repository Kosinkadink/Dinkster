"""EfficientSAM-Ti box-prompt segmentation through official ONNX exports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import onnxruntime as ort
from dinkster_api.v1 import Detection, Region, coerce_detection, declared_asset

_ENCODER_ASSET = "efficient-sam-vitt-encoder"
_DECODER_ASSET = "efficient-sam-vitt-decoder"

_Sessions = tuple[ort.InferenceSession, ort.InferenceSession]
_SESSIONS: _Sessions | None = None
_SESSION_DIGESTS: tuple[str, str] = ("", "")


def _session_options(intra_op_num_threads: int | None) -> ort.SessionOptions:
    options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def _create_sessions(*, intra_op_num_threads: int | None = None) -> _Sessions:
    options = _session_options(intra_op_num_threads)
    assets = (declared_asset(_ENCODER_ASSET), declared_asset(_DECODER_ASSET))
    payloads: list[bytes] = []
    for asset in assets:
        with asset.open() as stream:
            payloads.append(stream.read())
    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(payloads[0], sess_options=options, providers=providers),
        ort.InferenceSession(payloads[1], sess_options=options, providers=providers),
    )


def _load_sessions() -> _Sessions:
    global _SESSIONS, _SESSION_DIGESTS
    digests = (
        declared_asset(_ENCODER_ASSET).digest,
        declared_asset(_DECODER_ASSET).digest,
    )
    if _SESSIONS is None or _SESSION_DIGESTS != digests:
        _SESSIONS = _create_sessions()
        _SESSION_DIGESTS = digests
    return _SESSIONS


def _frame(image: object) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(
            "EfficientSAM segmentation requires exactly one non-empty BHWC image frame, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("EfficientSAM segmentation requires finite pixel values")
    array = np.clip(array, 0.0, 1.0)
    if array.shape[3] == 3:
        return np.ascontiguousarray(array[0])
    if array.shape[3] == 1:
        return np.ascontiguousarray(np.repeat(array[0], 3, axis=2))
    if array.shape[3] == 4:
        color = array[0, ..., :3]
        alpha = array[0, ..., 3:4]
        return np.ascontiguousarray(np.clip(color * alpha + (1.0 - alpha), 0.0, 1.0))
    raise ValueError(f"image frame must have 1, 3, or 4 channels, got {array.shape}")


def _corners(
    region: Region, *, height: int, width: int
) -> tuple[float, float, float, float] | None:
    left = min(max(float(region.x), 0.0), float(width))
    top = min(max(float(region.y), 0.0), float(height))
    right = min(max(float(region.right), 0.0), float(width))
    bottom = min(max(float(region.bottom), 0.0), float(height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _encode_frame(
    frame: np.ndarray,
    encoder: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.ascontiguousarray(frame.transpose(2, 0, 1)[None], dtype=np.float32)
    embedding = cast(
        "np.ndarray",
        encoder.run(["image_embeddings"], {"batched_images": image})[0],
    )
    return embedding, np.asarray(frame.shape[:2], dtype=np.int64)


def _decode_candidates(
    embedding: np.ndarray,
    size: np.ndarray,
    box: tuple[float, float, float, float],
    decoder: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = box
    points = np.asarray([[[[left, top], [right, bottom]]]], dtype=np.float32)
    labels = np.asarray([[[2.0, 3.0]]], dtype=np.float32)
    outputs = decoder.run(
        ["output_masks", "iou_predictions"],
        {
            "image_embeddings": embedding,
            "batched_point_coords": points,
            "batched_point_labels": labels,
            "orig_im_size": size,
        },
    )
    return (
        cast("np.ndarray", outputs[0])[0, 0],
        cast("np.ndarray", outputs[1])[0, 0],
    )


def _predict_candidates(
    frame: np.ndarray,
    boxes: Sequence[tuple[float, float, float, float]],
    sessions: _Sessions,
) -> tuple[np.ndarray, np.ndarray]:
    encoder, decoder = sessions
    embedding, size = _encode_frame(frame, encoder)
    logits: list[np.ndarray] = []
    ious: list[np.ndarray] = []
    for box in boxes:
        candidate_logits, candidate_ious = _decode_candidates(embedding, size, box, decoder)
        logits.append(candidate_logits)
        ious.append(candidate_ious)
    return np.stack(logits), np.stack(ious)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _predict_masks(
    frame: np.ndarray,
    boxes: Sequence[tuple[float, float, float, float]],
    sessions: _Sessions,
) -> list[np.ndarray]:
    encoder, decoder = sessions
    embedding, size = _encode_frame(frame, encoder)
    masks: list[np.ndarray] = []
    for box in boxes:
        logits, ious = _decode_candidates(embedding, size, box, decoder)
        masks.append(_sigmoid(logits[int(np.argmax(ious))]))
    return masks


def execute_segment(
    image: object,
    detections: Sequence[object],
) -> tuple[list[Detection], list[np.ndarray]]:
    """Attach one full-frame EfficientSAM mask to each box-prompt detection."""
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

    soft_masks = [np.zeros((height, width), dtype=np.float32) for _ in items]
    if valid:
        predictions = _predict_masks(frame, [box for _, box in valid], _load_sessions())
        for mask, (index, _) in zip(predictions, valid, strict=True):
            soft_masks[index] = mask

    segmented = [
        Detection(item.label, item.score, item.region, mask)
        for item, mask in zip(items, soft_masks, strict=True)
    ]
    masks = [
        cast("np.ndarray", detection.mask).reshape(1, height, width) for detection in segmented
    ]
    return segmented, masks


__all__ = ["execute_segment"]
