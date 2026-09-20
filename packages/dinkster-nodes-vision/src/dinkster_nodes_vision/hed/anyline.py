"""AnyLine combines MTEED with one of four line-art extractors."""

from __future__ import annotations

from contextlib import nullcontext
from typing import cast

import cv2
import numpy as np
import torch

from .cache import MODEL_CACHE
from .lineart import execute_anime, execute_manga, execute_realistic
from .model import _frames, _hwc3, _resize_with_pad
from .teed import execute_teed

LINEART_KINDS = (
    "lineart_standard",
    "lineart_realisitic",
    "lineart_anime",
    "manga_line",
)


def _standard(image: object, resolution: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for frame in _frames(image):
        resized, target_height, target_width = _resize_with_pad(frame, resolution)
        source = resized.astype(np.float32)
        blurred = cv2.GaussianBlur(source, (0, 0), 2.0)
        intensity = np.clip(np.min(blurred - source, axis=2), 0.0, 255.0)
        selected = intensity[intensity > 3]
        divisor = max(16.0, float(np.median(selected))) if selected.size else 16.0
        result = np.clip(intensity / divisor * 127.0, 0.0, 255.0).astype(np.uint8)
        outputs.append(_hwc3(result[:target_height, :target_width]))
    return np.ascontiguousarray(np.asarray(outputs, dtype=np.float32) / 255.0)


def _remove_small_objects(image: np.ndarray, *, minimum_size: int, connectivity: int) -> np.ndarray:
    mask = image[:, :, 0].astype(bool).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=4 if connectivity == 1 else 8
    )
    keep = np.zeros(count, dtype=bool)
    keep[1:] = 3 * stats[1:, cv2.CC_STAT_AREA] > minimum_size
    cleaned = keep[np.asarray(labels, dtype=np.int32)]
    return image * cleaned[:, :, None]


def _lineart(image: object, kind: str, resolution: int) -> np.ndarray:
    if kind == "lineart_standard":
        return _standard(image, resolution)
    if kind == "lineart_realisitic":
        return execute_realistic(image, coarse=False, resolution=resolution)
    if kind == "lineart_anime":
        return execute_anime(image, resolution=resolution)
    if kind == "manga_line":
        return execute_manga(image, resolution=resolution)
    raise ValueError(f"unknown AnyLine merge extractor: {kind}")


def execute_anyline(
    image: object,
    *,
    merge_with_lineart: str,
    resolution: int,
    lineart_lower_bound: float,
    lineart_upper_bound: float,
    object_min_size: int,
    object_connectivity: int,
) -> np.ndarray:
    if merge_with_lineart not in LINEART_KINDS:
        raise ValueError(f"unknown AnyLine merge extractor: {merge_with_lineart}")
    if not 0.0 <= lineart_lower_bound <= 1.0:
        raise ValueError("lineart_lower_bound must be between 0 and 1")
    if not 0.0 <= lineart_upper_bound <= 1.0:
        raise ValueError("lineart_upper_bound must be between 0 and 1")
    if object_min_size < 1 or object_connectivity < 1:
        raise ValueError("object_min_size and object_connectivity must be positive")

    parking = (
        MODEL_CACHE.park("Anime Lineart", torch.device("cpu"))
        if merge_with_lineart == "lineart_anime"
        else nullcontext(False)
    )
    with parking:
        try:
            mteed = execute_teed(image, safe_steps=2, resolution=resolution, asset_id="mteed-model")
        finally:
            MODEL_CACHE.discard("MTEED")
    lineart = _lineart(image, merge_with_lineart, resolution)
    if mteed.shape != lineart.shape:
        raise ValueError(
            f"AnyLine extractors produced mismatched shapes: {mteed.shape} and {lineart.shape}"
        )
    outputs: list[np.ndarray] = []
    for base, top in zip(mteed, lineart, strict=True):
        intensity = top[:, :, 0]
        selected = np.where(
            (intensity >= lineart_lower_bound) & (intensity <= lineart_upper_bound),
            intensity,
            0.0,
        )
        selected_rgb = np.repeat(selected[:, :, None], 3, axis=2)
        selected_rgb = _remove_small_objects(
            selected_rgb,
            minimum_size=object_min_size,
            connectivity=object_connectivity,
        )
        mask = selected_rgb.astype(bool)
        union = 1.0 - (1.0 - selected_rgb) * (1.0 - base)
        outputs.append(base * ~mask + union * mask)
    return cast("np.ndarray", np.ascontiguousarray(np.asarray(outputs, dtype=np.float32)))


__all__ = ["LINEART_KINDS", "execute_anyline"]
