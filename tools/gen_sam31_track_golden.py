"""Generate a SAM 3.1 video tracking vector with pinned ComfyUI code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_sam31_track_golden.py \
        /path/to/ComfyUI-at-8dc3f3f2 \
        /path/to/sam3.1_multiplex_fp16.safetensors \
        /path/to/neon_guitarist.png

Use Python 3.12 with the package versions recorded in the generated payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import PIL
import torch
from gen_sam31_golden import (
    BASELINE,
    IMAGE_SIZE,
    MODEL_SHA256,
    SOURCE_BASELINE,
    SOURCE_SHA256,
    SOURCE_URL,
    TORCH_NUM_THREADS,
    _check_reference,
    _load_models,
    _record,
    _segment,
    _sha256,
)
from PIL import Image
from torch.nn import functional as F

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "sam31_track_8dc3f3f.json"
BOXES = (
    (14.5, 17.0, 121.75, 128.0),
    (45.0, 40.0, 90.0, 120.0),
)


def _initial_masks(
    tracker: Any,
    interactive: list[torch.Tensor],
    boxes: torch.Tensor,
    source_size: tuple[int, int],
) -> torch.Tensor:
    masks = []
    for box in boxes:
        _, first = _segment(tracker, interactive, box=box.unsqueeze(0))
        _, refined = _segment(tracker, interactive, mask=first)
        restored = F.interpolate(
            refined,
            size=source_size,
            mode="bilinear",
            align_corners=False,
        )
        masks.append(restored > 0)
    return torch.cat(masks).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    model_path = args.model.resolve()
    source_path = args.source.resolve()
    _check_reference(reference)
    if _sha256(model_path) != MODEL_SHA256:
        raise SystemExit("model SHA-256 does not match the pinned SAM 3.1 artifact")
    if _sha256(source_path) != SOURCE_SHA256:
        raise SystemExit("source SHA-256 does not match the pinned workflow input")

    torch.set_num_threads(TORCH_NUM_THREADS)
    source = np.asarray(
        Image.open(source_path).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
    )
    frames = np.stack(
        (
            source,
            np.roll(source, shift=4, axis=1),
            np.roll(source, shift=8, axis=1),
        )
    )
    images = torch.from_numpy(frames.astype(np.float32) / 255.0).movedim(-1, 1)
    prepared = F.interpolate(
        images[:1],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
    )
    boxes = torch.tensor(BOXES, dtype=torch.float32).reshape(-1, 2, 2)
    boxes *= IMAGE_SIZE / 128
    backbone, tracker = _load_models(reference, model_path)
    tracker_module = sys.modules["comfy.ldm.sam3.tracker"]
    with torch.inference_mode():
        trunk = backbone.trunk(prepared)
        _, _, interactive, _ = backbone(
            prepared,
            tracker_mode="interactive",
            cached_trunk=trunk,
            tracker_only=True,
        )
        initial = _initial_masks(tracker, interactive, boxes, source.shape[:2])

        def backbone_fn(frame: torch.Tensor, frame_idx: int | None = None):
            del frame_idx
            frame_trunk = backbone.trunk(frame)
            _, _, features, positions = backbone(
                frame,
                tracker_mode="propagation",
                cached_trunk=frame_trunk,
                tracker_only=True,
            )
            return features, positions, frame_trunk

        result = tracker.track_video_with_detection(
            backbone_fn,
            images,
            initial,
            detect_fn=None,
            backbone_obj=backbone,
            target_device=torch.device("cpu"),
            target_dtype=torch.float32,
        )
        packed = cast("torch.Tensor", result["packed_masks"])
        tracked = tracker_module.unpack_masks(packed).float()
        restored = F.interpolate(
            tracked,
            size=source.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
        combined = F.interpolate(
            tracked.amax(dim=1, keepdim=True),
            size=source.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    document = {
        "baseline": BASELINE,
        "boxes": [list(box) for box in BOXES],
        "combinedMask": _record(combined.numpy(), dtype=np.dtype(np.float32)),
        "initialMasks": _record(initial.numpy(), dtype=np.dtype(np.float32)),
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "sourceFrames": _record(frames, dtype=np.dtype(np.uint8)),
        "sourceImageBaseline": SOURCE_BASELINE,
        "sourceImageSha256": SOURCE_SHA256,
        "sourceImageUrl": SOURCE_URL,
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
        "trackedMasks": _record(restored.numpy(), dtype=np.dtype(np.float32)),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
