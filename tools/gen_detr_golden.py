"""Generate a DETR ResNet-50 parity vector with pinned reference code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_detr_golden.py \
        /path/to/detr-at-29901c5 /path/to/detr-r50-e632da11.pth

Use Python 3.12 with the package versions recorded in the generated payload.
The reference import also requires torchvision and scipy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from golden_platform import cpu_identity

BASELINE = "29901c51d7fe8712168b8d0d64351170bc0f83e0"
MODEL_SHA256 = "e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc"
MODEL_BLAKE3 = "blake3:2bb221c9ab83ea68d6a66bdc4cfe7bce4c49a1784287f5923521d34d23d150a2"
TORCH_NUM_THREADS = 1
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "detr_r50_29901c5.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SHORTEST_SIDE = 800
LONGEST_SIDE = 1333


def _source() -> np.ndarray:
    height, width = 96, 128
    y, x = np.mgrid[:height, :width]
    frame = np.stack(
        (
            (x * 2 + y) % 256,
            (x + y * 3) % 256,
            ((x // 8) * 31 + (y // 8) * 17) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    disk = (x - 33) ** 2 + (y - 42) ** 2 <= 21**2
    frame[disk] = (230, 20, 80)
    frame[16:75, 70:117] = (15, 210, 130)
    return frame


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _float_record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "shape": list(contiguous.shape),
        "float32Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _prepare_frame(frame: np.ndarray) -> torch.Tensor:
    """Match dinkster_nodes_vision.detr.model.prepare_frame; divergence fails the
    provider's golden test."""
    height, width = frame.shape[:2]
    scale = SHORTEST_SIDE / min(height, width)
    if scale * max(height, width) > LONGEST_SIDE:
        scale = LONGEST_SIDE / max(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    mean = torch.tensor(IMAGENET_MEAN).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(3, 1, 1)
    return (tensor - mean) / std


def _check_reference(reference: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != BASELINE:
        raise SystemExit(f"detr must be checked out at {BASELINE}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise SystemExit("detr checkout must be clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    model_path = args.model.resolve()
    _check_reference(reference)
    with model_path.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if model_sha256 != MODEL_SHA256:
        raise SystemExit("model SHA-256 does not match the pinned DETR artifact")

    torch.set_num_threads(TORCH_NUM_THREADS)
    sys.path.insert(0, str(reference))
    from models.backbone import Backbone, Joiner
    from models.detr import DETR
    from models.position_encoding import PositionEmbeddingSine
    from models.transformer import Transformer

    backbone = Backbone("resnet50", train_backbone=True, return_interm_layers=False, dilation=False)
    joiner = Joiner(backbone, PositionEmbeddingSine(128, normalize=True))
    joiner.num_channels = backbone.num_channels
    transformer = Transformer(d_model=256, return_intermediate_dec=True)
    model = DETR(joiner, transformer, num_classes=91, num_queries=100)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.float().eval()

    source = _source()
    prepared = _prepare_frame(source.astype(np.float32) / 255.0)
    with torch.no_grad():
        outputs = model(prepared.unsqueeze(0))
    logits = outputs["pred_logits"][0].numpy()
    boxes = outputs["pred_boxes"][0].numpy()

    document = {
        "baseline": BASELINE,
        "generationCpu": cpu_identity(),
        "modelBlake3": MODEL_BLAKE3,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
        "source": _record(source),
        "logits": _float_record(logits),
        "boxes": _float_record(boxes),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
