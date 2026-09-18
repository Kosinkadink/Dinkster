"""Generate a Depth Anything V2 vector with pinned controlnet_aux code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_depth_anything_v2_golden.py \
        /path/to/controlnet_aux-at-59b1fc4 /path/to/depth_anything_v2_vitl.pth

Use Python 3.12 with the package versions recorded in the generated payload.
The reference import also requires torchvision.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BASELINE = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
MODEL_SHA256 = "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "depth_anything_v2_controlnet_aux_59b1fc4.json"


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
    cv2.circle(frame, (33, 42), 21, (230, 20, 80), -1)
    cv2.rectangle(frame, (70, 16), (116, 74), (15, 210, 130), -1)
    return frame


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _check_reference(reference: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != BASELINE:
        raise SystemExit(f"controlnet_aux must be checked out at {BASELINE}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise SystemExit("controlnet_aux checkout must be clean")


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
        raise SystemExit("model SHA-256 does not match the pinned Depth Anything V2 artifact")

    sys.path.insert(0, str(reference / "src"))
    from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
    from custom_controlnet_aux.depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.float().eval()
    source = _source()
    detector = DepthAnythingV2Detector(model, "depth_anything_v2_vitl.pth")
    output = detector(
        source,
        detect_resolution=64,
        output_type="np",
        max_depth=1,
    )

    document = {
        "baseline": BASELINE,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "source": _record(source),
        "output": _record(output),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
