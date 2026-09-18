"""Generate HED parity vectors with the pinned controlnet_aux implementation.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_hed_golden.py \
        /path/to/controlnet_aux-at-59b1fc4 /path/to/ControlNetHED.pth

Use Python 3.12 with the package versions recorded in the generated payload.
The reference import also requires Pillow, einops, and huggingface_hub.
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
MODEL_BLAKE3 = "blake3:36ea9a81b5e5f69c9f98b81eacce0c70b7bb444af4d821201b8a910e05792da9"
MODEL_SHA256 = "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "hed_controlnet_aux_59b1fc4.json"


def _source() -> np.ndarray:
    frame = np.full((96, 128, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (15, 15), (70, 80), (0, 0, 0), -1)
    cv2.circle(frame, (95, 48), 25, (255, 0, 0), -1)
    cv2.line(frame, (0, 95), (127, 0), (0, 255, 0), 4)
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
    model_path = args.model
    _check_reference(reference)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise SystemExit("model SHA-256 does not match the pinned ControlNet HED artifact")

    sys.path.insert(0, str(reference / "src"))
    from custom_controlnet_aux.hed import ControlNetHED_Apache2, HEDdetector

    model = ControlNetHED_Apache2()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=True
    )
    detector = HEDdetector(model.float().eval()).to("cpu")
    source = _source()
    cases = {
        "soft": detector(
            source, detect_resolution=64, safe=False, scribble=False, output_type="np"
        ),
        "safe": detector(source, detect_resolution=64, safe=True, scribble=False, output_type="np"),
        "scribble": detector(
            source, detect_resolution=64, safe=True, scribble=True, output_type="np"
        ),
    }

    document = {
        "baseline": BASELINE,
        "modelBlake3": MODEL_BLAKE3,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "source": _record(source),
        "cases": {
            name: _record(np.asarray(output, dtype=np.uint8)) for name, output in cases.items()
        },
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
