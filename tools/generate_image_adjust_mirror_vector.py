"""Generate the deterministic image adjust mirror parity vector.

The committed vector (tests/fixtures/mirror-parity/image_adjust_v1.json) is
the golden corpus for the dinkster.image.adjust GLSL mirror. Input frames are
small deterministic synthetic images and expected outputs are the
authoritative CPU results of ``ImageAdjust.execute``. A conforming GLSL
runner must agree with every expected channel within the vector's declared
per-channel tolerance.

Values roundtrip losslessly through JSON: every recorded frame and
expected-output value is a binary32 float, binary32 widens exactly to
binary64, and the shortest-repr decimal of a binary64 parses back bit for
bit. Scalar case parameters and the tolerance are recorded as binary64 JSON
numbers, which roundtrip exactly too. The operations use only correctly
rounded IEEE-754 binary32 arithmetic (add, subtract, multiply, divide,
clamp), so regeneration is bit-stable on every platform and the committed
data must never change under a rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from dinkster_nodes_image.adjust import ADJUST_MIRROR_PER_CHANNEL_TOLERANCE, ImageAdjust

OUTPUT_PATH = (
    Path(__file__).parents[1] / "tests" / "fixtures" / "mirror-parity" / "image_adjust_v1.json"
)


class CaseSpec(TypedDict):
    id: str
    frame: str
    inputs: dict[str, object]


CASES: tuple[CaseSpec, ...] = (
    {"id": "invert-gradient", "frame": "gradient_rgb", "inputs": {"operation": "invert"}},
    {"id": "invert-out-of-range", "frame": "extremes_rgba", "inputs": {"operation": "invert"}},
    {
        "id": "invert-batch",
        "frame": "batch_single_channel",
        "inputs": {"operation": "invert"},
    },
    {"id": "normalize-defaults", "frame": "gradient_rgb", "inputs": {"operation": "normalize"}},
    {
        "id": "normalize-shifted",
        "frame": "extremes_rgba",
        "inputs": {"operation": "normalize", "mean": 0.4, "standard_deviation": 0.25},
    },
    {
        "id": "normalize-small-deviation",
        "frame": "gradient_rgb",
        "inputs": {"operation": "normalize", "mean": -2.0, "standard_deviation": 0.001},
    },
    {
        "id": "brightness-zero",
        "frame": "gradient_rgb",
        "inputs": {"operation": "brightness", "factor": 0.0},
    },
    {
        "id": "brightness-identity-clamps",
        "frame": "extremes_rgba",
        "inputs": {"operation": "brightness", "factor": 1.0},
    },
    {
        "id": "brightness-boost",
        "frame": "gradient_rgb",
        "inputs": {"operation": "brightness", "factor": 1.25},
    },
    {
        "id": "brightness-max",
        "frame": "extremes_rgba",
        "inputs": {"operation": "brightness", "factor": 10.0},
    },
    {
        "id": "brightness-batch",
        "frame": "batch_single_channel",
        "inputs": {"operation": "brightness", "factor": 1.25},
    },
    {
        "id": "contrast-flat",
        "frame": "gradient_rgb",
        "inputs": {"operation": "contrast", "factor": 0.0},
    },
    {
        "id": "contrast-boost",
        "frame": "extremes_rgba",
        "inputs": {"operation": "contrast", "factor": 1.25},
    },
    {
        "id": "contrast-max",
        "frame": "gradient_rgb",
        "inputs": {"operation": "contrast", "factor": 10.0},
    },
)


def build_frames() -> dict[str, np.ndarray]:
    gradient = np.linspace(0.0, 1.0, 60, dtype=np.float64).astype(np.float32)
    extremes = np.linspace(-0.25, 1.5, 48, dtype=np.float64).astype(np.float32)
    batch = np.linspace(0.05, 0.95, 12, dtype=np.float64).astype(np.float32)
    return {
        "gradient_rgb": gradient.reshape(1, 4, 5, 3),
        "extremes_rgba": extremes.reshape(1, 3, 4, 4),
        "batch_single_channel": batch.reshape(2, 2, 3, 1),
    }


def _array_record(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": [int(size) for size in array.shape],
        "values": [float(value) for value in array.reshape(-1)],
    }


def build_vector() -> dict[str, object]:
    frames = build_frames()
    cases: list[dict[str, object]] = []
    for spec in CASES:
        result = ImageAdjust.execute(
            image=frames[spec["frame"]], **cast("dict[str, Any]", spec["inputs"])
        )
        expected = cast("np.ndarray", result["image"])
        cases.append({**spec, "expected": _array_record(expected)})
    return {
        "format_version": 1,
        "node_type": "dinkster.image.adjust",
        "mirror_per_channel_tolerance": ADJUST_MIRROR_PER_CHANNEL_TOLERANCE,
        "frames": {name: _array_record(frame) for name, frame in frames.items()},
        "cases": cases,
    }


def main() -> None:
    OUTPUT_PATH.write_text(
        json.dumps(build_vector(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
