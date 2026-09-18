"""Generate the deterministic image filter mirror parity vector.

The committed vector (tests/fixtures/mirror-parity/image_filter_v1.json) is
the golden corpus for the mirrored operations of dinkster.image.filter
(gaussian_blur and sharpen; the mirror's applies scope excludes the rest).
Input frames are small deterministic synthetic images and expected outputs
are the authoritative CPU results of ``ImageFilter.execute``. A conforming
GLSL runner must agree with every expected channel within the vector's
declared per-channel tolerance.

Frames roundtrip losslessly through JSON exactly as in the adjust corpus:
every frame value is a binary32 float from np.linspace, binary32 widens
exactly to binary64, and the shortest-repr decimal of a binary64 parses back
bit for bit. Unlike the adjust corpus, the expected outputs are NOT
guaranteed bit-stable under regeneration across platforms: the gaussian
kernel uses np.exp (libm/SIMD implementations differ by ulps) and the
convolution uses np.einsum (reduction order varies with SIMD width). The
parity test therefore compares regenerated expected outputs against the
committed data within EXPECTED_REGENERATION_ATOL instead of bit equality;
everything else in the vector must match exactly.

Corpus sharpen strengths stay at or below 1.25 (widget maximum is 5) so the
10 * strength amplification keeps regeneration drift far below both
EXPECTED_REGENERATION_ATOL and the declared mirror tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from dinkster_nodes_image.filters import FILTER_MIRROR_PER_CHANNEL_TOLERANCE, ImageFilter

OUTPUT_PATH = (
    Path(__file__).parents[1] / "tests" / "fixtures" / "mirror-parity" / "image_filter_v1.json"
)

# Bound on regenerate-vs-committed drift of expected outputs. Per-weight
# np.exp differences are ~1e-7 relative and 63-term binary32 einsum
# reductions move by a few ulps, so blur outputs drift ~1e-6 and the corpus
# sharpen cases (10 * strength <= 12.5) drift ~1e-5; 1e-4 gives an order of
# magnitude headroom while staying ~40x below the declared mirror tolerance.
EXPECTED_REGENERATION_ATOL = 1e-4


class CaseSpec(TypedDict):
    id: str
    frame: str
    inputs: dict[str, object]


CASES: tuple[CaseSpec, ...] = (
    {
        "id": "blur-radius-zero-copies",
        "frame": "gradient_rgb",
        "inputs": {"operation": "gaussian_blur", "radius": 0, "sigma": 1.0},
    },
    {
        "id": "blur-defaults",
        "frame": "gradient_rgb",
        "inputs": {"operation": "gaussian_blur", "radius": 1, "sigma": 1.0},
    },
    {
        "id": "blur-radius-exceeds-frame",
        "frame": "gradient_rgb",
        "inputs": {"operation": "gaussian_blur", "radius": 5, "sigma": 2.0},
    },
    {
        "id": "blur-near-delta-kernel",
        "frame": "gradient_rgb",
        "inputs": {"operation": "gaussian_blur", "radius": 3, "sigma": 0.1},
    },
    {
        "id": "blur-near-box-kernel-out-of-range",
        "frame": "extremes_rgba",
        "inputs": {"operation": "gaussian_blur", "radius": 3, "sigma": 10.0},
    },
    {
        "id": "blur-single-column-edge",
        "frame": "tall_column_rgb",
        "inputs": {"operation": "gaussian_blur", "radius": 2, "sigma": 1.0},
    },
    {
        "id": "blur-single-row-edge",
        "frame": "wide_row_single_channel",
        "inputs": {"operation": "gaussian_blur", "radius": 2, "sigma": 1.0},
    },
    {
        "id": "blur-batch-multi-bounce",
        "frame": "batch_single_channel",
        "inputs": {"operation": "gaussian_blur", "radius": 5, "sigma": 1.5},
    },
    {
        "id": "sharpen-strength-zero-clamps",
        "frame": "extremes_rgba",
        "inputs": {"operation": "sharpen", "radius": 1, "sigma": 1.0, "strength": 0.0},
    },
    {
        "id": "sharpen-radius-zero-clamps",
        "frame": "extremes_rgba",
        "inputs": {"operation": "sharpen", "radius": 0, "sigma": 1.0, "strength": 1.0},
    },
    {
        "id": "sharpen-defaults",
        "frame": "gradient_rgb",
        "inputs": {"operation": "sharpen", "radius": 1, "sigma": 1.0, "strength": 1.0},
    },
    {
        "id": "sharpen-corpus-max-strength",
        "frame": "gradient_rgb",
        "inputs": {"operation": "sharpen", "radius": 2, "sigma": 1.0, "strength": 1.25},
    },
    {
        "id": "sharpen-out-of-range",
        "frame": "extremes_rgba",
        "inputs": {"operation": "sharpen", "radius": 1, "sigma": 1.0, "strength": 0.5},
    },
)


def build_frames() -> dict[str, np.ndarray]:
    gradient = np.linspace(0.0, 1.0, 60, dtype=np.float64).astype(np.float32)
    extremes = np.linspace(-0.25, 1.5, 48, dtype=np.float64).astype(np.float32)
    column = np.linspace(0.1, 0.9, 18, dtype=np.float64).astype(np.float32)
    row = np.linspace(0.2, 0.8, 6, dtype=np.float64).astype(np.float32)
    batch = np.linspace(0.05, 0.95, 12, dtype=np.float64).astype(np.float32)
    return {
        "gradient_rgb": gradient.reshape(1, 4, 5, 3),
        "extremes_rgba": extremes.reshape(1, 3, 4, 4),
        "tall_column_rgb": column.reshape(1, 6, 1, 3),
        "wide_row_single_channel": row.reshape(1, 1, 6, 1),
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
        result = ImageFilter.execute(
            image=frames[spec["frame"]], **cast("dict[str, Any]", spec["inputs"])
        )
        expected = cast("np.ndarray", result["image"])
        cases.append({**spec, "expected": _array_record(expected)})
    return {
        "format_version": 1,
        "node_type": "dinkster.image.filter",
        "mirror_per_channel_tolerance": FILTER_MIRROR_PER_CHANNEL_TOLERANCE,
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
