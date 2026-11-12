"""Reference wire schema and observable image-document composition."""

from __future__ import annotations

from typing import Any

import numpy as np
from dinkster_image.image_compositor import ImageCompositor


def test_compositor_places_layers_and_rejects_stale_editor_changes() -> None:
    red = np.zeros((1, 2, 3, 3), dtype=np.float32)
    red[..., 0] = 1
    green = np.zeros((1, 1, 2, 3), dtype=np.float32)
    green[..., 1] = 1
    layers = {
        "version": 1,
        "canvas": (5, 3),
        "layers": [
            {"image": red, "x": 1, "y": 1, "z_index": 0},
            {"image": green, "x": 2, "y": 1, "z_index": 1},
        ],
    }
    result: Any = ImageCompositor.execute(layers=layers)
    image, mask = np.asarray(result["IMAGE"]), np.asarray(result["MASK"])
    assert image.shape[:3] == (1, 3, 5)
    np.testing.assert_array_equal(image[0, 1, 1, :3], [1, 0, 0])
    np.testing.assert_array_equal(image[0, 1, 2:4, :3], [[0, 1, 0], [0, 1, 0]])
    np.testing.assert_array_equal(mask[0], [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]])
    stale: Any = ImageCompositor.execute(
        layers=layers,
        compositor={
            "version": 2,
            "documentDigest": "blake3:" + "0" * 64,
            "commands": [{"op": "remove", "id": "not-a-layer"}],
        },
    )
    np.testing.assert_array_equal(stale["IMAGE"], image)
    np.testing.assert_array_equal(stale["MASK"], mask)
