"""Camera frames use the reference schema and granted image decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_assets import MountDef, MountTable
from dinkster_nodes_image.webcam import WebcamCapture
from PIL import Image

from tools.compare_nodes import compare


def test_webcam_schema_matches_reference_except_frozen_id() -> None:
    reference = json.loads(
        (Path(__file__).parents[3] / "tools/data/comfy_core_reference.json").read_text()
    )
    row = next(item for item in reference["nodes"] if item["class_name"] == "WebcamCapture")
    schema = WebcamCapture.define_schema()
    assert schema.node_type == "dinkster.image.webcam_capture"
    assert "WebcamCapture" in schema.search_terms
    assert schema.aliases == ()
    wire = schema_to_wire(schema)
    wire["nodeType"] = "WebcamCapture"
    assert compare(row, wire) == []


def test_webcam_decodes_uploaded_frame_without_applying_browser_dimensions(tmp_path: Path) -> None:
    pixels = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    Image.fromarray(pixels).save(tmp_path / "frame.png")
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef("frames", tmp_path), kind="media/image")
    asset = table.bind_name("frame.png", "media/image")
    result: Any = WebcamCapture.execute(image=asset, width=400, height=200, capture_on_queue=False)[
        "IMAGE"
    ]
    assert result.shape == (1, 3, 5, 3)
    np.testing.assert_allclose(result[0], pixels.astype(np.float32) / 255, rtol=0, atol=1e-7)


def test_webcam_refuses_ungranted_paths() -> None:
    with pytest.raises(TypeError, match="image must be an AssetRef or LocalAssetRef"):
        WebcamCapture.execute(image="/tmp/camera.png")
