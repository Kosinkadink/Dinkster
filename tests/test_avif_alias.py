"""Execution coverage for the ComfyUI advanced AVIF save alias."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from tests.test_video_alias_templates import _execute
from tests.test_video_io import _mount


@pytest.mark.parametrize(("save_mode", "asset_count"), [("still images", 3), ("animated", 1)])
def test_save_image_advanced_avif_alias_executes_still_and_animated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_mode: str,
    asset_count: int,
) -> None:
    root, _ = _mount(tmp_path, monkeypatch)
    images = np.zeros((3, 64, 64, 3), dtype=np.float32)
    images[0, ..., 0] = 0.8
    images[1, ..., 1] = 0.6
    images[2, ..., 2] = 0.4
    outputs, results = _execute(
        "SaveImageAdvanced",
        {
            "filename_prefix": "avif/alias",
            "format": "avif",
            "save_mode": save_mode,
            "bit_depth": "8-bit YUV420",
            "input_color_space": "sRGB",
            "crf": 21,
            "fps": 7.0,
            "loop_count": 3,
        },
        {"images": images},
    )
    assert outputs["_0_IMAGE_"] is images
    assets = results[""]["assets"]
    assert len(assets) == asset_count
    for asset in assets:
        assert asset.name.endswith(".avif")
        assert (root / "avif" / asset.name).is_file()
    with av.open(root / "avif" / assets[0].name) as container:
        sequence = max(container.streams.video, key=lambda stream: stream.frames)
        assert sequence.frames == (3 if save_mode == "animated" else 1)
        assert sequence.average_rate == (7 if save_mode == "animated" else 1)
