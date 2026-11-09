from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_flux.flux_image_layout import Flux3LayoutPrompt, FluxKontextImageScale
from PIL import Image


@pytest.mark.parametrize(
    ("height", "width", "target_height", "target_width", "crop_left", "crop_right"),
    ((11, 17, 832, 1248, 0, 17), (50, 83, 800, 1328, 0, 83), (8, 32, 672, 1568, 7, 25)),
)
def test_flux_image_scale_preserves_preferred_geometry_and_lanczos_pixels(
    height: int, width: int, target_height: int, target_width: int, crop_left: int, crop_right: int
) -> None:
    image = torch.linspace(-0.2, 1.2, height * width * 3).reshape(1, height, width, 3)
    source = image.clone()
    scaled: Any = FluxKontextImageScale.execute(image=image.numpy())["IMAGE"]
    assert torch.equal(image, source)
    assert scaled.shape == (1, target_height, target_width, 3)
    pixels = np.clip(255 * source[0, :, crop_left:crop_right].numpy(), 0, 255).astype(np.uint8)
    expected = (
        np.asarray(
            Image.fromarray(pixels).resize((target_width, target_height), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        / 255
    )
    torch.testing.assert_close(scaled[0], torch.from_numpy(expected), rtol=0, atol=0)


def test_flux_layout_preserves_order_quoting_and_original_prompt() -> None:
    prompt = "A cafe  \n"
    elements = [
        {"bbox": [1, 2, 3, 4], "desc": "sign", "type": "text", "text": "Cafe"},
        {"bbox": [5, 6, 7, 8], "desc": "table", "type": "image"},
    ]
    original = copy.deepcopy(elements)
    result = Flux3LayoutPrompt.execute(prompt=prompt, element=elements)["prompt"]
    assert result == (
        'A cafe [{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": "sign \\"Cafe\\""}, '
        '{"id": "obj_2", "bbox": [5, 6, 7, 8], "desc": "table"}]'
    )
    assert elements == original
    assert Flux3LayoutPrompt.execute(prompt=prompt, element={})["prompt"] == prompt
    assert Flux3LayoutPrompt.execute(prompt=prompt, element=[])["prompt"] == prompt
