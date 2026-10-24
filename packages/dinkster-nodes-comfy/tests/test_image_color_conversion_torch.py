"""CPU behavior and schema coverage for image color conversion nodes."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.image_color_conversion import ImageColorSpace, ImageSDRToHDR


def test_color_conversion_schemas_preserve_node_ids_and_dynamic_algorithm_slot() -> None:
    hdr = ImageSDRToHDR.define_schema()
    color = ImageColorSpace.define_schema()

    assert hdr.node_type == "ImageSDRToHDR"
    assert hdr.combos[0].id == "algorithm"
    assert hdr.combos[0].options[0].key == "lottes"
    assert hdr.combos[0].options[0].inputs[0].id == "target_luminance"
    assert color.node_type == "ImageColorSpace"
    assert tuple(option.value for option in color.inputs[1].widget.options) == (
        "sRGB",
        "HDR",
        "HDR ACEScct",
        "HDR LogC3",
        "HDR PQ",
        "linear",
        "linear ACEScg",
    )


def test_color_conversion_preserves_straight_alpha_and_linear_extended_values() -> None:
    image = torch.tensor([[[[0.25, 0.5, 1.5, 0.3]]]], dtype=torch.float32)
    linear = ImageColorSpace.execute(image=image, source="sRGB", destination="linear")["IMAGE"]
    restored = ImageColorSpace.execute(image=linear, source="linear", destination="sRGB")["IMAGE"]
    hdr = ImageSDRToHDR.execute(
        image=image, algorithm={"algorithm": "lottes", "target_luminance": 650.0}
    )["IMAGE"]

    assert linear[0, 0, 0, 2] > 1.0
    torch.testing.assert_close(restored[..., 3], image[..., 3])
    torch.testing.assert_close(hdr[..., 3], image[..., 3])
    assert torch.isfinite(hdr).all()
