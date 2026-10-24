"""Public ComfyUI compositing interfaces and CPU numerical regressions."""

from __future__ import annotations

import torch
from dinkster_api.v1 import schema_to_wire
from dinkster_inference import utils
from dinkster_nodes_comfy.compositing import (
    COMPOSITING_NODES,
    JoinImageWithAlpha,
    PorterDuffImageComposite,
    PorterDuffMode,
    SplitImageWithAlpha,
)


def test_compositing_schemas_match_the_fixed_reference_contract() -> None:
    expected = (
        (
            "PorterDuffImageComposite",
            "Porter-Duff Image Composite",
            ("source", "source_alpha", "destination", "destination_alpha", "mode"),
            ("IMAGE", "MASK"),
        ),
        ("SplitImageWithAlpha", "Split Image with Alpha", ("image",), ("IMAGE", "MASK")),
        ("JoinImageWithAlpha", "Join Image with Alpha", ("image", "alpha"), ("IMAGE",)),
    )
    actual = tuple(
        (
            node.schema().node_type,
            node.schema().display_name,
            tuple(input_spec.id for input_spec in node.schema().inputs),
            tuple(output_spec.id for output_spec in node.schema().outputs),
        )
        for node in COMPOSITING_NODES
    )
    assert actual == expected
    schema = PorterDuffImageComposite.schema()
    assert schema.inputs[-1].default == "DST"
    assert tuple(option.value for option in schema.inputs[-1].widget.options) == tuple(
        mode.name for mode in PorterDuffMode
    )
    for node in COMPOSITING_NODES:
        assert schema_to_wire(node.schema())


def test_porter_duff_source_over_inverts_asymmetric_transparency_masks() -> None:
    source = torch.tensor([[[[0.8, 0.2, 0.4]]]], dtype=torch.float32)
    destination = torch.tensor([[[[0.1, 0.6, 0.3]]]], dtype=torch.float32)
    source_alpha = torch.tensor([[[0.75]]], dtype=torch.float32)
    destination_alpha = torch.tensor([[[0.25]]], dtype=torch.float32)

    result = PorterDuffImageComposite.execute(
        source=source,
        source_alpha=source_alpha,
        destination=destination,
        destination_alpha=destination_alpha,
        mode="SRC_OVER",
    )

    source_opacity = 1.0 - source_alpha[..., None]
    destination_opacity = 1.0 - destination_alpha[..., None]
    output_opacity = source_opacity + (1.0 - source_opacity) * destination_opacity
    expected_image = (
        source * source_opacity + (1.0 - source_opacity) * destination * destination_opacity
    ) / output_opacity
    torch.testing.assert_close(result["IMAGE"], expected_image)
    torch.testing.assert_close(result["MASK"], 1.0 - output_opacity[..., 0])


def test_porter_duff_truncates_batches_and_resizes_images_and_masks() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3) / 24.0
    source_alpha = torch.tensor(
        [[[0.0, 0.25], [0.5, 0.75]], [[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32
    )
    destination = torch.full((1, 3, 4, 3), 0.5, dtype=torch.float32)
    destination_alpha = torch.tensor([[[0.1, 0.4], [0.6, 0.9]]], dtype=torch.float32)

    result = PorterDuffImageComposite.execute(
        source=source,
        source_alpha=source_alpha,
        destination=destination,
        destination_alpha=destination_alpha,
        mode="SRC",
    )

    expected_image = utils.common_upscale(
        source[:1].movedim(-1, 1), 4, 3, "bicubic", "center"
    ).movedim(1, -1)
    expected_image = torch.clamp(expected_image, 0, 1)
    expected_mask = utils.common_upscale(
        source_alpha[:1].unsqueeze(1), 4, 3, "bicubic", "center"
    ).squeeze(1)
    assert result["IMAGE"].shape == (1, 3, 4, 3)
    assert result["MASK"].shape == (1, 3, 4)
    torch.testing.assert_close(result["IMAGE"], expected_image)
    torch.testing.assert_close(result["MASK"], expected_mask)


def test_split_and_join_preserve_alpha_polarity_across_resized_batches() -> None:
    image = torch.tensor(
        [
            [[[0.1, 0.2, 0.3, 0.25], [0.4, 0.5, 0.6, 0.75]]],
            [[[0.7, 0.8, 0.9, 0.5], [0.2, 0.3, 0.4, 1.0]]],
        ],
        dtype=torch.float32,
    )
    split = SplitImageWithAlpha.execute(image=image)
    torch.testing.assert_close(split["IMAGE"], image[..., :3])
    torch.testing.assert_close(split["MASK"], 1.0 - image[..., 3])

    alpha = torch.tensor([[[0.0], [1.0]]], dtype=torch.float32)
    joined = JoinImageWithAlpha.execute(image=split["IMAGE"], alpha=alpha)["IMAGE"]
    expected_alpha = 1.0 - torch.nn.functional.interpolate(
        alpha.reshape(1, 1, 2, 1), size=(1, 2), mode="bilinear"
    ).squeeze(1)
    expected_alpha = utils.repeat_to_batch_size(expected_alpha, 2)
    torch.testing.assert_close(joined[..., :3], image[..., :3])
    torch.testing.assert_close(joined[..., 3], expected_alpha)
