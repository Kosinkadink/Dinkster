"""CPU contracts for native Wan image conditioning."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_nodes_comfy.wan_image_conditioning import (
    WanFirstLastFrameToVideo,
    WanFunInpaintToVideo,
    WanImageToVideo,
)


class VAE:
    latent_channels = 7

    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def spacial_compression_encode(self) -> int:
        return 4

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return (
            image[::4, ::4, ::4, 0]
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, self.latent_channels, 1, 1, 1)
        )


def _conditioning(label: str) -> list[list[Any]]:
    return [[torch.tensor([[1.0]]), {"label": label}]]


def _image(values: list[float], channels: int = 4) -> torch.Tensor:
    return torch.stack(
        [torch.full((4, 6, channels), value, dtype=torch.float64) for value in values]
    )


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        image.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False
    ).movedim(1, -1)


def _vision(values: list[float]) -> dict[str, torch.Tensor]:
    return {
        "penultimate_hidden_states": torch.tensor(values, dtype=torch.float64).reshape(1, -1, 1)
    }


def test_wan_image_schema_and_empty_optional_inputs() -> None:
    schema = WanImageToVideo.define_schema()
    assert [input.id for input in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "start_image",
        "ref_pad_image",
    ]
    assert [output.id for output in schema.outputs] == ["positive", "negative", "latent"]
    assert schema.category == "model/conditioning/wan"
    assert [input.default for input in schema.inputs[3:7]] == [832, 480, 81, 1]
    assert schema.inputs[-1].doc == (
        "Fills the padding frames of the image conditioning with this image instead "
        "of gray, anchoring identity without pinning frames (SVI-style anti-drift "
        "padding, used by models such as ID-V2V)."
    )
    positive, negative = _conditioning("positive"), _conditioning("negative")
    output = WanImageToVideo.execute(
        positive=positive, negative=negative, vae=VAE(), width=16, height=8, length=9, batch_size=2
    )
    assert output["positive"] is positive and output["negative"] is negative
    assert output["latent"]["samples"].shape == (2, 16, 3, 1, 2)
    assert output["latent"]["samples"].dtype is torch.float32
    assert not output["latent"]["samples"].count_nonzero()


def test_wan_image_start_padding_ref_and_optional_clip_preserve_sources() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    start, ref = _image([1.0, 2.0], 5), _image([9.0, 10.0], 4)
    start_before, ref_before = start.clone(), ref.clone()
    vision = _vision([3.0, 5.0])
    vae = VAE()
    output = WanImageToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=8,
        length=9,
        batch_size=1,
        start_image=start,
        ref_pad_image=ref,
        clip_vision_output=vision,
    )
    resized_start, resized_ref = _resize(start, 8, 16), _resize(ref[:1], 8, 16)
    expected = torch.full((9, 8, 16, 5), 0.5, dtype=torch.float64)
    expected[:, :, :, :3] = resized_ref[:, :, :, :3]
    expected[:2] = resized_start
    assert vae.images[0].shape == (9, 8, 16, 3)
    torch.testing.assert_close(vae.images[0], expected[:, :, :, :3], rtol=0, atol=0)
    for conditioning, label in ((output["positive"], "positive"), (output["negative"], "negative")):
        metadata = conditioning[0][1]
        assert metadata["label"] == label
        torch.testing.assert_close(
            metadata["clip_vision_output"].penultimate_hidden_states,
            vision["penultimate_hidden_states"],
            rtol=0,
            atol=0,
        )
        assert metadata["concat_mask"].dtype is torch.float64
        assert torch.equal(
            metadata["concat_mask"][:, :, :, 0, 0],
            torch.tensor([[[0.0, 1.0, 1.0]]], dtype=torch.float64),
        )
    assert positive == _conditioning("positive") and negative == _conditioning("negative")
    assert torch.equal(start, start_before) and torch.equal(ref, ref_before)


def test_first_last_encodes_without_images_and_uses_source_temporal_masks() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    vae = VAE()
    output = WanFirstLastFrameToVideo.execute(
        positive=positive, negative=negative, vae=vae, width=16, height=8, length=9, batch_size=2
    )
    assert vae.images[0].shape == (9, 8, 16, 3)
    assert vae.images[0].dtype is torch.float32 and torch.equal(
        vae.images[0], torch.full_like(vae.images[0], 0.5)
    )
    samples = output["latent"]["samples"]
    assert samples.shape == (2, 7, 3, 2, 4) and samples.dtype is torch.float32
    mask = output["positive"][0][1]["concat_mask"]
    assert (
        mask.shape == (1, 4, 3, 2, 4)
        and mask.dtype is torch.float32
        and torch.equal(mask, torch.ones_like(mask))
    )


def test_first_last_asymmetric_overlap_masks_and_vision_order_do_not_mutate_inputs() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    start, end = _image([1.0, 2.0, 3.0], 3), _image([7.0, 8.0, 9.0, 10.0], 3)
    start_before, end_before = start.clone(), end.clone()
    first, last = _vision([11.0, 12.0]), _vision([21.0, 22.0, 23.0])
    vae = VAE()
    output = WanFirstLastFrameToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=8,
        length=5,
        batch_size=1,
        start_image=start,
        end_image=end,
        clip_vision_start_image=first,
        clip_vision_end_image=last,
    )
    resized_start, resized_end = _resize(start, 8, 16), _resize(end, 8, 16)
    expected = torch.full((5, 8, 16, 3), 0.5)
    expected[:3] = resized_start[:, :, :, :3]
    expected[-4:] = resized_end[:, :, :, :3]
    torch.testing.assert_close(vae.images[0], expected, rtol=0, atol=0)
    mask = output["positive"][0][1]["concat_mask"]
    expected_mask = torch.ones((1, 1, 8, 2, 4))
    expected_mask[:, :, :6] = 0.0
    expected_mask[:, :, -4:] = 0.0
    expected_mask = expected_mask.view(1, 2, 4, 2, 4).transpose(1, 2)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
    vision = output["positive"][0][1]["clip_vision_output"]
    assert vision is not first and vision is not last
    torch.testing.assert_close(
        vision.penultimate_hidden_states,
        torch.tensor([[[11.0], [12.0], [21.0], [22.0], [23.0]]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        first["penultimate_hidden_states"], _vision([11.0, 12.0])["penultimate_hidden_states"]
    )
    assert torch.equal(
        last["penultimate_hidden_states"], _vision([21.0, 22.0, 23.0])["penultimate_hidden_states"]
    )
    assert positive == _conditioning("positive") and negative == _conditioning("negative")
    assert torch.equal(start, start_before) and torch.equal(end, end_before)


def test_first_last_single_vision_identity_and_inpaint_delegation() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    vision = _vision([2.0])
    output = WanFirstLastFrameToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=16,
        height=8,
        length=1,
        batch_size=1,
        clip_vision_end_image=vision,
    )
    torch.testing.assert_close(
        output["positive"][0][1]["clip_vision_output"].penultimate_hidden_states,
        vision["penultimate_hidden_states"],
        rtol=0,
        atol=0,
    )
    inpaint = WanFunInpaintToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        width=16,
        height=8,
        length=1,
        batch_size=1,
        clip_vision_output=vision,
    )
    torch.testing.assert_close(
        inpaint["positive"][0][1]["clip_vision_output"].penultimate_hidden_states,
        vision["penultimate_hidden_states"],
        rtol=0,
        atol=0,
    )
    assert WanFunInpaintToVideo.define_schema().category == "model/conditioning/wan/fun inpaint"
