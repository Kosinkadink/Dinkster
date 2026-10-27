"""CPU contracts for generated keyframe pinning and image guide preparation."""

from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.ltx_guides import add_image_guide
from dinkster_nodes_comfy.ltx_keyframe_guides import (
    LTXVGeneratedKeyframesToGuides,
    LTXVSeparateGeneratedKeyframes,
)


class Vae:
    downscale_index_formula = (8, 2, 3)

    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []
        self.decoded: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.encoded.append(pixels.clone())
        indices = torch.arange(0, pixels.shape[0], 8)
        return pixels[indices, ::3, ::2, 0].unsqueeze(0).unsqueeze(0).repeat(1, 128, 1, 1, 1)

    def decode(self, samples: torch.Tensor) -> torch.Tensor:
        self.decoded.append(samples.clone())
        return samples[:, :3].permute(0, 2, 3, 4, 1).repeat_interleave(3, 2).repeat_interleave(2, 3)


def inputs(resize: bool = False) -> dict[str, Any]:
    return dict(
        positive=[[torch.ones(1), {"keep": "positive"}]],
        negative=[[torch.zeros(1), {"keep": "negative"}]],
        vae=Vae(),
        latent={"samples": torch.zeros(1, 128, 5, 2, 3), "extra": 17},
        keyframes={
            "samples": torch.stack(
                (
                    torch.full((128, 1 if resize else 2, 2 if resize else 3), 0.25),
                    torch.full((128, 1 if resize else 2, 2 if resize else 3), 0.75),
                ),
                dim=1,
            ).unsqueeze(0),
            "generated_keyframe_indices": [12, 4],
            "generated_keyframe_num_frames": 17,
        },
        strength=0.625,
    )


@pytest.mark.parametrize(
    "resize,override", [(False, ""), (False, "7, 32"), (True, ""), (True, "7 32")]
)
def test_generated_guides_order_time_remapping_and_preservation(
    resize: bool, override: str
) -> None:
    args = inputs(resize)
    video_before, frames_before = (
        args["latent"]["samples"].clone(),
        args["keyframes"]["samples"].clone(),
    )
    result = LTXVGeneratedKeyframesToGuides.execute(**args, override_frame_indices=override)
    latent = result["latent"]
    assert latent["samples"].shape == (1, 128, 7, 2, 3)
    assert torch.equal(latent["samples"][:, :, :5], video_before)
    assert torch.equal(latent["samples"][:, :, 5], torch.full((1, 128, 2, 3), 0.25))
    assert torch.equal(latent["samples"][:, :, 6], torch.full((1, 128, 2, 3), 0.75))
    assert torch.equal(latent["noise_mask"][:, :, :5], torch.ones(1, 1, 5, 1, 1))
    assert torch.equal(latent["noise_mask"][:, :, 5:], torch.full((1, 1, 2, 1, 1), 0.375))
    positions = [7, 32] if override else [24, 8]
    for name, label in (("positive", "positive"), ("negative", "negative")):
        metadata = result[name][0][1]
        assert metadata["keep"] == label
        assert metadata["keyframe_idxs"][0, 0, ::6].tolist() == [
            [index, index + 1] for index in positions
        ]
        assert [entry["latent_shape"] for entry in metadata["guide_attention_entries"]] == [
            [1, 2, 3]
        ] * 2
        assert [entry["pre_filter_count"] for entry in metadata["guide_attention_entries"]] == [
            6,
            6,
        ]
    assert torch.equal(args["latent"]["samples"], video_before)
    assert torch.equal(args["keyframes"]["samples"], frames_before)
    assert args["positive"][0][1] == {"keep": "positive"}
    assert args["keyframes"]["generated_keyframe_indices"] == [12, 4]
    if resize:
        assert args["vae"].decoded[0].shape == (2, 128, 1, 1, 2)
        assert len(args["vae"].encoded) == 2
    else:
        assert latent["extra"] == 17
        assert not args["vae"].decoded and not args["vae"].encoded


@pytest.mark.parametrize(
    "count,index,encoded_count,guide_count,start",
    [
        (10, 0, 9, 2, 0),
        (10, 7, 9, 1, 1),
        (10, -9, 9, 1, 17),
        (3, 7, 1, 1, 7),
    ],
)
def test_image_guide_causal_encoding_crop_and_negative_index(
    count, index, encoded_count, guide_count, start
) -> None:
    vae = Vae()
    images = torch.arange(count * 6 * 6 * 4.0).reshape(count, 6, 6, 4) / 1000
    before = images.clone()
    source = {"samples": torch.zeros(1, 128, 5, 2, 3)}
    positive, negative, latent = add_image_guide(
        [[torch.ones(1), {}]],
        [[torch.zeros(1), {}]],
        vae,
        source,
        images,
        index,
        1.5,
    )
    assert vae.encoded[0].shape == (encoded_count, 6, 6, 3)
    assert latent["samples"].shape[2] == 5 + guide_count
    assert torch.count_nonzero(latent["noise_mask"][:, :, 5:]) == 0
    assert positive[0][1]["keyframe_idxs"][0, 0, 0, 0].item() == start
    assert torch.equal(positive[0][1]["keyframe_idxs"], negative[0][1]["keyframe_idxs"])
    assert torch.equal(images, before)
    assert torch.count_nonzero(source["samples"]) == 0


@pytest.mark.parametrize("factor", [2, "2.4", "invalid", None])
def test_image_guide_sparse_factor_and_attention(factor) -> None:
    source = {"samples": torch.zeros(1, 128, 3, 4, 6)}
    attention = torch.arange(12.0).reshape(1, 3, 4)
    positive, _, latent = add_image_guide(
        [[torch.ones(1), {}]],
        [],
        Vae(),
        source,
        torch.ones(1, 12, 12, 3),
        0,
        0.25,
        attention_mask=attention,
        iclora_parameters={"reference_downscale_factor": factor},
    )
    sparse = factor in (2, "2.4")
    guide = latent["samples"][:, :, -1:]
    assert guide.shape[-2:] == (4, 6)
    if sparse:
        assert torch.count_nonzero(guide[..., 1::2, :]) == 0
        assert torch.equal(
            latent["noise_mask"][:, :, -1:, ::2, ::2], torch.full((1, 1, 1, 2, 3), 0.75)
        )
        assert latent["noise_mask"][0, 0, -1, 1, 1].item() == -1.25
    entry = positive[0][1]["guide_attention_entries"][0]
    assert entry["latent_shape"] == ([1, 2, 3] if sparse else [1, 4, 6])
    assert entry["pre_filter_count"] == 24
    assert torch.equal(entry["pixel_mask"], attention[None, None])


@pytest.mark.parametrize(
    "override,message",
    [
        ("1", "Provide exactly 2"),
        ("2,2", "same pixel"),
        ("0,3", "between 1 and 32"),
        ("3,x", "could not parse"),
    ],
)
def test_bad_override_rejects(override, message) -> None:
    args = inputs()
    with pytest.raises(ValueError, match=message):
        LTXVGeneratedKeyframesToGuides.execute(**args, override_frame_indices=override)
    assert torch.count_nonzero(args["latent"]["samples"]) == 0


def test_collapsed_temporal_positions_and_existing_metadata_reject() -> None:
    args = inputs()
    args["keyframes"]["generated_keyframe_indices"] = [1, 2]
    args["keyframes"]["generated_keyframe_num_frames"] = 129
    with pytest.raises(ValueError, match="collapsed"):
        LTXVGeneratedKeyframesToGuides.execute(**args)
    args["positive"][0][1]["generated_keyframes"] = {"exists": True}
    with pytest.raises(ValueError, match="still carries"):
        LTXVGeneratedKeyframesToGuides.execute(**args)


def test_image_guide_rejects_indivisible_canvas_before_encode() -> None:
    vae = Vae()
    with pytest.raises(ValueError, match="must be divisible"):
        add_image_guide(
            [],
            [],
            vae,
            {"samples": torch.zeros(1, 128, 3, 4, 5)},
            torch.ones(1, 12, 12, 3),
            0,
            1.0,
            iclora_parameters={"reference_downscale_factor": 2},
        )
    assert not vae.encoded


def separation_inputs(recorded_length: bool = True) -> dict[str, Any]:
    samples = torch.arange(2 * 128 * 7 * 2 * 3.0).reshape(2, 128, 7, 2, 3)
    indices = torch.arange(3 * 24 * 2).reshape(1, 3, 24, 2)
    record = dict(
        first_latent_frame=4,
        num_keyframes=2,
        frame_indices=[7, 16],
        guide_entry_index=1,
        tokens_per_frame=6,
    )
    if recorded_length:
        record["num_pixel_frames"] = 17

    def cond(label: str) -> Any:
        entries = [
            dict(pre_filter_count=6, latent_shape=[1, 2, 3], label=label),
            dict(pre_filter_count=12, latent_shape=[2, 2, 3]),
            dict(pre_filter_count=6, latent_shape=[1, 2, 3], after=True),
        ]
        return [
            [
                torch.ones(1),
                dict(
                    keyframe_idxs=indices.clone(),
                    guide_attention_entries=entries,
                    generated_keyframes=record,
                    keep=label,
                ),
            ]
        ]

    return dict(
        positive=cond("p"),
        negative=cond("n"),
        latent=dict(
            samples=samples, noise_mask=torch.arange(7.0).reshape(1, 1, 7, 1, 1), extra="keep"
        ),
    )


@pytest.mark.parametrize(
    "batched,recorded_length", [(False, False), (False, True), (True, False), (True, True)]
)
def test_separate_preserves_surrounding_guides_and_storage(
    batched: bool, recorded_length: bool
) -> None:
    args = separation_inputs(recorded_length)
    before = args["latent"]["samples"].clone()
    source_indices = args["positive"][0][1]["keyframe_idxs"].clone()
    args["negative"][0][1]["keyframe_idxs"] += 1000
    result = LTXVSeparateGeneratedKeyframes.execute(**args, keyframes_to_batch=batched)
    assert torch.equal(
        result["latent"]["samples"], torch.cat((before[:, :, :4], before[:, :, 6:]), dim=2)
    )
    frames = before[:, :, 4:6]
    expected = frames.movedim(2, 1).reshape(4, 128, 1, 2, 3) if batched else frames
    assert torch.equal(result["keyframes"]["samples"], expected)
    assert result["keyframes"]["generated_keyframe_indices"] == [7, 16]
    assert result["keyframes"]["generated_keyframe_num_frames"] == (17 if recorded_length else 25)
    assert result["latent"]["extra"] == "keep"
    assert result["latent"]["noise_mask"].flatten().tolist() == [0, 1, 2, 3, 6]
    for name, label in (("positive", "p"), ("negative", "n")):
        metadata = result[name][0][1]
        assert metadata["keep"] == label and metadata["generated_keyframes"] is None
        assert torch.equal(
            metadata["keyframe_idxs"],
            torch.cat((source_indices[:, :, :6], source_indices[:, :, 18:]), dim=2)
            + (1000 if label == "n" else 0),
        )
        assert len(metadata["guide_attention_entries"]) == 2
        assert metadata["guide_attention_entries"][0]["label"] == label
        assert metadata["guide_attention_entries"][1]["after"] is True
        assert args[name][0][1]["generated_keyframes"]["num_keyframes"] == 2
    result["keyframes"]["samples"].zero_()
    result["latent"]["samples"].zero_()
    assert torch.equal(args["latent"]["samples"], before)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("tokens_per_frame", 8, "rescaled"),
        ("num_keyframes", 8, "different latent"),
        ("guide_entry_index", 9, "rebuilt"),
    ],
)
def test_separate_invalid_record_rejects(field, value, message) -> None:
    args = separation_inputs()
    args["positive"][0][1]["generated_keyframes"][field] = value
    with pytest.raises(ValueError, match=message):
        LTXVSeparateGeneratedKeyframes.execute(**args)
