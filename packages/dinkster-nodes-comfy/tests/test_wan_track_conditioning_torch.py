"""CPU contracts for native Wan motion-track conditioning."""

from __future__ import annotations

from typing import Any

import numpy
import torch
from dinkster_nodes_comfy.wan_track_conditioning import (
    WanTrackToVideo,
    pad_pts,
    parse_json_tracks,
    patch_motion,
    process_tracks,
)


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        return image[::4, ::8, ::8, :1].movedim(-1, 0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning(label: str) -> list[list[Any]]:
    return [[torch.tensor([[1.0]]), {"label": label}]]


def _image(values: list[float]) -> torch.Tensor:
    return torch.stack([torch.full((8, 16, 3), value) for value in values])


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y}


def test_schema_uses_native_string_tracks_and_source_defaults() -> None:
    schema = WanTrackToVideo.define_schema()
    assert schema.category == "model/conditioning/wan/move"
    assert schema.search_terms == (
        "motion tracking",
        "trajectory video",
        "point tracking",
        "keypoint animation",
    )
    assert [input.id for input in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "tracks",
        "width",
        "height",
        "length",
        "batch_size",
        "temperature",
        "topk",
        "start_image",
        "clip_vision_output",
    ]
    assert [input.default for input in schema.inputs[3:10]] == ["[]", 832, 480, 81, 1, 220.0, 2]
    assert schema.inputs[3].widget.multiline is True
    assert schema.inputs[8].advanced and schema.inputs[9].advanced
    assert schema.inputs[10].required and not schema.inputs[11].required


def test_track_json_parsing_and_fixed_point_padding() -> None:
    single = "[{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]"
    list_strings = ["[{'x': 5, 'y': 6}]", "[{'x': 7, 'y': 8}]"]
    assert parse_json_tracks(single) == [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]]
    assert parse_json_tracks(list_strings) == [
        [{"x": 5, "y": 6}],
        [{"x": 7, "y": 8}],
    ]
    assert parse_json_tracks("not json") == []
    padded = pad_pts([_point(2, 3)])
    truncated = pad_pts([_point(float(index), float(index + 1)) for index in range(123)])
    assert padded.shape == (121, 1, 3) and numpy.array_equal(padded[0, 0], numpy.array([2, 3, 1]))
    assert not padded[1:].any()
    assert truncated.shape == (121, 1, 3) and numpy.array_equal(
        truncated[-1, 0], numpy.array([120, 121, 1], dtype=numpy.float32)
    )


def test_process_tracks_preserves_asymmetric_coordinates_and_visibility() -> None:
    source = numpy.stack(
        [
            pad_pts([_point(0, 0), _point(16, 8)]),
            pad_pts([_point(24, 4), _point(8, 6)]),
        ]
    )
    processed = process_tracks(source, (32, 8), 8)
    assert processed.shape == (9, 2, 4)
    torch.testing.assert_close(processed[0, :, 0], torch.full((2,), -1.0), rtol=0, atol=0)
    torch.testing.assert_close(processed[-1, :, 0], torch.full((2,), 0.7833333), rtol=0, atol=1e-7)
    torch.testing.assert_close(processed[0, 0, 1:3], torch.tensor([-4.0, -1.0]), rtol=0, atol=0)
    torch.testing.assert_close(processed[1, 1, 1:3], torch.tensor([-4.0, -1.0]), rtol=0, atol=0)
    torch.testing.assert_close(processed[0, :, 3], torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(processed[1:, :, 3], -torch.ones((8, 2)), rtol=0, atol=0)


def test_patch_motion_uses_visible_tracks_for_each_batch_and_topk() -> None:
    video = torch.zeros((2, 1, 3, 1, 2))
    video[0, 0, 0, 0] = torch.tensor([2.0, 6.0])
    video[1, 0, 0, 0] = torch.tensor([10.0, 14.0])
    tracks = torch.tensor(
        [
            [
                [
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[0.0, -1.0, -1.0, 1.0]],
                    [[0.5, -1.0, -1.0, 1.0]],
                    [[1.0, -1.0, -1.0, 1.0]],
                    [[1.0, -1.0, -1.0, 1.0]],
                    [[1.0, -1.0, -1.0, 1.0]],
                    [[1.0, -1.0, -1.0, 1.0]],
                    [[1.0, -1.0, -1.0, 1.0]],
                    [[1.0, -1.0, -1.0, 1.0]],
                ]
            ],
            [
                [
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, -1.0, 1.0]],
                ]
            ],
        ]
    )
    masks, features = patch_motion(tracks, video, temperature=1.0, topk=1)
    assert masks.shape == (2, 4, 3, 1, 2) and features.shape == video.shape
    torch.testing.assert_close(features[:, :, 0], video[:, :, 0], rtol=0, atol=0)
    near, far = torch.tensor(0.3678776), torch.tensor(0.00012341156)
    torch.testing.assert_close(features[0, 0, 1, 0], torch.stack([near, far]), rtol=0, atol=1e-7)
    torch.testing.assert_close(
        features[1, 0, 1, 0], torch.tensor([1.8393880, 0.0006170578]), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        masks[:, 0, 1, 0],
        torch.stack([torch.stack([near, far]), torch.stack([near, far])]),
        rtol=0,
        atol=1e-7,
    )


def test_empty_invalid_tracks_delegate_and_track_batches_materialize_inputs() -> None:
    positive, negative = _conditioning("positive"), _conditioning("negative")
    image = _image([2.0, 8.0])
    image_before = image.clone()
    vision = {"image_embeds": torch.tensor([[3.0]])}
    empty = WanTrackToVideo.execute(
        positive=positive,
        negative=negative,
        vae=VAE(),
        tracks="[]",
        width=16,
        height=8,
        length=9,
        batch_size=2,
        temperature=220.0,
        topk=2,
        start_image=image,
        clip_vision_output=vision,
    )
    assert empty["latent"]["samples"].shape == (2, 16, 3, 1, 2)
    assert "concat_latent_image" in empty["positive"][0][1]
    assert empty["positive"][0][1]["clip_vision_output"].image_embeds.equal(vision["image_embeds"])
    invalid = WanTrackToVideo.execute(
        positive=_conditioning("positive"),
        negative=_conditioning("negative"),
        vae=VAE(),
        tracks="not json",
        width=16,
        height=8,
        length=9,
        batch_size=1,
        temperature=220.0,
        topk=2,
    )
    assert invalid["latent"]["samples"].shape == (1, 16, 3, 1, 2)
    assert "concat_latent_image" not in invalid["positive"][0][1]

    tracks = (
        "[[[{'x': 0, 'y': 0}], [{'x': 16, 'y': 8}]], [[{'x': 16, 'y': 0}], [{'x': 0, 'y': 8}]]]"
    )
    vae = VAE()
    output = WanTrackToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        tracks=tracks,
        width=16,
        height=8,
        length=9,
        batch_size=3,
        temperature=220.0,
        topk=2,
        start_image=image,
    )
    assert len(vae.images) == 3
    assert output["latent"]["samples"].shape == (3, 16, 3, 1, 2)
    mask = output["positive"][0][1]["concat_mask"]
    assert mask.shape == (3, 4, 3, 1, 2)
    assert mask[:, :, 0].count_nonzero() == 0
    assert torch.equal(image, image_before)
    assert positive == _conditioning("positive") and negative == _conditioning("negative")
