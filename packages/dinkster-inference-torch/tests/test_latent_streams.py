"""CPU proofs for generic multi-stream tensor packing."""

import pytest
import torch
from dinkster_inference import (
    MINIMAX_H3_AUDIO_MASK_MAPPING,
    MINIMAX_H3_VIDEO_MASK_MAPPING,
    LatentStream,
    MultiStreamLatent,
)
from dinkster_inference_torch import (
    content_mask_to_latent_mask,
    latent_mask_preview,
    latent_mask_to_content_mask,
    pack_latent_mask,
    pack_latent_streams,
    time_ranges_to_latent_mask,
    unpack_latent_streams,
)
from dinkster_values import (
    EncodedLatentTensor,
    EncodedMultiStreamLatent,
    decode_latent,
    encode_latent,
)


def value(video: torch.Tensor, audio: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))


def sentinel_value() -> MultiStreamLatent[torch.Tensor]:
    video = torch.arange(720, dtype=torch.float32).reshape(1, 24, 2, 3, 5)
    audio = torch.arange(448, dtype=torch.float32).reshape(1, 32, 2, 7)
    return value(video, audio)


def test_generic_pack_roundtrips_ordered_roles_and_shapes() -> None:
    original = sentinel_value()
    packed, layout = pack_latent_streams(original)
    assert layout.roles == ("video", "audio")
    assert layout.by_role("video").elements == 720
    assert layout.by_role("audio").offset == 720
    assert tuple(packed.shape) == (1, 1, 1168)
    restored = unpack_latent_streams(packed, layout)
    assert restored.roles == original.roles
    assert torch.equal(restored.by_role("video"), original.by_role("video"))
    assert torch.equal(restored.by_role("audio"), original.by_role("audio"))


def test_generic_pack_accepts_noncontiguous_strided_streams() -> None:
    video = torch.arange(720.0).reshape(1, 24, 2, 5, 3).transpose(3, 4)
    audio = torch.arange(448.0).reshape(1, 32, 7, 2).transpose(2, 3)
    restored = unpack_latent_streams(*pack_latent_streams(value(video, audio)))
    assert torch.equal(restored.by_role("video"), video)
    assert torch.equal(restored.by_role("audio"), audio)


def test_generic_mask_pack_matches_comfy_singular_missing_and_batch_repeat() -> None:
    latent = value(torch.empty(2, 3, 2, 3, 5), torch.empty(2, 4, 2, 7))
    singular = torch.zeros(1, 1, 3, 5)
    packed = pack_latent_mask(singular, latent)
    assert tuple(packed.shape) == (2, 1, 146)
    assert torch.count_nonzero(packed[..., :90]) == 0
    assert torch.count_nonzero(packed[..., 90:]) == 56 * 2

    audio_mask = torch.zeros(1, 1, 2, 7)
    structural = MultiStreamLatent.from_pairs((("audio", audio_mask),))
    packed = pack_latent_mask(structural, latent)
    assert torch.count_nonzero(packed[..., :90]) == 90 * 2
    assert torch.count_nonzero(packed[..., 90:]) == 0


def test_generic_mask_pack_preserves_full_rank_batch_and_channel_axes() -> None:
    latent = value(torch.empty(2, 3, 2, 3, 5), torch.empty(2, 4, 2, 7))
    audio_mask = torch.zeros(2, 4, 2, 7)
    audio_mask[0, 1, 0, 2] = 0.25
    audio_mask[1, 3, 1, 6] = 0.75

    packed = pack_latent_mask(MultiStreamLatent.from_pairs((("audio", audio_mask),)), latent)
    restored = unpack_latent_streams(packed, pack_latent_streams(latent)[1])

    assert torch.all(restored.by_role("video") == 1.0)
    assert torch.equal(restored.by_role("audio"), audio_mask)


def test_generic_mask_pack_rejects_oversized_full_rank_channels() -> None:
    latent = value(torch.empty(2, 3, 2, 3, 5), torch.empty(2, 4, 2, 7))
    audio_mask = torch.zeros(2, 5, 2, 7)

    with pytest.raises(ValueError, match="mask channels must not exceed"):
        pack_latent_mask(MultiStreamLatent.from_pairs((("audio", audio_mask),)), latent)


def test_h3_content_mask_reduces_exact_causal_groups_and_spatial_blocks() -> None:
    target = torch.empty(1, 24, 7, 2, 3)
    content = torch.zeros(22, 32, 48)
    content[0, :16, :16] = 0.25
    content[1:5, :16, :16] = torch.tensor((0.1, 0.2, 0.3, 0.4)).reshape(4, 1, 1)
    content[5:9, 16:, 32:] = torch.tensor((0.2, 0.4, 0.6, 0.8)).reshape(4, 1, 1)

    latent = content_mask_to_latent_mask(
        content,
        target,
        MINIMAX_H3_VIDEO_MASK_MAPPING,
        spatial_reduction="mean",
        temporal_reduction="mean",
    )

    assert tuple(latent.shape) == (7, 2, 3)
    assert latent.dtype == torch.float32
    assert latent[0, 0, 0].item() == pytest.approx(0.25)
    assert latent[1, 0, 0].item() == pytest.approx(0.25)
    assert latent[2, 1, 2].item() == pytest.approx(0.5)
    assert torch.count_nonzero(latent) == 3


@pytest.mark.parametrize(
    ("spatial_reduction", "temporal_reduction", "expected"),
    (
        ("max", "max", 1.0),
        ("min", "max", 0.0),
        ("max", "min", 0.25),
        ("max", "first", 0.25),
        ("max", "last", 1.0),
    ),
)
def test_content_mask_supports_declared_reductions(
    spatial_reduction: str,
    temporal_reduction: str,
    expected: float,
) -> None:
    target = torch.empty(1, 24, 2, 1, 1)
    content = torch.zeros(5, 16, 16)
    content[1, 0, 0] = 0.25
    content[2, 0, 0] = 0.5
    content[3, 0, 0] = 0.75
    content[4, 0, 0] = 1.0

    latent = content_mask_to_latent_mask(
        content,
        target,
        MINIMAX_H3_VIDEO_MASK_MAPPING,
        spatial_reduction=spatial_reduction,
        temporal_reduction=temporal_reduction,
    )

    assert latent[1, 0, 0].item() == pytest.approx(expected)


def test_h3_latent_mask_expands_back_to_exact_content_geometry() -> None:
    target = torch.empty(1, 24, 7, 2, 3)
    normalized = torch.zeros_like(target)
    normalized[:, :, 1, 0, 2] = 0.4
    normalized[:, :, 5, 1, 0] = 0.75

    content = latent_mask_to_content_mask(
        normalized,
        target,
        MINIMAX_H3_VIDEO_MASK_MAPPING,
    )

    assert tuple(content.shape) == (22, 32, 48)
    assert torch.all(content[1:5, :16, 32:] == 0.4)
    assert torch.all(content[17:18, 16:, :16] == 0.75)
    assert torch.count_nonzero(content) == 5 * 16 * 16


def test_audio_time_ranges_paint_fractional_role_mask() -> None:
    target = torch.empty(1, 32, 2, 80)
    mask = time_ranges_to_latent_mask(
        target,
        MINIMAX_H3_AUDIO_MASK_MAPPING,
        ((0.25, 0.5), (1.75, 2.0)),
        selected=0.4,
        unselected=0.1,
    )

    assert tuple(mask.shape) == (1, 2, 80)
    assert torch.all(mask[..., :10] == 0.1)
    assert torch.all(mask[..., 10:20] == 0.4)
    assert torch.all(mask[..., 20:70] == 0.1)
    assert torch.all(mask[..., 70:] == 0.4)


def test_audio_time_ranges_select_every_continuously_overlapped_frame() -> None:
    target = torch.empty(1, 32, 2, 4)
    within_first = time_ranges_to_latent_mask(
        target,
        MINIMAX_H3_AUDIO_MASK_MAPPING,
        ((0.001, 0.010),),
    )
    across_boundary = time_ranges_to_latent_mask(
        target,
        MINIMAX_H3_AUDIO_MASK_MAPPING,
        ((0.020, 0.026),),
    )

    assert torch.equal(within_first[0, 0], torch.tensor((1.0, 0.0, 0.0, 0.0)))
    assert torch.equal(across_boundary[0, 0], torch.tensor((1.0, 1.0, 0.0, 0.0)))


def test_audio_time_ranges_refuse_invalid_values_bounds_and_video_mappings() -> None:
    target = torch.empty(1, 32, 2, 80)
    with pytest.raises(ValueError, match=r"within \[0, 2\]"):
        time_ranges_to_latent_mask(
            target,
            MINIMAX_H3_AUDIO_MASK_MAPPING,
            ((1.0, 2.5),),
        )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        time_ranges_to_latent_mask(
            target,
            MINIMAX_H3_AUDIO_MASK_MAPPING,
            (),
            selected=float("nan"),
        )
    with pytest.raises(TypeError, match="time-last"):
        time_ranges_to_latent_mask(
            target,
            MINIMAX_H3_VIDEO_MASK_MAPPING,
            ((0.0, 0.5),),
        )


def test_latent_mask_preview_preserves_timeline_dimensions() -> None:
    video = torch.zeros(2, 3, 4, 5, 6)
    video[1, 2, 3, 4, 5] = 0.75
    audio = torch.zeros(2, 3, 2, 7)
    audio[0, 1, 1, 6] = 0.5

    video_preview = latent_mask_preview(video, video)
    audio_preview = latent_mask_preview(audio, audio)

    assert tuple(video_preview.shape) == (4, 5, 6)
    assert video_preview[3, 4, 5].item() == 0.75
    assert tuple(audio_preview.shape) == (1, 2, 7)
    assert audio_preview[0, 1, 6].item() == 0.5


def test_content_mask_conversion_refuses_shape_values_and_unknown_reductions() -> None:
    target = torch.empty(1, 24, 2, 1, 1)
    content = torch.zeros(5, 16, 16)
    with pytest.raises(ValueError, match="shape"):
        content_mask_to_latent_mask(content[:-1], target, MINIMAX_H3_VIDEO_MASK_MAPPING)
    content[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        content_mask_to_latent_mask(content, target, MINIMAX_H3_VIDEO_MASK_MAPPING)
    content[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="unknown spatial"):
        content_mask_to_latent_mask(
            content,
            target,
            MINIMAX_H3_VIDEO_MASK_MAPPING,
            spatial_reduction="sum",
        )


def test_generic_pack_rejects_mixed_dtype_batch_and_device() -> None:
    video = torch.empty(1, 24, 2, 3, 5)
    with pytest.raises(TypeError, match="dtype"):
        pack_latent_streams(value(video, torch.empty(1, 32, 2, 7, dtype=torch.float64)))
    with pytest.raises(ValueError, match="batch"):
        pack_latent_streams(value(video, torch.empty(2, 32, 2, 7)))
    with pytest.raises(ValueError, match="device"):
        pack_latent_streams(value(video, torch.empty(1, 32, 2, 7, device="meta")))


@pytest.mark.parametrize(
    "shape",
    ((1168,), (1, 1168), (2, 1, 1168), (1, 2, 1168), (1, 1, 1167)),
)
def test_generic_unpack_rejects_wrong_pack_shape(shape: tuple[int, ...]) -> None:
    _, layout = pack_latent_streams(sentinel_value())
    with pytest.raises(ValueError, match="shape"):
        unpack_latent_streams(torch.empty(shape), layout)


def test_torch_multistream_codec_roundtrip_preserves_raw_storage() -> None:
    original = sentinel_value()
    decoded = decode_latent(encode_latent({"samples": original}))
    assert isinstance(decoded, dict)
    streams = decoded["samples"]
    assert type(streams) is EncodedMultiStreamLatent
    assert streams.roles == ("video", "audio")
    for (_, record), source in zip(streams.streams, original.streams, strict=True):
        assert type(record) is EncodedLatentTensor
        assert record.dtype == "float32"
        assert record.shape == tuple(source.payload.shape)
        assert len(record.data) == source.payload.numel() * source.payload.element_size()
        assert record.data == bytes(source.payload.untyped_storage())


def test_torch_latent_codec_preserves_bfloat16_offset_view() -> None:
    source = torch.arange(12, dtype=torch.bfloat16)[2:10]
    decoded = decode_latent(encode_latent({"samples": source}))
    assert isinstance(decoded, dict)
    record = decoded["samples"]
    assert type(record) is EncodedLatentTensor
    assert record.dtype == "bfloat16"
    assert record.shape == (8,)
    item_size = source.element_size()
    start = source.storage_offset() * item_size
    stop = start + source.numel() * item_size
    assert record.data == bytes(source.untyped_storage())[start:stop]
