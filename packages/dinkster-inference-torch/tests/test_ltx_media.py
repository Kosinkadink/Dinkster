"""Content-owned LTX media-conditioning contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
from dinkster_inference import (
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ExtensionInputValue,
    MultiStreamLatent,
    PayloadDescriptor,
    PayloadReference,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    decode_conditioning_carrier,
    encode_conditioning_carrier,
    make_conditioning_carrier,
)
from dinkster_inference_torch.ltx_media import (
    LTXMediaError,
    ltxav_reference_audio_conditioning,
    ltxv_add_guide,
    ltxv_condition_initial_frames,
    ltxv_crop_guides,
    materialize_ltxav_reference_audio,
    materialize_ltxv_guides,
)
from dinkster_inference_torch.payloads import tensor_to_payload_binding


def _carrier(family_id: str) -> ConditioningCarrier:
    role = "t5xxl" if family_id == "dinkster.ltxv" else "gemma3_12b"
    text = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    binding = tensor_to_payload_binding("text", text, space="conditioning-text")
    descriptor = PayloadDescriptor(
        PayloadReference(binding.reference_id), binding.shape, binding.dtype, binding.space
    )
    record = ConditioningRecord(
        channels=((ConditioningChannel.TEXT, descriptor),),
        token_layout=TokenLayoutDescriptor(
            family_id,
            1,
            (role,),
            (TokenSegmentDescriptor(role, role, 0, 4),),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (binding,))


def _video(frames: int = 3) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent.from_pairs((("video", torch.zeros((1, 4, frames, 2, 2))),))


def test_initial_frames_replace_the_latent_and_build_the_reference_mask() -> None:
    latent = _video()
    encoded = torch.full((1, 4, 1, 2, 2), 7.0)

    conditioned, mask = ltxv_condition_initial_frames(latent, encoded, strength=0.75)

    assert torch.equal(conditioned.by_role("video")[:, :, :1], encoded)
    assert torch.count_nonzero(conditioned.by_role("video")[:, :, 1:]) == 0
    expected = torch.ones((1, 1, 3, 2, 2))
    expected[:, :, :1] = 0.25
    assert torch.equal(mask.by_role("video"), expected)
    assert torch.count_nonzero(latent.by_role("video")) == 0


def test_guides_preserve_order_real_frame_coordinates_and_crop_cleanly() -> None:
    positive = _carrier("dinkster.ltxv")
    negative = _carrier("dinkster.ltxv")
    first = ltxv_add_guide(
        positive,
        negative,
        _video(),
        torch.ones((1, 4, 1, 2, 2)),
        frame_index=0,
        strength=0.5,
    )
    final = ltxv_add_guide(
        first.positive,
        first.negative,
        first.latent,
        torch.full((1, 4, 1, 2, 2), 2.0),
        frame_index=-1,
        strength=2.0,
        denoise_mask=first.denoise_mask,
    )

    assert final.latent.by_role("video").shape == (1, 4, 5, 2, 2)
    assert torch.equal(
        final.denoise_mask.by_role("video")[:, :, -2:],
        torch.tensor([0.5, 0.0]).reshape(1, 1, 2, 1, 1).expand(1, 1, 2, 2, 2),
    )
    serialized = encode_conditioning_carrier(final.positive)
    restored = decode_conditioning_carrier(serialized)
    stripped, guides = materialize_ltxv_guides(restored)
    assert len(guides) == 2
    assert guides[0].keyframe_indices[0, 0, 0].tolist() == [0, 1]
    assert guides[1].keyframe_indices[0, 0, 0].tolist() == [16, 17]
    assert guides[0].keyframe_indices[0, 1, :, 0].tolist() == [0, 0, 32, 32]
    assert not stripped.conditioning.records[0].extension_metadata
    assert len(stripped.bindings) == 1

    cropped = ltxv_crop_guides(
        final.positive,
        final.negative,
        final.latent,
        final.denoise_mask,
    )
    assert cropped.latent.by_role("video").shape == (1, 4, 3, 2, 2)
    assert cropped.denoise_mask.by_role("video").shape == (1, 1, 3, 2, 2)
    assert not cropped.positive.conditioning.records[0].extension_metadata
    assert not cropped.negative.conditioning.records[0].extension_metadata


def test_crop_guides_passes_through_matching_ltxav_without_guide_metadata() -> None:
    positive = _carrier("dinkster.ltxav")
    negative = _carrier("dinkster.ltxav")
    audio = torch.zeros((1, 8, 1, 16))
    positive, negative = ltxav_reference_audio_conditioning(positive, negative, audio)
    latent = _video()
    mask = torch.ones((1, 1, 3, 2, 2))

    cropped = ltxv_crop_guides(positive, negative, latent, mask)

    assert cropped.positive is positive
    assert cropped.negative is negative
    assert cropped.latent is latent
    assert torch.equal(cropped.denoise_mask.by_role("video"), mask)

    with pytest.raises(LTXMediaError, match="family 'dinkster.ltxav'"):
        ltxv_crop_guides(positive, _carrier("dinkster.ltxv"), latent, mask)

    record = positive.conditioning.records[0]
    malformed = make_conditioning_carrier(
        ConditioningSet(
            (
                replace(
                    record,
                    extension_metadata=(
                        *record.extension_metadata,
                        (
                            "dinkster-model-ltx/guides",
                            (),
                        ),
                    ),
                ),
            )
        ),
        positive.bindings,
    )
    with pytest.raises(LTXMediaError, match="guide metadata requires family 'dinkster.ltxv'"):
        ltxv_crop_guides(malformed, malformed, latent, mask)


def test_multiframe_guides_align_nonzero_real_frame_indices_to_the_causal_grid() -> None:
    result = ltxv_add_guide(
        _carrier("dinkster.ltxv"),
        _carrier("dinkster.ltxv"),
        _video(4),
        torch.ones((1, 4, 2, 2, 2)),
        frame_index=14,
    )

    _, guides = materialize_ltxv_guides(result.positive)
    temporal = guides[0].keyframe_indices[0, 0, :, 0].reshape(2, 2, 2)
    assert torch.equal(temporal[0], torch.full((2, 2), 9))
    assert torch.equal(temporal[1], torch.full((2, 2), 17))

    one_latent = ltxv_add_guide(
        _carrier("dinkster.ltxv"),
        _carrier("dinkster.ltxv"),
        _video(4),
        torch.ones((1, 4, 1, 2, 2)),
        frame_index=14,
        causal_fix=False,
    )
    _, one_latent_guides = materialize_ltxv_guides(one_latent.positive)
    assert one_latent_guides[0].keyframe_indices[0, 0, 0].tolist() == [9, 17]


def test_guide_attention_masks_are_content_owned_and_shape_checked() -> None:
    attention = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 64, 64)
    result = ltxv_add_guide(
        _carrier("dinkster.ltxv"),
        _carrier("dinkster.ltxv"),
        _video(),
        torch.ones((1, 4, 1, 2, 2)),
        frame_index=0,
        attention_mask=attention,
    )

    _, guides = materialize_ltxv_guides(result.positive)
    assert guides[0].attention_mask is not None
    assert torch.equal(guides[0].attention_mask, attention.unsqueeze(0).unsqueeze(0))

    noncausal_attention = torch.linspace(0.0, 1.0, 32 * 48).reshape(1, 32, 48)
    noncausal = ltxv_add_guide(
        _carrier("dinkster.ltxv"),
        _carrier("dinkster.ltxv"),
        _video(),
        torch.ones((1, 4, 1, 2, 2)),
        frame_index=14,
        attention_mask=noncausal_attention,
        causal_fix=False,
    )
    _, noncausal_guides = materialize_ltxv_guides(noncausal.positive)
    assert noncausal_guides[0].keyframe_indices[0, 0, 0].tolist() == [9, 17]
    assert noncausal_guides[0].attention_mask is not None
    assert torch.equal(
        noncausal_guides[0].attention_mask,
        noncausal_attention.unsqueeze(0).unsqueeze(0),
    )

    with pytest.raises(TypeError, match="nonempty floating"):
        ltxv_add_guide(
            _carrier("dinkster.ltxv"),
            _carrier("dinkster.ltxv"),
            _video(),
            torch.ones((1, 4, 1, 2, 2)),
            frame_index=0,
            attention_mask=torch.ones((1, 1, 32, 64)),
        )


def test_guides_require_matching_family_carriers_and_unit_denoise_masks() -> None:
    with pytest.raises(LTXMediaError, match="family"):
        ltxv_add_guide(
            _carrier("dinkster.ltxv"),
            _carrier("dinkster.ltxav"),
            _video(),
            torch.ones((1, 4, 1, 2, 2)),
            frame_index=0,
        )
    with pytest.raises(LTXMediaError, match=r"within \[0, 1\]"):
        ltxv_add_guide(
            _carrier("dinkster.ltxv"),
            _carrier("dinkster.ltxv"),
            _video(),
            torch.ones((1, 4, 1, 2, 2)),
            frame_index=0,
            denoise_mask=torch.full((1, 1, 3, 2, 2), 2.0),
        )


def test_guide_mutations_refuse_malformed_existing_metadata() -> None:
    carrier = _carrier("dinkster.ltxv")
    record = carrier.conditioning.records[0]
    malformed = make_conditioning_carrier(
        ConditioningSet(
            (
                replace(
                    record,
                    extension_metadata=(("dinkster-model-ltx/guides", ({"pre_filter_count": 4},)),),
                ),
            )
        ),
        carrier.bindings,
    )
    latent = _video()

    with pytest.raises(LTXMediaError, match="malformed"):
        ltxv_add_guide(
            malformed,
            malformed,
            latent,
            torch.ones((1, 4, 1, 2, 2)),
            frame_index=0,
        )
    with pytest.raises(LTXMediaError, match="malformed"):
        ltxv_crop_guides(
            malformed,
            malformed,
            latent,
            torch.ones((1, 1, 3, 2, 2)),
        )


@pytest.mark.parametrize(("bad_count", "spatial"), ((4.0, 2), (True, 1)))
def test_guide_mutations_refuse_serialized_non_integer_token_counts(
    bad_count: object, spatial: int
) -> None:
    latent = MultiStreamLatent.from_pairs((("video", torch.zeros((1, 4, 3, spatial, spatial))),))
    conditioned = ltxv_add_guide(
        _carrier("dinkster.ltxv"),
        _carrier("dinkster.ltxv"),
        latent,
        torch.ones((1, 4, 1, spatial, spatial)),
        frame_index=0,
    )
    record = conditioned.positive.conditioning.records[0]
    metadata = dict(record.extension_metadata)
    entries = cast(
        "tuple[dict[str, ExtensionInputValue], ...]", metadata["dinkster-model-ltx/guides"]
    )
    entry = dict(entries[0])
    entry["pre_filter_count"] = cast("ExtensionInputValue", bad_count)
    metadata["dinkster-model-ltx/guides"] = (entry,)
    malformed = make_conditioning_carrier(
        ConditioningSet((replace(record, extension_metadata=tuple(metadata.items())),)),
        conditioned.positive.bindings,
    )
    malformed = decode_conditioning_carrier(encode_conditioning_carrier(malformed))

    with pytest.raises(LTXMediaError, match="token count"):
        ltxv_add_guide(
            malformed,
            malformed,
            conditioned.latent,
            torch.ones((1, 4, 1, spatial, spatial)),
            frame_index=0,
            denoise_mask=conditioned.denoise_mask,
        )
    with pytest.raises(LTXMediaError, match="token count"):
        ltxv_crop_guides(
            malformed,
            malformed,
            conditioned.latent,
            conditioned.denoise_mask,
        )


def test_guide_mutations_require_existing_guides_to_match_latent_geometry() -> None:
    conditioned = ltxv_add_guide(
        _carrier("dinkster.ltxv"),
        _carrier("dinkster.ltxv"),
        _video(),
        torch.ones((1, 4, 1, 2, 2)),
        frame_index=0,
    )
    latent = MultiStreamLatent.from_pairs((("video", torch.zeros((1, 4, 4, 3, 3))),))
    mask = MultiStreamLatent.from_pairs((("video", torch.ones((1, 1, 4, 3, 3))),))

    with pytest.raises(LTXMediaError, match="latent geometry"):
        ltxv_add_guide(
            conditioned.positive,
            conditioned.negative,
            latent,
            torch.ones((1, 4, 1, 3, 3)),
            frame_index=0,
            denoise_mask=mask,
        )
    with pytest.raises(LTXMediaError, match="latent geometry"):
        ltxv_crop_guides(
            conditioned.positive,
            conditioned.negative,
            latent,
            mask,
        )


def test_reference_audio_round_trips_and_duplicate_overlay_is_refused() -> None:
    positive = _carrier("dinkster.ltxav")
    negative = _carrier("dinkster.ltxav")
    audio = torch.arange(2 * 8 * 3 * 16, dtype=torch.float32).reshape(2, 8, 3, 16)

    positive, negative = ltxav_reference_audio_conditioning(positive, negative, audio)
    positive = decode_conditioning_carrier(encode_conditioning_carrier(positive))
    stripped, tokens = materialize_ltxav_reference_audio(positive, token_width=128)

    assert tokens is not None
    assert tokens.shape == (2, 3, 128)
    assert torch.equal(tokens, audio.permute(0, 2, 1, 3).reshape(2, 3, 128))
    assert not stripped.conditioning.records[0].extension_metadata
    assert len(stripped.bindings) == 1
    with pytest.raises(LTXMediaError, match="already has reference audio"):
        ltxav_reference_audio_conditioning(positive, negative, audio)


def test_reference_audio_refuses_malformed_or_foreign_overlays() -> None:
    carrier = _carrier("dinkster.ltxav")
    for malformed in (
        torch.ones((1, 8, 16)),
        torch.ones((1, 1, 2, 128)),
        torch.ones((1, 16, 2, 8)),
    ):
        with pytest.raises(TypeError, match=r"\[B,8,T,16\]"):
            ltxav_reference_audio_conditioning(carrier, carrier, malformed)

    record = carrier.conditioning.records[0]
    malformed = make_conditioning_carrier(
        ConditioningSet(
            (
                replace(
                    record,
                    extension_metadata=(("dinkster-model-ltx/reference-audio", "not-a-reference"),),
                ),
            )
        ),
        carrier.bindings,
    )
    with pytest.raises(LTXMediaError, match="malformed"):
        materialize_ltxav_reference_audio(malformed, token_width=128)
